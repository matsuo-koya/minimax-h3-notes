#!/usr/bin/env python3
"""Per-node timing for a ComfyUI workflow, from the gaps between `executing` websocket events.

ComfyUI's /history only reports the total for a prompt; this is the only way to see where the
time actually goes (sampler vs VAE decode vs save). Time spent *inside* the sampler is all
attributed to the sampler node — if you need step-level detail, accumulate the `progress` events
that arrive on the same socket in the same way.

What is measured is the interval between *client-side receipt* of events. That is fine for
"which node is slow"; for chasing a long silence, add server-side timestamps too so a stalled
connection and a stalled node cannot be confused.

Outcomes are kept apart: success, execution_error, execution_interrupted, and a socket that
closes before `executing(node=None)`. Only success returns a summary; the CLI exit code is 1
for anything else. Events of other prompts on the same socket are ignored.

Usage:
    python profile_nodes.py workflow_api.json [--label "t2v 1280x704"] [--comfy http://127.0.0.1:8188]

`workflow_api.json` is the API-format graph (the dict you would POST to /prompt), e.g. what
"Save (API Format)" writes from the ComfyUI menu.

Only dependency: aiohttp.
"""
import argparse
import asyncio
import json
import time
import uuid

import aiohttp


async def run(wf, label, comfy="http://127.0.0.1:8188"):
    """Returns (per-class seconds, total seconds) on success, None otherwise."""
    cid = uuid.uuid4().hex
    types = {k: v["class_type"] for k, v in wf.items()}
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"{comfy.replace('http', 'ws')}/ws?clientId={cid}", max_msg_size=0) as ws:
            r = await s.post(f"{comfy}/prompt", json={"prompt": wf, "client_id": cid})
            pid = (await r.json())["prompt_id"]
            t_submit = time.perf_counter()
            t_start = None                      # execution_start of our prompt
            cur = tcur = None
            spans = []
            outcome = "incomplete"              # socket closed before we saw the end
            async for m in ws:
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                d = json.loads(m.data)
                kind, data = d.get("type"), d.get("data") or {}
                if data.get("prompt_id") != pid:
                    continue                    # another job's events
                now = time.perf_counter()
                if kind == "execution_start":
                    t_start = now
                elif kind == "executing":
                    node = data.get("node")
                    if cur is not None:
                        spans.append((cur, now - tcur))
                    cur, tcur = node, now
                    if node is None:            # end of this prompt
                        outcome = "success"
                        break
                elif kind == "execution_success":
                    outcome = "success"
                    break
                elif kind == "execution_error":
                    print("ERROR", data.get("node_type"), data.get("exception_message"))
                    outcome = "error"
                    break
                elif kind == "execution_interrupted":
                    print("INTERRUPTED at node", data.get("node_id"))
                    outcome = "interrupted"
                    break
            t_end = time.perf_counter()
    if outcome != "success":
        print(f"\n== {label}  {outcome} after {t_end - t_submit:.1f}s")
        return None
    total = t_end - t_submit
    queue_wait = (t_start - t_submit) if t_start is not None else None
    print(f"\n== {label}  total {total:.1f}s"
          + (f"  (queue wait {queue_wait:.1f}s, execution {t_end - t_start:.1f}s)" if queue_wait is not None else ""))
    agg = {}
    for n, dt in spans:
        t = types.get(n, n)
        agg[t] = agg.get(t, 0.0) + dt
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {k:32} {v:7.2f}s  {100 * v / total:5.1f}%")
    return agg, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow", help="API-format workflow JSON")
    ap.add_argument("--label", default=None)
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    wf = json.load(open(a.workflow))
    res = asyncio.run(run(wf, a.label or a.workflow, a.comfy))
    return 0 if res is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
