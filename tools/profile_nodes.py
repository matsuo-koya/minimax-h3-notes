#!/usr/bin/env python3
"""Per-node timing for a ComfyUI workflow, from the gaps between `executing` websocket events.

ComfyUI's /history only reports the total for a prompt; this is the only way to see where the
time actually goes (sampler vs VAE decode vs save). Time spent *inside* the sampler is all
attributed to the sampler node — if you need step-level detail, accumulate the `progress` events
that arrive on the same socket in the same way.

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
    cid = uuid.uuid4().hex
    types = {k: v["class_type"] for k, v in wf.items()}
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"{comfy.replace('http', 'ws')}/ws?clientId={cid}", max_msg_size=0) as ws:
            r = await s.post(f"{comfy}/prompt", json={"prompt": wf, "client_id": cid})
            pid = (await r.json())["prompt_id"]
            t0 = time.time()
            cur = tcur = None
            spans = []
            async for m in ws:
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                d = json.loads(m.data)
                if d.get("type") == "executing" and d["data"].get("prompt_id") == pid:
                    now = time.time()
                    node = d["data"]["node"]
                    if cur is not None:
                        spans.append((cur, now - tcur))
                    cur, tcur = node, now
                    if node is None:          # end of this prompt
                        break
                elif d.get("type") == "execution_error":
                    print("ERROR", d["data"].get("exception_message"))
                    return None
            total = time.time() - t0
    print(f"\n== {label}  total {total:.1f}s")
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
    asyncio.run(run(wf, a.label or a.workflow, a.comfy))


if __name__ == "__main__":
    main()
