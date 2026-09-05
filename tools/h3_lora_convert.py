#!/usr/bin/env python3
"""diffusers 形式の MiniMax-H3 LoRA を ComfyUI 形式へ変換する。

対応表は **推測ではなく実測で確かめてある**。lightx2v が同じ LoRA を
diffusers 版と ComfyUI 版の両方で配っているので、それを突き合わせた
(2026-08-29):

  qkv  lora_A = [A_q; A_k; A_v] を縦に積む / lora_B は q,k,v の順の対角ブロック
       → 6 通りの並びを試して qkv だけが誤差 0.000000
  fc1  lora_B の **前半後半を入れ替える**（diffusers は [value; gate]、
       ComfyUI は [gate; value]）。入れ替えで誤差 0.0、そのままだと 0.0141

⚠ pruned/convrot の本体は **時刻埋め込みの経路が別物**。
  blocks.N.adaln_proj.linear.weight が [96768, 8]（adaln_t_table [1025,8] を引く）
  なのに対し、diffusers 側は入力 2688。**基底が違うので当てられない。**
  adaln / norm_out.linear / time_embedder の一族は落とす。落としたものは必ず表示する
  （黙って落とすと「全部当たった」ように見えてしまう）。

⚠ alpha は書かない。ComfyUI は alpha が無ければ scale=1.0 で当てる
  (comfy/weight_adapter/lora.py)。FastVideo の adapter は
  "W = W_base + lora_B @ lora_A" と自分で書いているので scale=1.0 が正しい。
"""
import argparse
import json
import re
import struct
import sys
from collections import Counter

import torch
from safetensors.torch import load_file, save_file

# 落とすもの（理由つき）。pruned 本体と基底が違う
DROP = [
    (re.compile(r"(^|\.)adaln_proj\.linear\."),
     "adaln の入力基底が違う（本体 8 次元 / adaln_t_table 経由）"),
    (re.compile(r"^norm_out\.linear\."),
     "final_layer.adaln_proj.linear も同じく 8 次元"),
    (re.compile(r"^time_embedder\."),
     "本体に時刻埋め込み層が無い（表に畳んである）"),
]

# 単発の名前替え（diffusers → ComfyUI）。形で確かめてある
TOP = {
    "proj_in": "video_patch_proj",
    "proj_out": "final_layer.video_out",
    "audio_proj_in": "audio_patch_proj",
    "audio_proj_out": "final_layer.audio_out",
    "context_embedder": "condition_proj",
    "norm_out.norm": "final_layer.norm",
}


def model_shapes(path):
    """本体の safetensors からヘッダだけ読む（21GB を開かない）。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        d = json.loads(f.read(n))
    d.pop("__metadata__", None)
    return {k: tuple(v["shape"]) for k, v in d.items()}


def _base(k):
    """テンソル名から lora_A/lora_B/diff などの語尾を落として、層の名前を返す。"""
    k = k.replace(".default.weight", ".weight")
    for suf in (".lora_A.weight", ".lora_B.weight", ".alpha",
                ".diff_b", ".diff"):
        if k.endswith(suf):
            return k[: -len(suf)], suf
    return k, ""


def _rename(mod):
    """層の名前を ComfyUI 側へ。qkv は呼び出し元で束ねるのでここでは触らない。"""
    mod = mod.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    mod = re.sub(r"^transformer_blocks\.", "blocks.", mod)
    mod = re.sub(r"\.attn\.to_out\.0$", ".attn.out_proj", mod)
    mod = re.sub(r"\.ff\.net\.0\.proj$", ".mlp.fc1", mod)
    mod = re.sub(r"\.ff\.net\.2$", ".mlp.fc2", mod)
    if mod in TOP:
        return TOP[mod]
    return mod


def convert(src, model=None, quiet=False):
    sd = load_file(src)
    shapes = model_shapes(model) if model else None

    out = {}
    dropped = Counter()
    qkv = {}          # 層 -> {"q":(A,B), "k":..., "v":...}
    notes = []

    for k, v in sd.items():
        for rx, why in DROP:
            if rx.search(k):
                dropped[why] += 1
                break
        else:
            mod, suf = _base(k)
            m = re.match(r"(.*)\.attn\.to_([qkv])$", mod)
            if m and suf in (".lora_A.weight", ".lora_B.weight"):
                stem = _rename(m.group(1) + ".attn")
                qkv.setdefault(stem, {}).setdefault(m.group(2), {})[suf] = v
                continue
            name = _rename(mod)
            if suf == ".lora_B.weight" and name.endswith(".mlp.fc1"):
                # diffusers は [value; gate]、ComfyUI は [gate; value]
                h = v.shape[0] // 2
                v = torch.cat([v[h:], v[:h]], 0)
            out["diffusion_model." + name + suf] = v

    # qkv を束ねる
    for stem, parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            notes.append(f"⚠ {stem}: q/k/v が揃っていない（{sorted(parts)}）")
            continue
        A = [parts[x][".lora_A.weight"] for x in "qkv"]
        B = [parts[x][".lora_B.weight"] for x in "qkv"]
        r = [a.shape[0] for a in A]
        big = torch.zeros(sum(b.shape[0] for b in B), sum(r), dtype=B[0].dtype)
        ro = co = 0
        for b, rr in zip(B, r):
            big[ro:ro + b.shape[0], co:co + rr] = b
            ro += b.shape[0]
            co += rr
        out[f"diffusion_model.{stem}.qkv_proj.lora_A.weight"] = torch.cat(A, 0)
        out[f"diffusion_model.{stem}.qkv_proj.lora_B.weight"] = big

    # 本体と形を突き合わせる
    bad = []
    if shapes:
        for k, v in out.items():
            mod, suf = _base(k[len("diffusion_model."):])
            tgt = shapes.get(mod + ".weight")
            tgtb = shapes.get(mod + ".bias")
            if suf == ".lora_A.weight" and tgt and v.shape[1] != tgt[1]:
                bad.append(f"{mod}: lora_A 入力 {v.shape[1]} ≠ 本体 {tgt[1]}")
            if suf == ".lora_B.weight" and tgt and v.shape[0] != tgt[0]:
                bad.append(f"{mod}: lora_B 出力 {v.shape[0]} ≠ 本体 {tgt[0]}")
            if suf == ".diff" and tgt and tuple(v.shape) != tgt:
                bad.append(f"{mod}: diff {tuple(v.shape)} ≠ 本体 {tgt}")
            if suf == ".diff_b" and tgtb and tuple(v.shape) != tgtb:
                bad.append(f"{mod}: diff_b {tuple(v.shape)} ≠ 本体 {tgtb}")
            if tgt is None and tgtb is None:
                bad.append(f"{mod}: 本体に無い")

    if not quiet:
        print(f"  読んだ    {len(sd)} 本")
        print(f"  書いた    {len(out)} 本")
        for why, n in dropped.items():
            print(f"  落とした  {n:4d} 本  — {why}")
        for n in notes:
            print("  " + n)
        for b in bad:
            print("  ⚠ " + b)
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--model", help="本体の safetensors。形を突き合わせる")
    ap.add_argument("--compare", help="公式の ComfyUI 版と突き合わせる（答え合わせ用）")
    a = ap.parse_args()

    out, bad = convert(a.src, a.model)

    if a.compare:
        ref = load_file(a.compare)
        ref = {k: v for k, v in ref.items() if not k.endswith(".alpha")}
        only_out = sorted(set(out) - set(ref))
        only_ref = sorted(set(ref) - set(out))
        print(f"\n  答え合わせ: こちら {len(out)} / 公式 {len(ref)}")
        for k in only_out[:8]:
            print("    こちらだけ:", k)
        for k in only_ref[:8]:
            print("    公式だけ  :", k)
        worst, wk = 0.0, None
        for k in set(out) & set(ref):
            d = (out[k].float() - ref[k].float()).abs().max().item()
            if d > worst:
                worst, wk = d, k
        print(f"    共通 {len(set(out) & set(ref))} 本の最大差 {worst:.8f}"
              + (f"  ({wk})" if worst else ""))
        if not only_out and not only_ref and worst == 0.0:
            print("    ✓ 公式の変換と完全一致")
        return 0 if (not only_out and not only_ref and worst == 0.0) else 1

    if bad:
        print("\n  形が合わないものがあるので書き出さない")
        return 1
    if a.dst:
        save_file(out, a.dst)
        print(f"\n  → {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
