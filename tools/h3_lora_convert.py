#!/usr/bin/env python3
"""diffusers 形式の MiniMax-H3 LoRA を ComfyUI 形式へ変換する。

対応表は **推測ではなく実測で確かめてある**。lightx2v が同じ LoRA を
diffusers 版と ComfyUI 版の両方で配っているので、それを突き合わせた
(2026-08-29):

  qkv  lora_A = [A_q; A_k; A_v] を縦に積む / lora_B は q,k,v の順の対角ブロック
       → 6 通りの並びを試して qkv だけが誤差 0.000000
  fc1  lora_B の **前半後半を入れ替える**（diffusers は [value; gate]、
       ComfyUI は [gate; value]）。入れ替えで誤差 0.0、そのままだと 0.0141
       （密な差分 .diff / .diff_b も同じ行の並べ替えを受ける）

⚠ pruned/convrot の本体は **時刻埋め込みの経路が別物**。
  blocks.N.adaln_proj.linear.weight が [96768, 8]（adaln_t_table [1025,8] を引く）
  なのに対し、diffusers 側は入力 2688。**基底が違うので当てられない。**
  adaln / norm_out.linear / time_embedder の一族は落とす。落としたものは必ず表示する
  （黙って落とすと「全部当たった」ように見えてしまう）。

⚠ alpha は書かない。ComfyUI は alpha が無ければ scale=1.0 で当てる
  (comfy/weight_adapter/lora.py)。FastVideo の adapter は
  "W = W_base + lora_B @ lora_A" と自分で書いていて、adapter_manifest にも
  lora_alpha は無く強度の既定は 1.0 なので scale=1.0 が正しい。
  ⚠ 一般の PEFT LoRA（alpha/r や rsLoRA の alpha/√r を掛けるもの）は対象外。
  入力に .alpha があれば **変換を拒否する**（黙って scale を落とさない）。

変換は「閉じて失敗する」。次のどれかがあれば何も書き出さず終了コード 1:
  - q/k/v が揃わない層がある（形が合っても順序が違えば壊れる、の変換器版）
  - lora_A と lora_B の片方しか無い / 内側の rank が合わない
  - NaN や inf を含む
  - --model の本体と形が合わない、または本体に無い層
  - 変換結果が空
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

SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".alpha", ".diff_b", ".diff")


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
    for suf in SUFFIXES:
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


def _swap_halves(v):
    """fc1 の出力行を [value; gate] → [gate; value] に。"""
    h = v.shape[0] // 2
    return torch.cat([v[h:], v[:h]], 0)


def _fuse_qkv(stem, parts, out, bad):
    """to_q / to_k / to_v を qkv_proj に束ねる。揃っていなければエラー。"""
    kinds = {suf for p in parts.values() for suf in p}
    for suf in kinds:
        have = [x for x in "qkv" if suf in parts.get(x, {})]
        if have != list("qkv"):
            bad.append(f"{stem}: {suf} の q/k/v が揃っていない（{have}）")
            return
    if ".lora_A.weight" in kinds or ".lora_B.weight" in kinds:
        if not {".lora_A.weight", ".lora_B.weight"} <= kinds:
            bad.append(f"{stem}: lora_A と lora_B の片方しか無い")
            return
        A = [parts[x][".lora_A.weight"] for x in "qkv"]
        B = [parts[x][".lora_B.weight"] for x in "qkv"]
        for x, a, b in zip("qkv", A, B):
            if a.shape[0] != b.shape[1]:
                bad.append(f"{stem}.to_{x}: rank が合わない lora_A {tuple(a.shape)} / lora_B {tuple(b.shape)}")
                return
        r = [a.shape[0] for a in A]
        big = torch.zeros(sum(b.shape[0] for b in B), sum(r), dtype=B[0].dtype)
        ro = co = 0
        for b, rr in zip(B, r):
            big[ro:ro + b.shape[0], co:co + rr] = b
            ro += b.shape[0]
            co += rr
        out[f"diffusion_model.{stem}.qkv_proj.lora_A.weight"] = torch.cat(A, 0)
        out[f"diffusion_model.{stem}.qkv_proj.lora_B.weight"] = big
    for suf in (".diff", ".diff_b"):
        if suf in kinds:
            out[f"diffusion_model.{stem}.qkv_proj{suf}"] = torch.cat([parts[x][suf] for x in "qkv"], 0)


def convert(src, model=None, quiet=False):
    """戻り値は (変換結果, 問題の一覧)。問題が一つでもあれば書き出してはいけない。"""
    sd = load_file(src)
    shapes = model_shapes(model) if model else None

    out = {}
    dropped = Counter()
    qkv = {}          # 層 -> {"q": {suffix: tensor}, "k": ..., "v": ...}
    bad = []

    for k, v in sd.items():
        for rx, why in DROP:
            if rx.search(k):
                dropped[why] += 1
                break
        else:
            mod, suf = _base(k)
            if suf == "":
                bad.append(f"{k}: 知らない語尾（lora_A/lora_B/diff/diff_b 以外）")
                continue
            if suf == ".alpha":
                bad.append(f"{k}: alpha 付きの LoRA は対象外（scale=1.0 前提の変換器）")
                continue
            m = re.match(r"(.*)\.attn\.to_([qkv])$", mod)
            if m:
                stem = _rename(m.group(1) + ".attn")
                qkv.setdefault(stem, {}).setdefault(m.group(2), {})[suf] = v
                continue
            name = _rename(mod)
            if name.endswith(".mlp.fc1") and suf in (".lora_B.weight", ".diff", ".diff_b"):
                # diffusers は [value; gate]、ComfyUI は [gate; value]（出力行の並べ替え）
                v = _swap_halves(v)
            out["diffusion_model." + name + suf] = v

    for stem, parts in qkv.items():
        _fuse_qkv(stem, parts, out, bad)

    # lora_A / lora_B の対と rank
    mods = {}
    for k in out:
        mod, suf = _base(k)
        mods.setdefault(mod, set()).add(suf)
    for mod, sufs in sorted(mods.items()):
        if (".lora_A.weight" in sufs) != (".lora_B.weight" in sufs):
            bad.append(f"{mod}: lora_A と lora_B の片方しか無い")
        elif ".lora_A.weight" in sufs:
            a, b = out[mod + ".lora_A.weight"], out[mod + ".lora_B.weight"]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                bad.append(f"{mod}: rank が合わない lora_A {tuple(a.shape)} / lora_B {tuple(b.shape)}")

    # 値の健全性
    for k, v in out.items():
        if not torch.isfinite(v.float()).all():
            bad.append(f"{k}: NaN / inf を含む")

    # 本体と形を突き合わせる
    if shapes:
        missing = set()
        for k, v in out.items():
            mod, suf = _base(k[len("diffusion_model."):])
            tgt = shapes.get(mod + ".weight")
            tgtb = shapes.get(mod + ".bias")
            if tgt is None and tgtb is None:
                if mod not in missing:
                    missing.add(mod)
                    bad.append(f"{mod}: 本体に無い")
                continue
            if suf == ".lora_A.weight" and tgt and v.shape[1] != tgt[1]:
                bad.append(f"{mod}: lora_A 入力 {v.shape[1]} ≠ 本体 {tgt[1]}")
            if suf == ".lora_B.weight" and tgt and v.shape[0] != tgt[0]:
                bad.append(f"{mod}: lora_B 出力 {v.shape[0]} ≠ 本体 {tgt[0]}")
            if suf == ".diff" and tgt and tuple(v.shape) != tgt:
                bad.append(f"{mod}: diff {tuple(v.shape)} ≠ 本体 {tgt}")
            if suf == ".diff_b" and tgtb and tuple(v.shape) != tgtb:
                bad.append(f"{mod}: diff_b {tuple(v.shape)} ≠ 本体 {tgtb}")

    if not out:
        bad.append("変換結果が空")

    if not quiet:
        print(f"  読んだ    {len(sd)} 本")
        print(f"  書いた    {len(out)} 本")
        for why, n in dropped.items():
            print(f"  落とした  {n:4d} 本  — {why}")
        for b in bad:
            print("  ⚠ " + b)
    return out, bad


def compare(out, ref_path, quiet=False):
    """公式の ComfyUI 版と突き合わせる。キー・形・有限性を見てから差を取る。戻り値は問題の一覧。"""
    ref = load_file(ref_path)
    ref = {k: v for k, v in ref.items() if not k.endswith(".alpha")}
    problems = []
    only_out = sorted(set(out) - set(ref))
    only_ref = sorted(set(ref) - set(out))
    problems += [f"こちらだけ: {k}" for k in only_out]
    problems += [f"公式だけ  : {k}" for k in only_ref]
    worst, wk = 0.0, None
    common = sorted(set(out) & set(ref))
    for k in common:
        a, b = out[k], ref[k]
        if tuple(a.shape) != tuple(b.shape):
            problems.append(f"形が違う: {k} こちら {tuple(a.shape)} / 公式 {tuple(b.shape)}")
            continue
        d = (a.float() - b.float()).abs()
        if not torch.isfinite(d).all():
            problems.append(f"NaN / inf: {k}")
            continue
        d = d.max().item()
        if d > worst:
            worst, wk = d, k
    if worst > 0.0:
        problems.append(f"最大差 {worst:.8f} ({wk})")
    if not quiet:
        print(f"\n  答え合わせ: こちら {len(out)} / 公式 {len(ref)} / 共通 {len(common)}")
        for p in problems[:16]:
            print("    ⚠ " + p)
        if not problems:
            print("    ✓ 公式の変換と完全一致")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--model", help="本体の safetensors。形を突き合わせる")
    ap.add_argument("--compare", help="公式の ComfyUI 版と突き合わせる（答え合わせ用）")
    a = ap.parse_args()

    out, bad = convert(a.src, a.model)
    rc = 0
    if bad:
        print(f"\n  問題が {len(bad)} 件あるので書き出さない")
        rc = 1

    if a.compare:
        if compare(out, a.compare):
            rc = 1

    if rc == 0 and a.dst:
        save_file(out, a.dst)
        print(f"\n  → {a.dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
