# MiniMax H3 speed-up notes

Field notes from three weeks (2026-08-14 → 09-05) of running [MiniMax H3](https://huggingface.co/MiniMaxAI)
— the open-weight video+audio diffusion model — on a single RTX 5090 (32 GB, WSL2, ComfyUI 0.33)
for a home lip-sync music-video studio, plus a 4090 worker and an Apple Silicon machine.

Everything here is measured on that setup. The full write-up (Japanese) is
**[H3_SPEEDUPS.md](H3_SPEEDUPS.md)**; this README is the short English version.

## Where the time went

1280×704, ~5 s clip, one 5090:

| stage | seconds |
|---|---|
| plain 20 steps (extrapolated) | ~270 |
| + official Turbo LoRA, 8 steps | 125 |
| + comfy-kitchen INT8 attention (`ModelAttentionBackend`, built into ComfyUI) | **79** |

A 7.6 s reference-driven lip-sync segment went 127 s → **69 s**.

## What's in here

| # | technique | measured effect | origin |
|---|---|---|---|
| 1 | int8 ConvRot weights (Comfy-Org) | 21 GB fits the 5090; true quantization error **0.90 %** (the 0.17 % quoted in the converter is re-encoding of already-int8 weights, not quantization error) | official / **our measurement** |
| 2 | Turbo LoRA, 8 steps | 2.2× vs 20 steps; Japanese text baked into the frame stays legible at 8 steps | official |
| 3 | FastH3 4-step LoRA (FastVideo DMD2, dense) converted to ComfyUI | 1.87× vs Turbo 8-step (118 s vs 225 s); identity and lip-sync intact; zero failures over a 30-segment MV | LoRA: FastVideo / **converter: ours** |
| 4 | comfy-kitchen INT8 attention | sampler ≈ **2×** on the 5090 (99→54 s, 91→45 s); 1.34× on a 4090 with Q4_K_M GGUF; ArcFace identity and lip correlation unchanged | feature: ComfyUI / **H3 measurements + production rollout: ours** |
| 5 | width 1280 → 1216 | VAE decode tiles 28→24, total **−15 %**. Bigger tiles are a dead end: 512/768/untiled decode is no faster (13.0→12.2 s) and the picture breaks (PSNR 40→21 dB, a 16×16-token grid) — the decoder is a 256-px ViT. Its time is 75 % fp16 GEMM, 9 % attention, so INT8 attention won't help the VAE either. Beware: ComfyUI's `VAE.decode` silently falls back to tiled decode on OOM. (MLX side: TAEH3 decodes 133× faster at PSNR-Y 25.8 dB — preview only) | ours (TAEH3: m4max) |
| 6 | pull-based distributed segment queue (5090 + 4090) | 28-segment MV 3.8 h → 2.7 h; grouping same-kind jobs per worker keeps model swap cost at 55 s instead of 210 s | ours (design) |
| 7 | the 345-frame cliff on the 4090 | 6 of 7 runs 3–17× slower; 328 f is stable (11 runs within 1.7 %). Not VRAM — memory lines identical; time is lost *outside* sampling. Cap set to 328 | ours (and a lesson: never set a limit from n=1) |
| 8 | Zironic H3-Optimizations `H3MemoryOptimization` | 1.27× (134→106 s), but **output no longer matches the plain model at the same seed** (mean pixel diff 2.27/255, undocumented; run-to-run reproducibility *with* the node was not measured) and it **fails on GGUF** (`3024x14336` = Q4_K_M container shape read as the weight shape) | node: Zironic / **findings: ours** |
| 9 | ToneCompensate (seam colour correction) | measured bias < 1/255 and not even consistent in sign → not needed here | negative result |
| 10 | Sage attention / FastVideo VSA / 148 GB full model | no nvcc here; VSA needs its own kernel + CUDA 13 + B200 | not usable |
| 11 | alibaba-pai Acc-LoRAs | not LoRAs: PDD weights (`pdd_num_steps: 32`, `proj_out (32, 96, 5376)`) — needs a PDD sampler ComfyUI doesn't have | not usable |
| 12–13 | ops: don't poll ComfyUI's HTTP during sampling (it stalls 18–43 s); ComfyUI *degrades* (1.8× slower clips) before its CUDA context dies — restart on that signal | ours |

### Two findings worth pulling out

**Converting diffusers-format H3 LoRAs to ComfyUI: the fused `fc1` halves are swapped.**
diffusers stores `[value; gate]`, ComfyUI's H3 stores `[gate; value]`. Shapes match, so nothing errors —
the LoRA just half-works: style transfers, identity doesn't, higher strength collapses, steps don't help.
The fused `qkv` needs a block-diagonal `lora_B`. Both mappings were verified bit-exact against a LoRA that
lightx2v ships in both formats (416/416 tensors). The `adaln` family is dropped by the converter: the pruned
checkpoints fold the time embedding into a `[1025, 8]` table, so a diffusers adaln LoRA has no direct
mapping. Tool: [`tools/h3_lora_convert.py`](tools/h3_lora_convert.py) (`--compare` reproduces the check).
We later *did* rebase FastH3's full adaln weights onto that table (see the next finding) — and it did not help,
so dropping stays the default.

**FastH3's transformer blocks look identical to H3's.** While checking the int8 ConvRot conversion,
"H3 vs FastH3" differences matched the int8 quantization error to three decimals on all ten blocks tested
(of 50; the rest were not checked). As far as tested, the 4-step distillation lives only in the `adaln` weights (`[96768, 2688]` full vs `[96768, 8]` pruned).
The pruned table satisfies `z(i) = B · silu(temb(t))` with residual 2.4e-6 (`i/1024 = t`), so an adaln
LoRA trained on the pruned model can be lifted to full rank with `ΔW_full = ΔW_pruned @ Z @ pinv(C)`.

**Rebasing FastH3's adaln onto the pruned table works mathematically and buys nothing.**
[`tools/h3_adaln_rebase.py`](tools/h3_adaln_rebase.py) evaluates both models' modulation curves on the
1025-row grid and least-squares fits `z_full(t) − z_pruned(t)` in the 9-dim `[table | 1]` basis; the
residual is what the pruned basis cannot carry, and the tool exits 1 above a threshold. For FastH3 all
51 layers fit with residual 5–6e-4 of the delta (the delta itself is ~2.5 % of `|z|`) — any smooth
curve over `t ∈ [0, 1]` does, since every sinusoid frequency in the time embedding is ≤ 1 rad. (We first
checked that the pruned adaln equals base H3's: 1.9e-4.) Merged into the 4-step LoRA and compared
against the adaln-less version at 1280×704, 124 f, same reference face and audio, 3 seeds: ArcFace
identity **0.669 → 0.613** (lower on all three seeds: −0.09 / −0.04 / −0.03), lip correlation unchanged
(+0.65 vs +0.64), no visible artifacts either way. n=3, so "slightly worse" is as far as we go — but there is
no reason to carry it. Note the FastVideo "datafree" LoRA's adaln `B·A` is *not* `W_fast − W_h3` in weight
space (0.67 vs 0.21) even though base + LoRA matches FastH3 along the time curve to 4e-4; the LoRA is
only meaningful on the curve, so rebase from full weights, not from the LoRA.

## Tools

- `tools/h3_lora_convert.py` — diffusers → ComfyUI LoRA converter for H3 (needs `torch`, `safetensors`).
  `python h3_lora_convert.py in.safetensors out.safetensors --model minimax_h3_....safetensors`
  checks every tensor against the checkpoint header (reads only the header, not the 21 GB).
  It fails closed: incomplete `qkv` triples, A/B rank mismatches, non-finite values, unknown
  suffixes, shapes that don't match the checkpoint, and `.alpha` keys (it has no way to know
  what scale the trainer meant — FastH3 and lightx2v ship none) all give exit 1 and no output file.
  `--compare ref.safetensors` demands an exact tensor-by-tensor match, including shapes.
- `tools/profile_nodes.py` — per-node timing from ComfyUI's websocket `executing` events
  (`/history` only gives you the total). Needs `aiohttp`.
  `python profile_nodes.py workflow_api.json --label "t2v 1280x704"`
  Events from other jobs are ignored; an error, interruption or truncated stream gives exit 1
  instead of a summary.
- `tools/h3_adaln_rebase.py` — rebases a full-rank H3 adaln (any `.safetensors` or HF sharded dir)
  onto a pruned checkpoint's `[1025, 8]` table as `.diff`/`.diff_b` tensors, optionally `--merge-into`
  an existing ComfyUI LoRA. Fails closed when the fit residual exceeds `--max-rel-resid`.
- `tests/test_review.py` — CPU-only tests for the three tools (synthetic safetensors, fake websocket).
  `python -m pytest tests -q`. They started life as an external review harness that reproduced
  nine real defects in the first published version (2026-09-06); the history of that is in the
  file's docstring.

## Measurement rules we had to learn the hard way

1. Never decide from one seed or one run (bit us twice: a merge's composition, the 345-frame cap).
2. Always take a control (a prompt alone scores 0.13 on ArcFace; only the *difference* means "the face came through").
3. Discard the cold first run; change the seed between runs (ComfyUI caches identical graphs → 0.1 s).
4. Same shape ≠ same order. Prove the order of fused weights by splitting and merging before you trust a conversion.
5. Profile per node — totals hide the VAE's fixed cost.

## License

MIT. Numbers and text are free to reuse with attribution.
