#!/usr/bin/env python3
"""Move a full-rank H3 finetune's adaln (+ time embedder) onto a pruned checkpoint.

Pruned ComfyUI checkpoints fold the time embedding into `adaln_t_table` [1025, 8]
and keep adaln as `blocks.N.adaln_proj.linear.weight` [96768, 8]:

    z_p(t) = W_p @ table[t * 1024] + b_p

A full checkpoint (base H3, FastH3, ...) computes

    z_f(t) = W_f @ silu(temb_f(t)) + b_f          W_f [96768, 2688]

A diffusers-format LoRA cannot be mapped key-by-key onto the pruned basis, so the
converter drops adaln.  This tool instead takes the *finetuned full checkpoint*,
evaluates both sides on the 1025-row time grid and fits the difference in the
pruned basis by least squares:

    z_f(t_i) - z_p(t_i)  ~  dW @ table[i] + db

The residual of that fit is exactly what the pruned model cannot express; the tool
prints it per layer and refuses to write if it exceeds --max-rel-resid.  Output is
a ComfyUI patch (`diffusion_model.blocks.N.adaln_proj.linear.diff` / `.diff_b`,
plus `final_layer`), optionally merged into an existing converted LoRA.

Measured on FastH3 (dense 4-step) vs minimax_h3_fl2va_pruned: the finetuned time
curve lies in the pruned basis to 1.7e-5, the fit residual is ~5.5e-4 of the delta
on every layer, and the pruned base matches base H3 to 1.9e-4 (block 0, checked
against the HF weights).  Quality-wise the rebased adaln did not help FastH3 on our
reference-driven test (3 seeds, 124 f: ArcFace 0.669 -> 0.613, lip sync unchanged),
so the converter's "drop adaln" default stands; this tool is here for the
measurement and for finetunes where the adaln actually matters.

    python h3_adaln_rebase.py --pruned minimax_h3_..._pruned_int8_convrot.safetensors \\
        --full /path/to/FastH3 (a sharded HF dir with model.safetensors.index.json, or one file) \\
        --out fasth3_adaln_patch.safetensors [--merge-into fasth3_dense_4step_comfyui.safetensors]

Needs torch and safetensors.  CPU is fine (about a minute).
"""
import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class Shards:
    """Lazy reader over one safetensors file or an HF sharded directory."""

    def __init__(self, path):
        self.open = {}
        if os.path.isdir(path):
            idx = os.path.join(path, 'model.safetensors.index.json')
            if not os.path.exists(idx):
                idx = os.path.join(path, 'diffusion_pytorch_model.safetensors.index.json')
            self.map = {k: os.path.join(path, v) for k, v in json.load(open(idx))['weight_map'].items()}
        else:
            with safe_open(path, 'pt') as f:
                self.map = {k: path for k in f.keys()}

    def __contains__(self, key):
        return key in self.map

    def get(self, key):
        p = self.map[key]
        if p not in self.open:
            self.open[p] = safe_open(p, 'pt')
        return self.open[p].get_tensor(key).float()


def temb_silu(t, w1, b1, w2, b2, freq_dim=256):
    # mirrors comfy.ldm.minimax.model.TimeEmbedder followed by AdalnProj's silu
    half = freq_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
    h = torch.nn.functional.silu(emb @ w1.T + b1)
    return torch.nn.functional.silu(h @ w2.T + b2)


# full checkpoints come in two namings: MiniMax's own and diffusers'
TEMB_KEYS = [('time_embedder.proj_in', 'time_embedder.proj_out'),
             ('time_embedder.linear_1', 'time_embedder.linear_2')]


def layer_names(full):
    n = 0
    while f'blocks.{n}.adaln_proj.linear.weight' in full or f'transformer_blocks.{n}.adaln_proj.linear.weight' in full:
        n += 1
    if n == 0:
        sys.exit('no adaln layers found in the full checkpoint')
    diffusers = f'transformer_blocks.0.adaln_proj.linear.weight' in full
    layers = []
    for i in range(n):
        layers.append((f'blocks.{i}.adaln_proj.linear',
                       f'{"transformer_blocks" if diffusers else "blocks"}.{i}.adaln_proj.linear'))
    layers.append(('final_layer.adaln_proj.linear',
                   'norm_out.linear' if diffusers else 'final_layer.adaln_proj.linear'))
    return layers


def rebase(pruned_path, full_path, max_rel_resid, quiet=False):
    full = Shards(full_path)
    pr = safe_open(pruned_path, 'pt')
    if 'adaln_t_table' not in pr.keys():
        sys.exit('--pruned has no adaln_t_table; it is not a pruned checkpoint')
    table = pr.get_tensor('adaln_t_table').float()                    # [1025, 8]
    G = table.shape[0]
    t = torch.arange(G, dtype=torch.float32) / (G - 1)
    for a, b in TEMB_KEYS:
        if a + '.weight' in full:
            C = temb_silu(t, full.get(a + '.weight'), full.get(a + '.bias'),
                             full.get(b + '.weight'), full.get(b + '.bias'))  # [1025, 2688]
            break
    else:
        sys.exit('no time_embedder in the full checkpoint')
    M = torch.cat([table, torch.ones(G, 1)], 1).double()               # [1025, 9]
    Pinv = torch.linalg.pinv(M)
    Gm = (C.double().T @ Pinv.T)                                        # [2688, 9]
    curve_res = (C.double().T - Gm @ M.T)
    stats = {'curve_rel_resid': (curve_res.norm() / C.norm()).item(), 'layers': {}}
    if not quiet:
        print(f'time curve in the pruned basis: rel resid {stats["curve_rel_resid"]:.2e}')
    Gm = Gm.float()

    out, worst = {}, 0.0
    for pk, fk in layer_names(full):
        Wf, bf = full.get(fk + '.weight'), full.get(fk + '.bias')
        Wp, bp = pr.get_tensor(pk + '.weight').float(), pr.get_tensor(pk + '.bias').float()
        if Wf.shape[0] != Wp.shape[0]:
            sys.exit(f'{pk}: rows {Wf.shape[0]} (full) vs {Wp.shape[0]} (pruned)')
        fit = Wf @ Gm                                                    # least squares in one matmul
        dW, db = fit[:, :8] - Wp, fit[:, 8] + bf - bp
        delta = (Wf @ C.T + bf[:, None]) - (Wp @ table.T + bp[:, None])
        res = delta - (dW @ table.T + db[:, None])
        rel = (res.norm() / delta.norm()).item()
        worst = max(worst, rel)
        stats['layers'][pk] = {'delta_rms': delta.pow(2).mean().sqrt().item(), 'resid_max_abs': res.abs().max().item(), 'rel_resid': rel}
        if not quiet:
            print(f'{pk:34s} delta rms {stats["layers"][pk]["delta_rms"]:.4f}  fit resid rel {rel:.2e}  max {stats["layers"][pk]["resid_max_abs"]:.1e}')
        if not (torch.isfinite(dW).all() and torch.isfinite(db).all()):
            sys.exit(f'{pk}: non-finite result')
        out[f'diffusion_model.{pk}.diff'] = dW.contiguous()
        out[f'diffusion_model.{pk}.diff_b'] = db.contiguous()
    stats['worst_rel_resid'] = worst
    if worst > max_rel_resid:
        sys.exit(f'fit residual {worst:.2e} exceeds --max-rel-resid {max_rel_resid}; the pruned basis cannot carry this finetune')
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pruned', required=True, help='pruned ComfyUI checkpoint (has adaln_t_table)')
    ap.add_argument('--full', required=True, help='finetuned full checkpoint: one .safetensors or a sharded HF directory')
    ap.add_argument('--out', required=True, help='output patch (.safetensors)')
    ap.add_argument('--merge-into', help='existing converted LoRA to combine with (its keys must not overlap)')
    ap.add_argument('--max-rel-resid', type=float, default=1e-2, help='refuse if any layer fits worse than this (default 1e-2)')
    ap.add_argument('--stats', help='write per-layer residuals to this JSON')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    out, stats = rebase(a.pruned, a.full, a.max_rel_resid, a.quiet)
    meta = {'source': f'{os.path.basename(a.full)} adaln + time_embedder rebased onto {os.path.basename(a.pruned)} adaln_t_table (least squares)',
            'worst_rel_resid': f'{stats["worst_rel_resid"]:.3e}'}
    if a.merge_into:
        base = load_file(a.merge_into)
        dup = set(base) & set(out)
        if dup:
            sys.exit(f'{len(dup)} keys already exist in {a.merge_into}, e.g. {sorted(dup)[0]}')
        out = {**base, **out}
        meta['merged_into'] = os.path.basename(a.merge_into)
    save_file(out, a.out, metadata=meta)
    if a.stats:
        json.dump(stats, open(a.stats, 'w'), indent=1)
    if not a.quiet:
        print(f'wrote {a.out}: {len(out)} tensors, worst fit residual {stats["worst_rel_resid"]:.2e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
