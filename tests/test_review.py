"""CPU-only checks for tools/h3_lora_convert.py and tools/profile_nodes.py.

No checkpoint, network, GPU or real ComfyUI server is needed: the converter is fed
tiny synthetic safetensors, the profiler is fed a fake websocket.

Origin: these cases were drafted 2026-09-06 as an external review harness
(ChatGPT "Astra", given the public repo). At that time nine of them were marked
xfail and reproduced real defects (incomplete qkv written as an empty file with
exit 0, A/B inner-rank mismatch accepted, fc1 `.diff` not permuted, NaN and
broadcast-compatible shapes passing `--compare`, `--compare` ignoring `--model`
errors, profiler not filtering `execution_error` by prompt_id, returning a summary
on a truncated or interrupted stream). The tools were fixed the same day; the
markers are gone and a few cases were added (alpha rejection, empty output,
`.diff_b` permutation, qkv `.diff` concatenation).

Run from the repo root:
    python -m pytest tests -q
Set H3_REVIEW_SOURCE_DIR to test a different copy of the tools.
"""
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(os.environ.get('H3_REVIEW_SOURCE_DIR', str(ROOT / 'tools')))


def import_file(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE_DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv = import_file('h3_lora_convert')
prof = import_file('profile_nodes')


def save(tmp_path, name, tensors):
    p = tmp_path / name
    save_file(tensors, str(p))
    return str(p)


def cli(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['h3_lora_convert.py', *args])
    return conv.main()


# ---------------------------------------------------------------- converter

def test_complete_qkv_preserves_effective_delta(tmp_path):
    src = {}
    deltas = []
    for i, branch in enumerate('qkv'):
        a = torch.arange(6, dtype=torch.float64).reshape(2, 3) + i
        b = torch.arange(4, dtype=torch.float64).reshape(2, 2) - i
        prefix = f'transformer_blocks.0.attn.to_{branch}'
        src[prefix + '.lora_A.weight'] = a
        src[prefix + '.lora_B.weight'] = b
        deltas.append(b @ a)
    out, bad = conv.convert(save(tmp_path, 'input.safetensors', src), quiet=True)
    prefix = 'diffusion_model.blocks.0.attn.qkv_proj'
    assert not bad
    assert torch.equal(out[prefix + '.lora_B.weight'] @ out[prefix + '.lora_A.weight'], torch.cat(deltas))


def test_qkv_dense_diff_is_concatenated_in_qkv_order(tmp_path):
    src = {}
    parts = []
    for i, branch in enumerate('qkv'):
        d = torch.full((2, 3), float(i))
        src[f'transformer_blocks.0.attn.to_{branch}.diff'] = d
        parts.append(d)
    out, bad = conv.convert(save(tmp_path, 'input.safetensors', src), quiet=True)
    assert not bad
    assert torch.equal(out['diffusion_model.blocks.0.attn.qkv_proj.diff'], torch.cat(parts))


def test_fc1_lora_permutation_is_correct(tmp_path):
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    src = {'transformer_blocks.0.ff.net.0.proj.lora_A.weight': a,
           'transformer_blocks.0.ff.net.0.proj.lora_B.weight': b}
    out, bad = conv.convert(save(tmp_path, 'input.safetensors', src), quiet=True)
    prefix = 'diffusion_model.blocks.0.mlp.fc1'
    delta = b @ a
    assert not bad
    assert torch.equal(out[prefix + '.lora_B.weight'] @ out[prefix + '.lora_A.weight'], torch.cat([delta[2:], delta[:2]]))


def test_fc1_dense_diff_gets_same_permutation(tmp_path):
    diff = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    p = save(tmp_path, 'input.safetensors', {'transformer_blocks.0.ff.net.0.proj.diff': diff})
    model = save(tmp_path, 'model.safetensors', {'blocks.0.mlp.fc1.weight': torch.zeros(4, 3)})
    out, bad = conv.convert(p, model, quiet=True)
    assert not bad
    assert torch.equal(out['diffusion_model.blocks.0.mlp.fc1.diff'], torch.cat([diff[2:], diff[:2]]))


def test_fc1_bias_diff_gets_same_permutation(tmp_path):
    diff_b = torch.arange(4, dtype=torch.float32)
    p = save(tmp_path, 'input.safetensors', {'transformer_blocks.0.ff.net.0.proj.diff_b': diff_b})
    out, bad = conv.convert(p, quiet=True)
    assert not bad
    assert torch.equal(out['diffusion_model.blocks.0.mlp.fc1.diff_b'], torch.tensor([2., 3., 0., 1.]))


def test_incomplete_qkv_fails_closed(tmp_path, monkeypatch):
    p = save(tmp_path, 'input.safetensors', {
        'transformer_blocks.0.attn.to_q.lora_A.weight': torch.ones(2, 3),
        'transformer_blocks.0.attn.to_q.lora_B.weight': torch.ones(4, 2),
    })
    dst = str(tmp_path / 'out.safetensors')
    rc = cli(monkeypatch, p, dst)
    assert rc != 0 and not Path(dst).exists()


def test_inner_rank_mismatch_is_rejected(tmp_path):
    p = save(tmp_path, 'input.safetensors', {
        'transformer_blocks.0.ff.net.2.lora_A.weight': torch.ones(2, 3),
        'transformer_blocks.0.ff.net.2.lora_B.weight': torch.ones(4, 3),
    })
    model = save(tmp_path, 'model.safetensors', {'blocks.0.mlp.fc2.weight': torch.zeros(4, 3)})
    out, bad = conv.convert(p, model, quiet=True)
    assert bad


def test_alpha_is_rejected(tmp_path, monkeypatch):
    # ComfyUI would apply alpha/rank as a scale; the converter has no way to know
    # what the source trainer meant by it, so refuse rather than silently rescale.
    p = save(tmp_path, 'input.safetensors', {
        'transformer_blocks.0.ff.net.2.lora_A.weight': torch.ones(2, 3),
        'transformer_blocks.0.ff.net.2.lora_B.weight': torch.ones(4, 2),
        'transformer_blocks.0.ff.net.2.alpha': torch.tensor(2.0),
    })
    dst = str(tmp_path / 'out.safetensors')
    assert cli(monkeypatch, p, dst) != 0 and not Path(dst).exists()


def test_empty_output_is_an_error(tmp_path, monkeypatch):
    # Everything maps to adaln → dropped → nothing left to save.
    p = save(tmp_path, 'input.safetensors', {
        'transformer_blocks.0.adaln_proj.linear.lora_A.weight': torch.ones(2, 3),
        'transformer_blocks.0.adaln_proj.linear.lora_B.weight': torch.ones(4, 2),
    })
    dst = str(tmp_path / 'out.safetensors')
    assert cli(monkeypatch, p, dst) != 0 and not Path(dst).exists()


def test_compare_rejects_nan(tmp_path, monkeypatch):
    p = save(tmp_path, 'input.safetensors', {'proj_in.diff': torch.full((2, 2), float('nan'))})
    ref = save(tmp_path, 'ref.safetensors', {'diffusion_model.video_patch_proj.diff': torch.zeros(2, 2)})
    assert cli(monkeypatch, p, '--compare', ref) != 0


def test_compare_checks_shapes_before_subtraction(tmp_path, monkeypatch):
    p = save(tmp_path, 'input.safetensors', {'proj_in.diff': torch.ones(2, 1)})
    ref = save(tmp_path, 'ref.safetensors', {'diffusion_model.video_patch_proj.diff': torch.ones(2, 3)})
    assert cli(monkeypatch, p, '--compare', ref) != 0


def test_compare_does_not_ignore_model_errors(tmp_path, monkeypatch):
    p = save(tmp_path, 'input.safetensors', {'proj_in.diff': torch.ones(2, 2)})
    ref = save(tmp_path, 'ref.safetensors', {'diffusion_model.video_patch_proj.diff': torch.ones(2, 2)})
    model = save(tmp_path, 'model.safetensors', {'video_patch_proj.weight': torch.zeros(3, 3)})
    assert cli(monkeypatch, p, '--compare', ref, '--model', model) != 0


def test_compare_exact_match_succeeds(tmp_path, monkeypatch):
    p = save(tmp_path, 'input.safetensors', {'proj_in.diff': torch.ones(2, 2)})
    ref = save(tmp_path, 'ref.safetensors', {'diffusion_model.video_patch_proj.diff': torch.ones(2, 2)})
    assert cli(monkeypatch, p, '--compare', ref) == 0


# ----------------------------------------------------------------- profiler

class FakeWS:
    def __init__(self, events):
        self.events = iter(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = next(self.events)
        except StopIteration:
            raise StopAsyncIteration
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event))


class FakeSession:
    def __init__(self, events):
        self.ws = FakeWS(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def ws_connect(self, *args, **kwargs):
        return self.ws

    async def post(self, *args, **kwargs):
        class Reply:
            async def json(self):
                return {'prompt_id': 'our-job'}
        return Reply()


def run_profile(monkeypatch, events):
    monkeypatch.setattr(prof.aiohttp, 'ClientSession', lambda: FakeSession(events))
    return asyncio.run(prof.run({'1': {'class_type': 'KSampler'}}, 'mock'))


def event(kind, prompt='our-job', **kwargs):
    return {'type': kind, 'data': {'prompt_id': prompt, **kwargs}}


def test_profiler_normal_completion(monkeypatch):
    result = run_profile(monkeypatch, [event('executing', node='1'), event('executing', node=None)])
    assert result is not None and 'KSampler' in result[0]


def test_profiler_ignores_other_jobs_error(monkeypatch):
    result = run_profile(monkeypatch, [
        event('execution_error', prompt='another-job', exception_message='other job failure'),
        event('executing', node='1'), event('executing', node=None)])
    assert result is not None


def test_profiler_rejects_incomplete_stream(monkeypatch):
    result = run_profile(monkeypatch, [event('executing', node='1')])
    assert result is None


def test_profiler_handles_interrupted(monkeypatch):
    result = run_profile(monkeypatch, [event('executing', node='1'), event('execution_interrupted', node_id='1')])
    assert result is None


def test_profiler_handles_own_error(monkeypatch):
    result = run_profile(monkeypatch, [
        event('executing', node='1'),
        event('execution_error', node_id='1', node_type='KSampler', exception_message='boom')])
    assert result is None
