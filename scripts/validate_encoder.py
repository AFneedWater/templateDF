"""Bounded stage-02 encoder evidence; synthetic embeddings, no ESM model loads."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch
from esm.modules import TransformerLayer

from templatedf.encoder import CascadeEncoder


def parameter_counts(model):
    counts = {f"blocks.{i}": sum(p.numel() for p in block.parameters())
              for i, block in enumerate(model.blocks)}
    counts.update(fusion=sum(p.numel() for p in model.fusion.parameters()),
                  query_pool=sum(p.numel() for p in model.pool.parameters()),
                  latent_queries=model.latent_queries.numel())
    counts['total'] = sum(p.numel() for p in model.parameters())
    assert counts['total'] == sum(value for key, value in counts.items() if key != 'total')
    return counts


def gradient_stats(parameters):
    parameters = list(parameters)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norms = [p.grad.float().norm().item() for p in parameters]
    result = {'parameter_tensors': len(parameters), 'finite': True,
              'nonzero_gradient_tensors': sum(n > 0 for n in norms),
              'gradient_l2': math.sqrt(sum(n * n for n in norms)),
              'min_tensor_gradient_l2': min(norms)}
    assert result['gradient_l2'] > 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    model = CascadeEncoder(dim=64, num_heads=4, num_blocks=4, dropout=0.1).eval()
    lengths = torch.tensor([10, 49, 50, 51, 150])
    mask = torch.arange(150)[None, :] < lengths[:, None]
    x = torch.randn(5, 150, 64)
    captured = {}
    def record_fusion(module, inputs):
        captured['fusion_input_shape'] = list(inputs[0].shape)
        assert not inputs[0][~mask].any()
    def record_memory(module, inputs):
        captured['memory_shape'] = list(inputs[1].shape)
        assert not inputs[1][~mask].any()
    handles = [model.fusion.register_forward_pre_hook(record_fusion),
               model.pool.register_forward_pre_hook(record_memory)]
    with torch.no_grad():
        z = model(x, mask)
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        corrupt = x.clone()
        corrupt[~mask] = float('nan')
        changed_padding = model(corrupt, mask)
        extended = model(torch.cat([x, torch.full((5, 17, 64), float('inf'))], 1),
                         torch.cat([mask, torch.zeros(5, 17, dtype=torch.bool)], 1))
        errors = {'changed_invalid_values': (changed_padding-z).abs().max().item(),
                  'appended_17_positions': (extended-z).abs().max().item(), 'single_vs_batch': {}}
        for variant in (changed_padding, extended):
            torch.testing.assert_close(variant, z, rtol=1e-5, atol=1e-6)
        for index, length in enumerate(lengths.tolist()):
            single = model(x[index:index+1, :length], mask[index:index+1, :length])
            torch.testing.assert_close(single[0], z[index], rtol=1e-5, atol=1e-6)
            errors['single_vs_batch'][str(length)] = (single[0]-z[index]).abs().max().item()
    model.train()
    grad_input = torch.randn(2, 49, 64, requires_grad=True)
    grad_mask = torch.arange(49)[None, :] < torch.tensor([10, 49])[:, None]
    levels = {}
    def keep_fusion_gradient(module, inputs):
        inputs[0].retain_grad()
        levels['tensor'] = inputs[0]
    handle = model.fusion.register_forward_pre_hook(keep_fusion_gradient)
    output = model(grad_input, grad_mask)
    loss = (output - torch.randn_like(output)).square().mean()
    loss.backward()
    handle.remove()
    grads = {f'blocks.{i}': gradient_stats(block.parameters()) for i, block in enumerate(model.blocks)}
    grads.update(fusion=gradient_stats(model.fusion.parameters()),
                 query_pool=gradient_stats(model.pool.parameters()),
                 latent_queries=gradient_stats([model.latent_queries]))
    level_grads = [chunk[grad_mask].norm().item() for chunk in levels['tensor'].grad.split(64, -1)]
    assert all(value > 0 for value in level_grads)
    query_row_norms = model.latent_queries.grad.norm(dim=-1)
    assert (query_row_norms > 0).all() and not grad_input.grad[~grad_mask].any()
    # Demonstrate why the inherited add_bias_kv default was changed.
    bias_evidence = {}
    for bias in (True, False):
        torch.manual_seed(42)
        layer = TransformerLayer(64, 128, 4, add_bias_kv=bias, use_rotary_embeddings=True).eval()
        tokens = torch.randn(10, 1, 64)
        padded = torch.cat([tokens, torch.zeros(13, 1, 64)])
        with torch.no_grad():
            single, _ = layer(tokens, self_attn_padding_mask=torch.zeros(1,10,dtype=torch.bool))
            batch, _ = layer(padded, self_attn_padding_mask=torch.arange(23)[None,:] >= 10)
        bias_evidence[str(bias)] = (single-batch[:10]).abs().max().item()
    with torch.device('meta'):
        default_model = CascadeEncoder()
    formal_counts = parameter_counts(default_model)
    device = torch.device(args.device)
    shallow = CascadeEncoder(dim=1280, num_heads=20, num_blocks=1).to(device).eval()
    big_x = torch.randn(2, 150, 1280, device=device)
    big_mask = torch.arange(150, device=device)[None,:] < torch.tensor([10,150], device=device)[:,None]
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        formal_z = shallow(big_x, big_mask)
        assert formal_z.shape == (2,50,1280) and torch.isfinite(formal_z).all()
        formal = {'device': str(device), 'num_blocks': 1, 'input_shape': list(big_x.shape),
                  'lengths': [10,150], 'output_shape': list(formal_z.shape),
                  'output_dtype': str(formal_z.dtype), 'finite': True,
                  'parameters': parameter_counts(shallow)}
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            formal['gpu_name'] = torch.cuda.get_device_name(device)
            formal['fp32_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            if torch.cuda.is_bf16_supported(including_emulation=False):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    bf16_z = shallow(big_x, big_mask)
                torch.cuda.synchronize(device)
                assert torch.isfinite(bf16_z).all()
                formal['bf16_autocast'] = {'dtype': str(bf16_z.dtype), 'finite': True,
                                          'shape': list(bf16_z.shape),
                                          'max_abs_difference_from_fp32': (bf16_z.float()-formal_z).abs().max().item()}
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
        'torch_version': torch.__version__, 'input_kind': 'synthetic embeddings; no ESM weights',
        'small_model': {'dim': 64, 'num_heads':4, 'num_blocks':4, 'num_latents':50,
                        'parameters': parameter_counts(model), 'input_shape': list(x.shape),
                        **captured, 'latent_shape': list(z.shape)},
        'max_abs_errors': errors, 'loss': loss.item(), 'gradient_groups': grads,
        'fusion_level_gradient_l2_H0_to_H4': level_grads,
        'query_rows_with_nonzero_gradient': int((query_row_norms > 0).sum()),
        'min_query_row_gradient_l2': query_row_norms.min().item(),
        'padded_input_gradient_is_zero': True,
        'upstream_bias_kv_padding_error': bias_evidence,
        'formal_default_parameter_counts_meta_only': formal_counts,
        'formal_shallow_forward': formal,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
