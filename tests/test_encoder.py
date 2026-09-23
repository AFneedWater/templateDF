import pytest
import torch
from torch import nn

from templatedf.blocks import ConvMasked, RaygunBlock
from templatedf.encoder import CascadeEncoder


def small_encoder(**kwargs):
    return CascadeEncoder(dim=64, num_heads=4, num_blocks=3, dropout=0.0, **kwargs)


def make_batch(lengths, dim=64):
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    return torch.randn(len(lengths), max(lengths), dim), mask


@pytest.mark.parametrize("length", [10, 49, 50, 51, 150])
def test_lengths_produce_50_latents_without_extending_input(length):
    model = small_encoder().eval()
    x, mask = make_batch([length])
    observed = []
    handle = model.blocks[0].register_forward_pre_hook(lambda module, args: observed.append(args[0].shape))
    with torch.no_grad():
        latent = model(x, mask)
    handle.remove()
    assert latent.shape == (1, 50, 64)
    assert torch.isfinite(latent).all()
    assert observed == [torch.Size((1, length, 64))]
    # Independently initialized query vectors must not collapse to a shared token.
    assert not torch.equal(latent[:, 0], latent[:, 1])


def test_padding_values_appended_padding_and_batch_isolation():
    torch.manual_seed(20)
    model = small_encoder().eval()
    x, mask = make_batch([10, 49, 50, 51, 150])
    with torch.no_grad():
        reference = model(x, mask)
        noisy = x.clone()
        noisy[~mask] = float("nan")
        torch.testing.assert_close(model(noisy, mask), reference, rtol=1e-5, atol=1e-6)
        appended = torch.cat([x, torch.full((5, 17, 64), float("inf"))], dim=1)
        extra_mask = torch.cat([mask, torch.zeros(5, 17, dtype=torch.bool)], dim=1)
        torch.testing.assert_close(model(appended, extra_mask), reference, rtol=1e-5, atol=1e-6)
        for row, length in enumerate(mask.sum(1).tolist()):
            alone = model(x[row:row + 1, :length], mask[row:row + 1, :length])
            torch.testing.assert_close(alone[0], reference[row], rtol=1e-5, atol=1e-6)
        perm = torch.tensor([4, 2, 0, 3, 1])
        torch.testing.assert_close(model(x[perm], mask[perm]), reference[perm], rtol=1e-5, atol=1e-6)
        changed = x.clone()
        changed[1:] = torch.randn_like(changed[1:]) * 20
        torch.testing.assert_close(model(changed, mask)[0], reference[0], rtol=1e-5, atol=1e-6)


def test_fusion_contains_h0_and_each_updated_residual_and_zeros_padding():
    class Increment(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, x, mask):
            return torch.full_like(x, self.value)  # Intentionally nonzero in padding.

    model = small_encoder().eval()
    model.blocks = nn.ModuleList([Increment(1), Increment(2), Increment(4)])
    x, mask = make_batch([10, 13])
    captured = {}
    def capture_fusion(module, args):
        captured['levels'] = args[0].detach().clone()
    def capture_pool(module, args, kwargs):
        captured['memory'] = args[1].detach().clone()
    handles = [model.fusion.register_forward_pre_hook(capture_fusion),
               model.pool.register_forward_pre_hook(capture_pool, with_kwargs=True)]
    with torch.no_grad():
        model(x, mask)
    for handle in handles:
        handle.remove()
    levels = captured['levels'].split(64, dim=-1)
    assert len(levels) == 4
    for level, cumulative in zip(levels, [0, 1, 3, 7]):
        torch.testing.assert_close(level[mask], x[mask] + cumulative)
        assert not level[~mask].any()
    assert not captured['memory'][~mask].any()


def test_all_levels_fusion_pool_and_queries_receive_finite_gradients():
    torch.manual_seed(21)
    model = small_encoder()
    x, mask = make_batch([10, 49])
    x.requires_grad_()
    captured = {}
    def capture(module, args):
        args[0].retain_grad()
        captured['fusion_input'] = args[0]
    handle = model.fusion.register_forward_pre_hook(capture)
    z = model(x, mask)
    target = torch.randn_like(z)
    loss = (z - target).square().mean()
    loss.backward()
    handle.remove()
    assert torch.isfinite(loss)
    groups = {f'block_{i}': list(block.parameters()) for i, block in enumerate(model.blocks)}
    groups.update(fusion=list(model.fusion.parameters()), pool=list(model.pool.parameters()),
                  queries=[model.latent_queries])
    for name, parameters in groups.items():
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), name
        assert sum(p.grad.abs().sum().item() for p in parameters) > 0, name
    assert (model.latent_queries.grad.norm(dim=-1) > 0).all()
    for grad in captured['fusion_input'].grad.split(64, dim=-1):
        assert torch.isfinite(grad).all() and grad[mask].abs().sum() > 0
    assert x.grad[mask].abs().sum() > 0
    assert not x.grad[~mask].any()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_block_and_padding_preserve_explicit_dtype_after_warmup(dtype):
    model = RaygunBlock(dim=64, num_heads=4, dropout=0).eval()
    x, mask = make_batch([10, 13])
    with torch.no_grad():
        model(x, mask)  # Populate rotary tables before dtype conversion.
        model.to(dtype=dtype)
        y = model(x.to(dtype=dtype), mask)
    assert y.dtype == dtype and y.device == x.device
    assert torch.isfinite(y).all() and not y[~mask].any()
    conv = ConvMasked(64, 32, 7).to(dtype=dtype)
    out = conv(x.to(dtype=dtype).transpose(1, 2), mask)
    assert out.dtype == dtype and out.shape == (2, 32, 13)


@pytest.mark.parametrize('kind', ['all_masked', 'hole', 'wrong_dtype', 'wrong_shape', 'empty', 'wrong_dim'])
def test_invalid_inputs_raise_clear_errors(kind):
    model = small_encoder()
    x, mask = make_batch([10])
    if kind == 'all_masked': mask[:] = False
    elif kind == 'hole': mask[0, 3] = False
    elif kind == 'wrong_dtype': mask = mask.long()
    elif kind == 'wrong_shape': mask = mask[:, :-1]
    elif kind == 'empty': x, mask = x[:, :0], mask[:, :0]
    elif kind == 'wrong_dim': x = x[:, :, :-1]
    with pytest.raises(ValueError):
        model(x, mask)


@pytest.mark.parametrize('kwargs', [dict(dim=62), dict(num_heads=3), dict(dim=60, num_heads=4),
                                     dict(num_blocks=0), dict(num_latents=0), dict(conv_kernel=1),
                                     dict(pool_ffn_ratio=0), dict(fusion_hidden_dim=0), dict(dropout=1)])
def test_invalid_configuration(kwargs):
    settings = dict(dim=64, num_heads=4)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        CascadeEncoder(**settings)


def test_formal_defaults_and_independent_import():
    # Shape-only allocation checks the real constructor without 500 MB of weights.
    with torch.device('meta'):
        model = CascadeEncoder()
    assert model.dim == 1280 and model.num_latents == 50 and len(model.blocks) == 4
    assert model.latent_queries.shape == (50, 1280)
    assert model.fusion[0].in_features == 5 * 1280
    assert model.fusion[0].out_features == 2560
    assert model.pool.attention.num_heads == 20
    assert model.pool.ffn[0].out_features == 4 * 1280
    for block in model.blocks:
        assert block.encoder.fc1.out_features == 2 * 1280
        assert block.encoder.self_attn.rot_emb is not None
        assert block.encoder.self_attn.bias_k is None
