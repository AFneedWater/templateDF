import inspect
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from templatedf.data import AA_ORDER
from templatedf.decoder import SequenceDecoder, sinusoidal_position_encoding
from templatedf.model import ProteinAutoencoder


def small_model():
    return ProteinAutoencoder(dim=64, num_heads=4, encoder_blocks=2, decoder_blocks=2,
                              dropout=0.0)


def make_batch(lengths):
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    return torch.randn(len(lengths), max(lengths), 64), mask


def check_result(result, lengths, *, with_latent):
    expected = {'reconstructed_embeddings', 'logits', 'output_mask', 'output_lengths'}
    assert set(result) == expected | ({'latent'} if with_latent else set())
    assert result['logits'].shape == (len(lengths), max(lengths), 20)
    assert result['reconstructed_embeddings'].shape == (len(lengths), max(lengths), 64)
    assert result['output_lengths'].tolist() == lengths
    assert result['output_lengths'].dtype == torch.long
    assert result['output_mask'].dtype == torch.bool
    assert result['output_mask'].sum(1).tolist() == lengths
    for name in ('logits', 'reconstructed_embeddings'):
        assert torch.isfinite(result[name]).all()
        assert not result[name][~result['output_mask']].any()
    if with_latent:
        assert result['latent'].shape == (len(lengths), 50, 64)


@pytest.mark.parametrize('input_length', [10, 50, 150])
@pytest.mark.parametrize('output_length', [10, 50, 150])
def test_independent_input_and_output_length_boundaries(input_length, output_length):
    model = small_model().eval()
    x, mask = make_batch([input_length])
    with torch.no_grad():
        result = model(x, mask, torch.tensor([output_length]))
    check_result(result, [output_length], with_latent=True)


def test_mixed_lengths_default_reconstruction_and_explicit_encode_decode():
    model = small_model().eval()
    x, mask = make_batch([10, 50, 150])
    with torch.no_grad():
        default = model(x, mask)
        changed = model(x, mask, torch.tensor([10, 80, 150], dtype=torch.int32))
        z = model.encode(x, mask)
        decoded = model.decode(z, torch.tensor([10, 80, 150]))
    check_result(default, [10, 50, 150], with_latent=True)
    check_result(changed, [10, 80, 150], with_latent=True)
    check_result(decoded, [10, 80, 150], with_latent=False)
    torch.testing.assert_close(changed['logits'], decoded['logits'])
    assert model.aa_order == AA_ORDER == 'ACDEFGHIKLMNPQRSTVWY'
    assert model.decoder.amino_acid_head.out_features == 20


def test_single_batch_padding_and_other_sample_isolation():
    torch.manual_seed(303)
    model = small_model().eval()
    x, mask = make_batch([10, 50, 150])
    out_lengths = torch.tensor([10, 80, 150])
    with torch.no_grad():
        reference = model(x, mask, out_lengths)
        changed = x.clone()
        changed[~mask] = float('nan')
        dirty = model(changed, mask, out_lengths)
        extended = model(torch.cat([x, torch.full((3, 19, 64), float('inf'))], dim=1),
                         torch.cat([mask, torch.zeros(3,19,dtype=torch.bool)], dim=1), out_lengths)
        for name in ('latent', 'reconstructed_embeddings', 'logits'):
            for candidate in (dirty, extended):
                torch.testing.assert_close(candidate[name], reference[name], rtol=1e-5, atol=2e-6)
        for row, length in enumerate([10,50,150]):
            alone = model(x[row:row+1,:length], mask[row:row+1,:length], out_lengths[row:row+1])
            for name in ('logits','reconstructed_embeddings'):
                torch.testing.assert_close(alone[name][0], reference[name][row,:out_lengths[row]],
                                           rtol=1e-5, atol=2e-6)
        changed_other = x.clone()
        changed_other[1:] = torch.randn_like(changed_other[1:]) * 5
        changed_lengths = torch.tensor([10,20,30])
        other = model(changed_other, mask, changed_lengths)
        torch.testing.assert_close(other['logits'][0,:10], reference['logits'][0,:10],
                                   rtol=1e-5, atol=2e-6)
        perm = torch.tensor([2,0,1])
        reordered = model(x[perm], mask[perm], out_lengths[perm])
        torch.testing.assert_close(reordered['logits'], reference['logits'][perm], rtol=1e-5, atol=2e-6)


def test_dynamic_positions_and_fixed_length_scale():
    torch.manual_seed(304)
    decoder = SequenceDecoder(dim=64, num_heads=4, num_blocks=1, dropout=0).eval()
    observed = []
    handle = decoder.length_encoder.register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[0].detach().clone()))
    z = torch.randn(2,50,64)
    with torch.no_grad():
        a = decoder(z, torch.tensor([10,80]))
        b = decoder(z, torch.tensor([10,150]))
    handle.remove()
    torch.testing.assert_close(observed[0][:,0], torch.tensor([10.,80.]).log1p()/math.log1p(150))
    assert torch.equal(observed[0][0], observed[1][0])
    torch.testing.assert_close(a['logits'][0,:10], b['logits'][0,:10], rtol=1e-5, atol=2e-6)
    pos10 = sinusoidal_position_encoding(10,64,device=torch.device('cpu'),dtype=torch.float32)
    pos150 = sinusoidal_position_encoding(150,64,device=torch.device('cpu'),dtype=torch.float32)
    assert torch.equal(pos10, pos150[:10])
    assert not torch.equal(pos10[0],pos10[1])


def test_decoder_uses_latent_and_receives_no_input_memory_or_targets():
    torch.manual_seed(305)
    model = small_model().eval()
    z = torch.randn(3,50,64)
    lengths = torch.tensor([10,80,150])
    with torch.no_grad():
        a = model.decode(z, lengths)
        b = model.decode(torch.randn_like(z), lengths)
    assert (a['logits']-b['logits'])[a['output_mask']].abs().max() > 1e-4
    assert list(inspect.signature(SequenceDecoder.forward).parameters) == ['self','latent','output_lengths']
    seen = []
    def record_call(module, inputs, kwargs):
        seen.append((inputs,kwargs))
    handle = model.decoder.register_forward_pre_hook(record_call,with_kwargs=True)
    x, mask = make_batch([10,50,150])
    with torch.no_grad():
        result = model(x, mask, lengths)
    handle.remove()
    assert len(seen) == 1 and len(seen[0][0]) == 2 and seen[0][1] == {}
    assert seen[0][0][0] is result['latent']
    assert torch.equal(seen[0][0][1], lengths)
    # Hold the bottleneck fixed: changing raw embeddings cannot reach the decoder.
    class FixedEncoder(nn.Module):
        def forward(self, embeddings, input_mask):
            return z.to(embeddings)
    model.encoder = FixedEncoder()
    with torch.no_grad():
        first = model(x,mask,lengths)
        second = model(torch.randn_like(x),mask,lengths)
    assert torch.equal(first['logits'],second['logits'])


def test_attention_is_noncausal_and_masks_only_output_padding_keys():
    decoder = SequenceDecoder(dim=64,num_heads=4,num_blocks=2,dropout=0).eval()
    lengths = torch.tensor([10,80,150])
    calls=[]
    def capture(module, args, kwargs):
        calls.append((args,kwargs))
    handles=[]
    for block in decoder.blocks:
        handles.extend([block.self_attention.register_forward_pre_hook(capture,with_kwargs=True),
                        block.cross_attention.register_forward_pre_hook(capture,with_kwargs=True)])
    with torch.no_grad():
        result=decoder(torch.randn(3,50,64),lengths)
    for h in handles: h.remove()
    for index,(args,kwargs) in enumerate(calls):
        assert kwargs.get('attn_mask') is None and not kwargs.get('is_causal',False)
        if index%2 == 0:
            assert torch.equal(kwargs['key_padding_mask'],~result['output_mask'])
        else:
            assert args[1].shape == (3,50,64) and kwargs.get('key_padding_mask') is None


def test_full_backward_covers_encoder_decoder_heads_queries_and_valid_input():
    torch.manual_seed(306)
    model = small_model().train()
    x,mask=make_batch([10,50,150])
    x.requires_grad_()
    output=model(x,mask,torch.tensor([10,80,150]))
    valid=output['output_mask']
    labels=torch.randint(0,20,valid.shape)
    labels[~valid]=-100
    ce=F.cross_entropy(output['logits'].transpose(1,2),labels,ignore_index=-100)
    target=torch.randn_like(output['reconstructed_embeddings'])
    mse=(output['reconstructed_embeddings']-target).square().mean(-1)[valid].mean()
    loss=ce+0.2*mse
    loss.backward()
    assert torch.isfinite(loss)
    groups={'encoder':model.encoder,'length_encoder':model.decoder.length_encoder,
            'embedding_head':model.decoder.embedding_head,'amino_acid_head':model.decoder.amino_acid_head}
    groups.update({f'decoder_block_{i}':b for i,b in enumerate(model.decoder.blocks)})
    for name,module in groups.items():
        params=list(module.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),name
        assert sum(p.grad.abs().sum().item() for p in params)>0,name
    assert (model.encoder.latent_queries.grad.norm(dim=-1)>0).all()
    assert torch.isfinite(x.grad).all() and x.grad[mask].abs().sum()>0
    assert not x.grad[~mask].any()


@pytest.mark.parametrize('method',['encode','decode','forward'])
@pytest.mark.parametrize('condition',[{},0,torch.tensor([1.])])
def test_non_none_condition_always_raises(method,condition):
    model=small_model()
    with pytest.raises(NotImplementedError,match='condition'):
        getattr(model,method)(None,None,condition=condition)


@pytest.mark.parametrize('bad_lengths',[
    torch.tensor([0]),torch.tensor([-1]),torch.tensor([10.]),torch.tensor([True]),
    torch.tensor([[10]]),torch.tensor([10,20]),torch.tensor([]),[10],
])
def test_decoder_rejects_invalid_lengths(bad_lengths):
    decoder=SequenceDecoder(dim=64,num_heads=4,num_blocks=1)
    with pytest.raises(ValueError,match='output_lengths'):
        decoder(torch.randn(1,50,64),bad_lengths)


@pytest.mark.parametrize('length',[9,151])
def test_ae_enforces_peptide_bounds_at_input_and_decode(length):
    model=small_model()
    x,mask=make_batch([length])
    with pytest.raises(ValueError,match='Input peptide lengths'):
        model.encode(x,mask)
    with pytest.raises(ValueError,match='output_lengths'):
        model.decode(torch.randn(1,50,64),torch.tensor([length]))
    good_x,good_mask=make_batch([10])
    with pytest.raises(ValueError,match='output_lengths'):
        model(good_x,good_mask,torch.tensor([length]))


@pytest.mark.parametrize('kind',['all_masked','hole','mask_dtype','mask_shape','input_dim','nan','rank','non_tensor'])
def test_ae_rejects_invalid_input(kind):
    model=small_model()
    x,mask=make_batch([10])
    if kind=='all_masked': mask[:]=False
    elif kind=='hole': mask[0,1]=False
    elif kind=='mask_dtype': mask=mask.long()
    elif kind=='mask_shape': mask=mask[:,:-1]
    elif kind=='input_dim': x=x[:,:,:-1]
    elif kind=='nan': x[0,0,0]=float('nan')
    elif kind=='rank': x=x[0]
    elif kind=='non_tensor': x=None
    with pytest.raises(ValueError): model(x,mask)


@pytest.mark.parametrize('kind',['rank','width','tokens','batch','integer','nan','wrong_n'])
def test_ae_rejects_invalid_latent(kind):
    model=small_model()
    z=torch.randn(1,50,64)
    lengths=torch.tensor([10])
    if kind=='rank': z=z[0]
    elif kind=='width': z=z[:,:,:-1]
    elif kind=='tokens': z=z[:,:0]
    elif kind=='batch': z=z[:0]; lengths=lengths[:0]
    elif kind=='integer': z=z.long()
    elif kind=='nan': z[0,0,0]=float('nan')
    elif kind=='wrong_n': z=z[:,:49]
    with pytest.raises(ValueError): model.decode(z,lengths)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.bfloat16])
def test_decoder_explicit_dtype_and_device(dtype):
    decoder=SequenceDecoder(dim=64,num_heads=4,num_blocks=1,dropout=0).to(dtype=dtype).eval()
    z=torch.randn(2,50,64,dtype=dtype)
    with torch.no_grad(): result=decoder(z,torch.tensor([10,50]))
    check_result(result,[10,50],with_latent=False)
    assert result['logits'].dtype == dtype and result['logits'].device == z.device


@pytest.mark.parametrize('settings',[{'dim':63},{'num_heads':3},{'num_blocks':0},
                                    {'ffn_ratio':0},{'dropout':1},{'length_scale':0},
                                    {'length_scale':float('nan')}])
def test_invalid_decoder_configuration(settings):
    kwargs=dict(dim=64,num_heads=4)
    kwargs.update(settings)
    with pytest.raises(ValueError): SequenceDecoder(**kwargs)


def test_formal_autoencoder_defaults_without_allocating_weights():
    with torch.device('meta'):
        model=ProteinAutoencoder()
    assert model.dim==1280 and model.num_latents==50
    assert len(model.encoder.blocks)==len(model.decoder.blocks)==4
    assert model.decoder.length_scale==150
    assert model.decoder.blocks[0].self_attention.num_heads==20
    assert model.decoder.blocks[0].ffn[0].out_features==5120
    assert model.decoder.embedding_head.out_features==1280
    assert model.decoder.amino_acid_head.out_features==20
