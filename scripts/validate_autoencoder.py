"""Bounded stage-03 data-flow/gradient evidence using synthetic embeddings only."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from templatedf.model import ProteinAutoencoder


def gradient_summary(parameters):
    parameters=list(parameters)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norms=[p.grad.float().norm().item() for p in parameters]
    norm=math.sqrt(sum(value*value for value in norms))
    assert norm>0
    return {'parameter_tensors':len(parameters),'nonzero_gradient_tensors':sum(value>0 for value in norms),
            'all_finite':True,'gradient_l2':norm}


def parameter_counts(model):
    encoder=sum(p.numel() for p in model.encoder.parameters())
    decoder=sum(p.numel() for p in model.decoder.parameters())
    return {'encoder':encoder,'decoder':decoder,'total':encoder+decoder}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260923)
    model=ProteinAutoencoder(dim=64,num_heads=4,encoder_blocks=2,decoder_blocks=2,dropout=0.1).eval()
    input_lengths=torch.tensor([10,50,150])
    output_lengths=torch.tensor([10,80,150])
    mask=torch.arange(150)[None,:]<input_lengths[:,None]
    x=torch.randn(3,150,64)
    with torch.no_grad():
        result=model(x,mask,output_lengths)
        same_length=model(x,mask)
        dirty=x.clone()
        dirty[~mask]=float('nan')
        dirty_result=model(dirty,mask,output_lengths)
        padded_result=model(torch.cat([x,torch.full((3,19,64),float('inf'))],1),
                            torch.cat([mask,torch.zeros(3,19,dtype=torch.bool)],1),output_lengths)
        def errors(candidate):
            values={key:(candidate[key]-result[key]).abs().max().item()
                    for key in ('latent','logits','reconstructed_embeddings')}
            for key in values:
                torch.testing.assert_close(candidate[key],result[key],rtol=1e-5,atol=2e-6)
            return values
        isolation={'changed_padding':errors(dirty_result),'extra_padding':errors(padded_result),
                   'single_vs_batch':[]}
        for row,length in enumerate(input_lengths.tolist()):
            single=model(x[row:row+1,:length],mask[row:row+1,:length],output_lengths[row:row+1])
            out_length=int(output_lengths[row])
            single_errors={}
            for key in ('logits','reconstructed_embeddings'):
                reference=result[key][row,:out_length]
                torch.testing.assert_close(single[key][0],reference,rtol=1e-5,atol=2e-6)
                single_errors[key]=(single[key][0]-reference).abs().max().item()
            isolation['single_vs_batch'].append({'input_length':length,'output_length':out_length,**single_errors})
        altered_z=torch.randn_like(result['latent'])
        changed=model.decode(altered_z,output_lengths)
        valid=result['output_mask']
        latent_difference=(changed['logits']-result['logits'])[valid]
        assert latent_difference.abs().max()>1e-4
        dependency={'changed_z_logits_max_abs_difference':latent_difference.abs().max().item(),
                    'changed_z_logits_mean_abs_difference':latent_difference.abs().mean().item()}
    original_encoder=model.encoder
    fixed_latent=result['latent'].detach()
    class FixedEncoder(nn.Module):
        def forward(self,embeddings,input_mask):
            return fixed_latent
    model.encoder=FixedEncoder()
    with torch.no_grad():
        fixed_a=model(x,mask,output_lengths)
        fixed_b=model(torch.randn_like(x),mask,output_lengths)
        dependency['fixed_z_changed_embeddings_logits_max_abs_difference']=(fixed_a['logits']-fixed_b['logits']).abs().max().item()
        assert torch.equal(fixed_a['logits'],fixed_b['logits'])
    model.encoder=original_encoder
    captured={}
    def decoder_args(module,inputs,kwargs):
        captured.update(positional_argument_count=len(inputs),keyword_arguments=list(kwargs),
                        latent_shape=list(inputs[0].shape),lengths_shape=list(inputs[1].shape))
    handle=model.decoder.register_forward_pre_hook(decoder_args,with_kwargs=True)
    model.train()
    grad_input=x.clone().requires_grad_()
    output=model(grad_input,mask,output_lengths)
    handle.remove()
    output['latent'].retain_grad()
    valid=output['output_mask']
    labels=torch.randint(0,20,valid.shape)
    labels[~valid]=-100
    ce=F.cross_entropy(output['logits'].transpose(1,2),labels,ignore_index=-100)
    target=torch.randn_like(output['reconstructed_embeddings'])
    embedding_mse=(output['reconstructed_embeddings']-target).square().mean(-1)[valid].mean()
    loss=ce+0.2*embedding_mse
    loss.backward()
    grads={f'encoder.blocks.{i}':gradient_summary(block.parameters()) for i,block in enumerate(model.encoder.blocks)}
    grads.update({
        'encoder.fusion':gradient_summary(model.encoder.fusion.parameters()),
        'encoder.query_pool':gradient_summary(model.encoder.pool.parameters()),
        'encoder.latent_queries':gradient_summary([model.encoder.latent_queries]),
        'decoder.length_encoder':gradient_summary(model.decoder.length_encoder.parameters()),
        'decoder.embedding_head':gradient_summary(model.decoder.embedding_head.parameters()),
        'decoder.amino_acid_head':gradient_summary(model.decoder.amino_acid_head.parameters()),
    })
    grads.update({f'decoder.blocks.{i}':gradient_summary(block.parameters()) for i,block in enumerate(model.decoder.blocks)})
    assert torch.isfinite(grad_input.grad).all() and grad_input.grad[mask].abs().sum()>0
    assert not grad_input.grad[~mask].any()
    assert output['latent'].grad.norm()>0
    with torch.device('meta'):
        formal_default=ProteinAutoencoder()
    device=torch.device(args.device)
    shallow=ProteinAutoencoder(encoder_blocks=1,decoder_blocks=1).to(device).eval()
    big_x=torch.randn(3,150,1280,device=device)
    big_mask=mask.to(device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        # CPU lengths deliberately exercise the explicit move to the latent device.
        formal=shallow(big_x,big_mask,output_lengths)
        assert formal['logits'].shape==(3,150,20)
        assert formal['output_mask'].sum(1).tolist()==[10,80,150]
        assert torch.isfinite(formal['logits']).all() and torch.isfinite(formal['reconstructed_embeddings']).all()
        gpu={'device':str(device),'encoder_blocks':1,'decoder_blocks':1,'parameters':parameter_counts(shallow),
             'latent_shape':list(formal['latent'].shape),'embedding_shape':list(formal['reconstructed_embeddings'].shape),
             'logits_shape':list(formal['logits'].shape),'output_mask_counts':formal['output_mask'].sum(1).tolist(),
             'output_lengths_device':str(formal['output_lengths'].device),'fp32_finite':True}
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            gpu['name']=torch.cuda.get_device_name(device)
            gpu['fp32_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
            if torch.cuda.is_bf16_supported(including_emulation=False):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    bf16=shallow(big_x,big_mask,output_lengths)
                torch.cuda.synchronize(device)
                assert torch.isfinite(bf16['logits']).all() and torch.isfinite(bf16['reconstructed_embeddings']).all()
                gpu['bf16_autocast']={'logits_dtype':str(bf16['logits'].dtype),
                                     'embedding_dtype':str(bf16['reconstructed_embeddings'].dtype),'finite':True,
                                     'valid_logits_max_abs_difference_from_fp32':(bf16['logits'].float()-formal['logits'])[formal['output_mask']].abs().max().item()}
    report={
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'torch':torch.__version__,
        'inputs':'synthetic; no ESM loading, real-sequence training or optimization',
        'small_model':{'dim':64,'num_latents':50,'encoder_blocks':2,'decoder_blocks':2,
                       'parameters':parameter_counts(model)},
        'shapes':{'input':list(x.shape),'input_lengths':input_lengths.tolist(),
                  'default_output_lengths':same_length['output_lengths'].tolist(),
                  'explicit_output_lengths':result['output_lengths'].tolist(),
                  'latent':list(result['latent'].shape),'logits':list(result['logits'].shape),
                  'reconstructed_embeddings':list(result['reconstructed_embeddings'].shape),
                  'output_mask_counts':result['output_mask'].sum(1).tolist()},
        'isolation_errors':isolation,'latent_dependency':dependency,'decoder_actual_arguments':captured,
        'gradient_check':{'loss':loss.item(),'ce':ce.item(),'embedding_mse':embedding_mse.item(),'mse_weight':0.2,
                          'groups':grads,'latent_gradient_l2':output['latent'].grad.norm().item(),
                          'valid_input_gradient_l2':grad_input.grad[mask].norm().item(),'padding_input_gradient_zero':True},
        'formal_default_parameter_counts_meta_only':parameter_counts(formal_default),
        'formal_shallow_forward':gpu,
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
