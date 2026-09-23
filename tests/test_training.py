from copy import deepcopy
import json
import math
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from templatedf.checkpoint import capture_rng, load_checkpoint
from templatedf.losses import MetricAccumulator, reconstruction_loss
from templatedf.training import TrainingEngine, TrainingSettings


def fixture_data(dim=32):
    generator=torch.Generator().manual_seed(9)
    def sample(index,length):
        return {'embeddings':torch.randn(length,dim,generator=generator),
                'labels':torch.full((length,),index%3,dtype=torch.long)}
    return [sample(i,10+i) for i in range(5)],[sample(i,11+i) for i in range(3)]


def fixture_config(steps=8,**changes):
    settings=dict(stage='1A',precision='fp32',device='cpu',batch_size=2,max_steps=steps,
                  learning_rate=0.002,weight_decay=0.01,grad_clip=0.1,seed=17,
                  eval_every=2,save_every=2,scheduler_gamma=0.95)
    settings.update(changes)
    return {'model':dict(dim=32,num_heads=4,encoder_blocks=1,decoder_blocks=1,dropout=0.1),
            'training':settings}


def provenance():
    return {'data_kind':'synthetic','esm_source':{'source':'mock:training-tests'},
            'train':{'fixture':'fixed-train-five'},'validation':{'fixture':'fixed-val-three'}}


def assert_nested_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: assert_nested_equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_nested_equal(x,y)
    else:
        assert a==b


def test_loss_manual_denominators_padding_and_gradients():
    mask=torch.tensor([[True,True,False,False],[True,True,True,True]])
    labels=torch.tensor([[0,1,-100,-100],[0,0,0,1]])
    logits=torch.zeros(2,4,20,requires_grad=True)
    reconstructed=torch.tensor([[[1.,1.]]*4,[[3.,3.]]*4],requires_grad=True)
    target=torch.zeros_like(reconstructed)
    result=reconstruction_loss(logits,reconstructed,target,labels,mask,lambda_emb=0.25)
    assert result['ce'].item()==pytest.approx(math.log(20))
    assert result['embedding_mse'].item()==pytest.approx(38/6)
    assert result['residue_accuracy'].item()==pytest.approx(4/6)
    assert result['sequence_accuracy'].item()==pytest.approx((1/2+3/4)/2)
    assert result['loss'].item()==pytest.approx(math.log(20)+0.25*38/6)
    extended_logits=torch.cat([logits.detach(),torch.full((2,3,20),float('nan'))],1)
    extended_reconstructed=torch.cat([reconstructed.detach(),torch.full((2,3,2),float('inf'))],1)
    extended_target=torch.cat([target,torch.full((2,3,2),float('nan'))],1)
    extended_labels=torch.cat([labels,torch.full((2,3),-100,dtype=torch.long)],1)
    extended_mask=torch.cat([mask,torch.zeros(2,3,dtype=torch.bool)],1)
    padded=reconstruction_loss(extended_logits,extended_reconstructed,extended_target,
                               extended_labels,extended_mask,lambda_emb=0.25)
    for key in ('loss','ce','embedding_mse','residue_accuracy','sequence_accuracy'):
        torch.testing.assert_close(padded[key],result[key])
    result['loss'].backward()
    assert not logits.grad[~mask].any() and not reconstructed.grad[~mask].any()


def test_metric_aggregation_weights_residues_and_sequences_not_batches():
    accumulator=MetricAccumulator()
    for length,correct in ((10,10),(30,0)):
        mask=torch.ones(1,length,dtype=torch.bool)
        labels=torch.zeros(1,length,dtype=torch.long) if correct else torch.ones(1,length,dtype=torch.long)
        r=reconstruction_loss(torch.zeros(1,length,20),torch.ones(1,length,2),
                              torch.zeros(1,length,2),labels,mask)
        accumulator.update(r)
    result=accumulator.compute(0)
    assert result['residue_accuracy']==0.25 and result['sequence_accuracy']==0.5
    assert result['residue_count']==40 and result['sequence_count']==2


def test_resume_restores_every_state_and_exact_future_training(tmp_path):
    train,val=fixture_data()
    cfg=fixture_config()
    first=TrainingEngine(cfg,train,val,data_provenance=provenance())
    for _ in range(3): first.train_step()
    path=tmp_path/'resume.pt'
    first.save(path)
    state=load_checkpoint(path)
    assert state['step']==3 and state['optimizer']['state'] and state['scheduler'] is not None
    assert state['sampler']['cursor']==5 and state['scaler'] is None
    batch=first._batch(val,[0,1])
    first.model.eval()
    with torch.no_grad(): before=first.model(batch['embeddings'],batch['input_mask'])['logits'].clone()
    expected_random=(random.random(),np.random.random(),torch.rand(4))
    expected_steps=[first.train_step() for _ in range(5)]
    resumed=TrainingEngine(cfg,train,val,data_provenance=provenance(),resume=path)
    assert resumed.step==3
    assert_nested_equal(resumed.optimizer.state_dict(),state['optimizer'])
    assert_nested_equal(resumed.scheduler.state_dict(),state['scheduler'])
    assert_nested_equal(resumed.sampler.state_dict(),state['sampler'])
    resumed.model.eval()
    with torch.no_grad(): after=resumed.model(batch['embeddings'],batch['input_mask'])['logits']
    torch.testing.assert_close(after,before,rtol=0,atol=0)
    actual_random=(random.random(),np.random.random(),torch.rand(4))
    assert_nested_equal(actual_random,expected_random)
    actual_steps=[resumed.train_step() for _ in range(5)]
    assert_nested_equal(actual_steps,expected_steps)
    assert_nested_equal(resumed.model.state_dict(),first.model.state_dict())
    assert_nested_equal(resumed.optimizer.state_dict(),first.optimizer.state_dict())
    assert_nested_equal(resumed.scheduler.state_dict(),first.scheduler.state_dict())
    assert resumed.step==8


def test_init_checkpoint_only_loads_weights_and_is_mutually_exclusive(tmp_path):
    train,val=fixture_data()
    cfg=fixture_config(3)
    engine=TrainingEngine(cfg,train,val,data_provenance=provenance())
    engine.train_step()
    path=tmp_path/'weights.pt';engine.save(path)
    initialized=TrainingEngine(cfg,train,val,data_provenance=provenance(),init_checkpoint=path)
    assert_nested_equal(initialized.model.state_dict(),engine.model.state_dict())
    assert initialized.step==0 and not initialized.optimizer.state
    assert initialized.sampler.cursor==0
    assert initialized.scheduler.last_epoch==0
    with pytest.raises(ValueError,match='mutually exclusive'):
        TrainingEngine(cfg,train,val,data_provenance=provenance(),resume=path,init_checkpoint=path)


def test_eval_mode_no_grad_rng_and_mask_alignment():
    train,val=fixture_data()
    engine=TrainingEngine(fixture_config(),train,val,data_provenance=provenance())
    captured=[]
    def capture(module,args,kwargs):
        captured.append((module.training,torch.is_grad_enabled()))
        assert len(args)==2 and set(kwargs)=={'output_lengths','condition'}
        assert torch.equal(kwargs['output_lengths'],args[1].sum(1))
        assert kwargs['condition'] is None
    handle=engine.model.register_forward_pre_hook(capture,with_kwargs=True)
    state=capture_rng(engine.device)
    result=engine.evaluate()
    handle.remove()
    assert captured and all(not training and not enabled for training,enabled in captured)
    assert engine.model.training and all(p.grad is None for p in engine.model.parameters())
    assert_nested_equal(state,capture_rng(engine.device))
    assert result['sequence_count']==3 and result['residue_count']==36


def test_budget_logging_clipping_and_resume_with_larger_total_budget(tmp_path):
    train,val=fixture_data()
    engine=TrainingEngine(fixture_config(3),train,val,data_provenance=provenance())
    summary=engine.fit(tmp_path/'first')
    rows=[json.loads(line) for line in (tmp_path/'first/metrics.jsonl').read_text().splitlines()]
    assert len([r for r in rows if r['event']=='train'])==3
    assert summary['step']==summary['max_steps']==3
    assert not summary['esm_alignment_supervised']
    assert all(r['data_kind']=='synthetic' for r in rows)
    assert torch.sqrt(sum(p.grad.square().sum() for p in engine.model.parameters())).item()<=0.10001
    with pytest.raises(RuntimeError,match='budget exhausted'): engine.train_step()
    larger=TrainingEngine(fixture_config(5),train,val,data_provenance=provenance(),resume=tmp_path/'first/last.pt')
    result=larger.fit(tmp_path/'continued')
    rows=[json.loads(line) for line in (tmp_path/'continued/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows if r['event']=='train']==[4,5]
    assert result['step']==5


@pytest.mark.parametrize('change',['source','aa_order','data','settings','budget'])
def test_resume_rejects_incompatible_state(tmp_path,change):
    train,val=fixture_data()
    cfg=fixture_config(3)
    engine=TrainingEngine(cfg,train,val,data_provenance=provenance())
    engine.train_step();engine.train_step()
    path=tmp_path/'checkpoint.pt';engine.save(path)
    prov=provenance()
    if change=='source': prov['esm_source']={'source':'mock:other'}
    elif change=='aa_order':
        state=load_checkpoint(path);state['aa_order']='OTHER';torch.save(state,path)
    elif change=='data': prov['train']={'fixture':'changed-data'}
    elif change=='settings': cfg['training']['batch_size']=3
    elif change=='budget': cfg['training']['max_steps']=1
    with pytest.raises(ValueError):
        TrainingEngine(cfg,train,val,data_provenance=prov,resume=path)


@pytest.mark.parametrize('settings',[{'max_steps':None},{'max_steps':0},{'max_steps':-1},
                                    {'max_steps':True},{'max_steps':1,'batch_size':0},
                                    {'max_steps':1,'precision':'fp16'}])
def test_invalid_budget_and_precision(settings):
    with pytest.raises(ValueError): TrainingSettings(**settings)


def test_unsupported_stage_and_bf16_cpu_are_explicit():
    with pytest.raises(NotImplementedError): TrainingSettings(stage='1B',max_steps=1)
    train,val=fixture_data()
    with pytest.raises(RuntimeError,match='requires a CUDA'):
        TrainingEngine(fixture_config(1,precision='bf16'),train,val,data_provenance=provenance())


def test_cli_help_imports_no_torch_or_esm_and_creates_nothing(tmp_path):
    code='''import importlib.abc, runpy, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','esm','raygun','yaml'}:
            raise AssertionError('Unexpected import '+fullname)
sys.meta_path.insert(0,BlockHeavy())
sys.argv=['templatedf.train','--help']
runpy.run_module('templatedf.train',run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--max-steps' in result.stdout and '--resume' in result.stdout
    assert list(tmp_path.iterdir())==[]


def test_fresh_run_does_not_overwrite_checkpoint_without_log(tmp_path):
    import hashlib
    train,val=fixture_data()
    engine=TrainingEngine(fixture_config(1),train,val,data_provenance=provenance())
    path=tmp_path/'last.pt'
    engine.save(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError,match='Existing run artifacts'):
        engine.fit(tmp_path)
    assert engine.step==0
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert not (tmp_path/'metrics.jsonl').exists()
