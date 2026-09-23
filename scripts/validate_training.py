"""Stage 04: fixed synthetic fit, exact resume, and bounded cached BF16 CLI."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import sys

import torch
import yaml

from templatedf.data import AA_ORDER, PeptideRecord
from templatedf.feature_cache import CacheSpec, FeatureCacheWriter
from templatedf.losses import reconstruction_loss
from templatedf.training import TrainingEngine


def max_state_difference(a,b):
    if isinstance(a,torch.Tensor):
        return (a.cpu().double()-b.cpu().double()).abs().max().item() if a.numel() else 0.0
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max((max_state_difference(a[k],b[k]) for k in a),default=0.0)
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        return max((max_state_difference(x,y) for x,y in zip(a,b)),default=0.0)
    assert a==b
    return 0.0


def make_samples(lengths,dim):
    result=[]
    for label,length in enumerate(lengths):
        embeddings=torch.zeros(length,dim)
        embeddings[:,label]=2.0
        embeddings[:,20]=torch.arange(length).float()/150
        result.append({'embeddings':embeddings,'labels':torch.full((length,),label,dtype=torch.long)})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--bf16-device',default='cuda:0')
    args=parser.parse_args()
    root=args.output_root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    logs=args.report.parent
    logs.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    train=make_samples([10,12,14],32)
    validation=make_samples([11,13,15],32)
    config={'model':{'dim':32,'num_heads':4,'encoder_blocks':1,'decoder_blocks':1,'dropout':0.1},
            'training':{'stage':'1A','device':'cpu','precision':'fp32','batch_size':3,
                        'max_steps':60,'learning_rate':0.003,'weight_decay':0.0,'grad_clip':1.0,
                        'seed':404,'eval_every':10,'save_every':20,'scheduler_gamma':0.99}}
    (root/'small_config.yaml').write_text(yaml.safe_dump(config))
    provenance={'data_kind':'synthetic','esm_source':{'source':'mock:onehot-plus-position-v1','dim':32},
                'train':{'labels':[0,1,2],'lengths':[10,12,14]},
                'validation':{'labels':[0,1,2],'lengths':[11,13,15]}}
    print('Starting bounded synthetic CPU fit: 60 optimizer steps.',flush=True)
    engine=TrainingEngine(config,train,validation,data_provenance=provenance)
    summary=engine.fit(root/'cpu_fit')
    rows=[json.loads(line) for line in (root/'cpu_fit/metrics.jsonl').read_text().splitlines()]
    train_rows=[r for r in rows if r['event']=='train']
    assert len(train_rows)==60 and train_rows[-1]['loss']<train_rows[0]['loss']*0.8
    engine.model.eval()
    batch=engine._batch(validation,[0,1,2])
    with torch.no_grad(): saved_output=engine.model(batch['embeddings'],batch['input_mask'])['logits'].clone()
    checkpoint=root/'cpu_fit/last.pt'
    # Explicit bounded comparison: three further updates, then replay the same three.
    engine.settings.max_steps=63
    engine.config['training']['max_steps']=63
    expected_steps=[engine.train_step() for _ in range(3)]
    resumed_config=deepcopy(config);resumed_config['training']['max_steps']=63
    resumed=TrainingEngine(resumed_config,train,validation,data_provenance=provenance,resume=checkpoint)
    step_at_restore=resumed.step
    assert step_at_restore==60 and resumed.optimizer.state and resumed.scheduler is not None
    resumed.model.eval()
    with torch.no_grad(): loaded_output=resumed.model(batch['embeddings'],batch['input_mask'])['logits']
    output_error=(saved_output-loaded_output).abs().max().item()
    assert output_error==0
    resumed_summary=resumed.fit(root/'cpu_resumed')
    actual_rows=[json.loads(line) for line in (root/'cpu_resumed/metrics.jsonl').read_text().splitlines()]
    actual_steps=[r for r in actual_rows if r['event']=='train']
    assert [r['step'] for r in actual_steps]==[61,62,63]
    assert [r['sample_indices'] for r in actual_steps]==[r['sample_indices'] for r in expected_steps]
    model_error=max_state_difference(engine.model.state_dict(),resumed.model.state_dict())
    optimizer_error=max_state_difference(engine.optimizer.state_dict(),resumed.optimizer.state_dict())
    scheduler_error=max_state_difference(engine.scheduler.state_dict(),resumed.scheduler.state_dict())
    assert model_error==optimizer_error==scheduler_error==0
    initializer=TrainingEngine(config,train,validation,data_provenance=provenance,init_checkpoint=checkpoint)
    assert initializer.step==0 and not initializer.optimizer.state and initializer.sampler.cursor==0
    report={'input_kind':'SYNTHETIC only; 3 constant-AA sequences, one-hot + position features; not ESM',
            'small_config':config,'small_summary':summary,'resumed_summary':resumed_summary,
            'fit_steps':60,'first_train_loss':train_rows[0]['loss'],'last_train_loss':train_rows[-1]['loss'],
            'curve':[{'step':r['step'],'loss':r['loss'],'ce':r['ce'],'embedding_mse':r['embedding_mse'],
                      'residue_accuracy':r['residue_accuracy'],'sequence_accuracy':r['sequence_accuracy']}
                     for r in train_rows if r['step']==1 or r['step']%10==0],
            'checkpoint_roundtrip':{'step_at_restore':step_at_restore,'resumed_final_step':resumed.step,
                'eval_output_max_abs_error':output_error,'future_model_max_abs_error':model_error,
                'future_optimizer_max_abs_error':optimizer_error,'future_scheduler_max_abs_error':scheduler_error,
                'three_future_batches_match':True,'init_checkpoint_step':initializer.step,
                'init_checkpoint_optimizer_empty':not bool(initializer.optimizer.state)}}
    mask=torch.tensor([[True,True,False,False],[True,True,True,True]])
    manual=reconstruction_loss(torch.zeros(2,4,20),torch.tensor([[[1.,1.]]*4,[[3.,3.]]*4]),
                               torch.zeros(2,4,2),torch.tensor([[0,1,-100,-100],[0,0,0,1]]),mask,lambda_emb=0.25)
    report['manual_loss']={k:float(manual[k]) for k in ('loss','ce','embedding_mse','residue_accuracy','sequence_accuracy')}
    if torch.cuda.is_available():
        device=torch.device(args.bf16_device)
        with torch.cuda.device(device):
            supported=torch.cuda.is_bf16_supported(including_emulation=False)
        if not supported: raise RuntimeError('Selected GPU does not support native BF16')
        # Exercise the actual cache CLI with synthetic 1280-D records, never ESM.
        spec=CacheSpec(source='mock:stage04-onehot-plus-position-v1')
        (root/'cache_spec.json').write_text(json.dumps(asdict(spec),indent=2)+'\n')
        cache_config=deepcopy(config)
        cache_config['model']={'dim':1280,'num_heads':20,'encoder_blocks':1,'decoder_blocks':1,'dropout':0.1}
        cache_config['training'].update(device=str(device),precision='bf16',max_steps=None,batch_size=2,
                                         eval_every=1,save_every=1,learning_rate=0.0001,scheduler_gamma=0.99)
        cache_config['data']={'train_path':str(root/'train.fasta'),'validation_path':str(root/'validation.fasta')}
        cache_config['cache']={'train_dir':str(root/'train_cache'),'validation_dir':str(root/'validation_cache'),
                               'spec_path':str(root/'cache_spec.json'),'read_dtype':'float16','storage_dtype':'float16'}
        for split,lengths in (('train',[10,12]),('validation',[11,13])):
            records=[PeptideRecord(f'synthetic_{split}_{i}',AA_ORDER[i]*length) for i,length in enumerate(lengths)]
            (root/f'{split}.fasta').write_text(''.join(f'>{r.id}\n{r.sequence}\n' for r in records))
            with FeatureCacheWriter(root/f'{split}_cache',spec) as cache:
                for record,item in zip(records,make_samples(lengths,1280)):
                    cache.write(record,item['embeddings'])
        config_path=root/'cached_bf16_config.yaml'
        config_path.write_text(yaml.safe_dump(cache_config))
        gpu_dir=root/'cached_bf16'
        cli_runs=[]
        for name,budget,resume_args in [('cached_bf16',1,[]),('cached_bf16_resume',2,['--resume',str(gpu_dir/'last.pt')])]:
            cmd=[sys.executable,'-m','templatedf.train','--config',str(config_path),
                 '--output-dir',str(gpu_dir),'--max-steps',str(budget),*resume_args]
            print('Starting '+name+' with cumulative max_steps='+str(budget),flush=True)
            result=subprocess.run(cmd,capture_output=True,text=True)
            output='$ '+shlex.join(cmd)+'\n'+result.stdout+result.stderr+f'\nEXIT_CODE={result.returncode}\n'
            (logs/f'{name}.log').write_text(output)
            if result.returncode:
                print(output,flush=True)
                raise RuntimeError(name+' failed')
            cli_runs.append(json.loads((gpu_dir/'summary.json').read_text()))
        assert cli_runs[0]['step']==1 and cli_runs[1]['step']==2
        assert cli_runs[1]['precision']['logits_dtype']=='torch.bfloat16'
        gpu_rows=[json.loads(line) for line in (gpu_dir/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in gpu_rows if r['event']=='train']==[1,2]
        report['cached_bf16']={'gpu_name':torch.cuda.get_device_name(device),'cli_runs':cli_runs,
                               'train_events':2,'data_kind':'synthetic','model_dim':1280,
                               'encoder_blocks':1,'decoder_blocks':1}
    else:
        report['cached_bf16']={'status':'UNVERIFIED: no CUDA GPU'}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
