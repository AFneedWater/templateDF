"""Bounded stage 06 demonstration using five confirmed held-out real templates."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/05_pilot/ce_run/last.pt')
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/06_demo')
    parser.add_argument('--log-dir',type=Path,default=ROOT/'reports/06_logs')
    args=parser.parse_args()
    import torch
    from templatedf.data import load_fasta,read_fasta,file_sha256
    from templatedf.inference import TemplateGenerator
    from templatedf.inference_io import inspect_fasta,write_candidates,export_raw_latents,iter_latent_records
    output=args.output_root.resolve();logs=args.log_dir.resolve()
    output.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=True)
    validation_path=ROOT/'artifacts/05_pilot/validation.fasta'
    validation,_=load_fasta(validation_path)
    pool=list(validation);records=[]
    for target in (10,25,40,80,150):
        record=min(pool,key=lambda r:(abs(r.length-target),r.id))
        records.append(record);pool.remove(record)
    fasta=output/'templates.fasta'
    fasta.write_text(''.join(f'>{r.description or r.id}\n{r.sequence}\n' for r in records))
    initial_hash=file_sha256(args.checkpoint)
    selection={'source':str(validation_path),'source_sha256':file_sha256(validation_path),
               'method':'One nearest available length to each of 10/25/40/80/150; unique selection from confirmed validation set',
               'templates':[{'id':r.id,'length':r.length,'sequence_hash':r.sequence_hash,'sequence':r.sequence} for r in records]}
    dump(logs/'template_selection.json',selection)
    def cli(module,directory,extra):
        cmd=[sys.executable,'-u','-m','templatedf.'+module,'--checkpoint',str(args.checkpoint.resolve()),
             '--fasta',str(fasta),'--output-dir',str(output/directory),'--limit','5','--batch-size','2',
             '--device','cuda:0','--precision','bf16',*extra]
        start=time.perf_counter()
        with (logs/f'{module}.log').open('w') as f:
            f.write('COMMAND: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 '+' '.join(cmd)+'\n');f.flush()
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
            f.write(f'\nEXIT_CODE={result.returncode}\n')
        if result.returncode:
            raise RuntimeError((logs/f'{module}.log').read_text())
        print(f'{module} CLI complete',flush=True)
        return {'command':cmd,'elapsed_seconds':time.perf_counter()-start,'exit_code':result.returncode}
    runs={}
    runs['reconstruct']=cli('reconstruct','reconstruction',[])
    runs['generate']=cli('generate','sampled_80',['--target-length','80','--temperature','0.8','--num-samples','3','--seed','626'])
    runs['export_latents']=cli('export_latents','latents',['--shard-size','2'])
    runtime=TemplateGenerator.from_checkpoint(args.checkpoint,device='cuda:0',precision='bf16')
    rng=torch.get_rng_state().clone()
    info=inspect_fasta(fasta,5)
    settings=dict(num_samples=3,temperature=.8,seed=626,batch_size=2)
    for name,length,temperature in [('sampled_80_repeat',80,.8),('sampled_10',10,.8),('sampled_150',150,.8),
                                    ('sampled_default',None,.8),('argmax_repeated',None,0.)]:
        write_candidates(runtime,records,output/name,expected_records=5,input_info=info,
                         **{**settings,'target_length':length,'temperature':temperature})
    assert torch.equal(rng,torch.get_rng_state())
    def metadata(name):return [json.loads(line) for line in (output/name/'metadata.jsonl').read_text().splitlines()]
    assert metadata('sampled_80')==metadata('sampled_80_repeat')
    export_raw_latents(runtime,records,output/'latents_repeat',expected_records=5,shard_size=2,batch_size=2,input_info=info)
    first=list(iter_latent_records(output/'latents',expected_identity=runtime.identity))
    second=list(iter_latent_records(output/'latents_repeat',expected_identity=runtime.identity))
    assert len(first)==len(second)==5
    latent_diff=max(float((a['latent'].float()-b['latent'].float()).abs().max()) for a,b in zip(first,second))
    assert latent_diff==0
    raw_diff=logit_diff=0.
    for start in range(0,5,2):
        chunk=records[start:start+2]
        direct=runtime.encode_records(chunk)
        saved=torch.stack([r['latent'] for r in first[start:start+2]])
        raw_diff=max(raw_diff,float((direct.cpu().float()-saved.float()).abs().max()))
        lengths=torch.tensor([r.length for r in chunk])
        a=runtime.decode_latents(direct,lengths)['logits']
        b=runtime.decode_latents(saved,lengths)['logits']
        logit_diff=max(logit_diff,float((a.float()-b.float()).abs().max()))
    assert raw_diff==0 and logit_diff==0
    wrong=deepcopy(runtime.identity);wrong['checkpoint_sha256']='0'*64
    try:list(iter_latent_records(output/'latents',expected_identity=wrong))
    except ValueError:pass
    else:raise AssertionError('Wrong checkpoint accepted')
    assert not runtime.model.training and not runtime.extractor.model.training
    assert all(not p.requires_grad for p in runtime.model.parameters())
    assert all(not p.requires_grad for p in runtime.extractor.model.parameters())
    assert file_sha256(args.checkpoint)==initial_hash
    cases={}
    for name in ('reconstruction','sampled_80','sampled_80_repeat','sampled_10','sampled_150','sampled_default','argmax_repeated'):
        manifest=json.loads((output/name/'manifest.json').read_text())
        rows=metadata(name);parsed=list(read_fasta(output/name/'candidates.fasta'))
        assert len(rows)==len(parsed)==manifest['candidate_count']
        for row,seq in zip(rows,parsed):
            record=records[row['template_index']]
            assert row['template_id']==record.id and row['template_sequence_hash']==record.sequence_hash
            assert seq.id==row['candidate_id'] and seq.sequence==row['sequence'] and len(seq.sequence)==row['target_length']
        cases[name]={'candidate_count':len(rows),'target_lengths':sorted({r['target_length'] for r in rows}),
                     'temperature':rows[0]['temperature'],'samples_per_template':manifest['generation']['num_samples'],
                     'within_template_duplicate_rate':manifest['within_template_duplicate_rate'],
                     'global_duplicate_rate':manifest['global_duplicate_rate']}
    latent_manifest=json.loads((output/'latents/manifest.json').read_text())
    recon=metadata('reconstruction');sampled=metadata('sampled_default')
    comparisons=[]
    for index,record in enumerate(records):
        candidate=recon[index]['sequence']
        comparisons.append({'id':record.id,'length':record.length,'template':record.sequence,
                            'argmax':candidate,'sampled':sampled[index*3]['sequence'],
                            'argmax_residue_accuracy':sum(a==b for a,b in zip(record.sequence,candidate))/record.length})
    report={'checkpoint':runtime.provenance['checkpoint'],'device':'cuda:0','gpu':torch.cuda.get_device_name(0),
            'precision':runtime.precision,'model_config':runtime.provenance['model_config'],
            'source_validation':selection['source'],'template_count':5,'data_kind':'real_templates',
            'cases':cases,'cli_runs':runs,'fixed_seed_repeat_equal':True,'global_cpu_rng_unchanged':True,
            'latent_export':{'records':len(first),'shard_counts':[s['count'] for s in latent_manifest['shards']],
                             'shape':list(first[0]['latent'].shape),'dtype':str(first[0]['latent'].dtype),
                             'normalization':latent_manifest['normalization'],'repeat_max_difference':latent_diff,
                             'direct_vs_saved_max_difference':raw_diff,'reloaded_decode_logits_max_difference':logit_diff,
                             'wrong_checkpoint_rejected':True},
            'ae_and_esm_frozen_eval':True,'checkpoint_unchanged':True,'additional_optimizer_steps':0,
            'comparison':comparisons,'metadata_example':sampled[0],
            'limitations':['Checkpoint is a low-accuracy 300-step 1A pilot.',
                           'Changed-length outputs only validate interfaces, not information/structure/function preservation.',
                           'Temperature sampling is not diffusion; no ranking, screening or 1B training was run.']}
    dump(logs/'generation_validation.json',report)
    print(json.dumps({k:report[k] for k in ('template_count','cases','latent_export','checkpoint_unchanged')},indent=2),flush=True)


if __name__=='__main__':
    main()
