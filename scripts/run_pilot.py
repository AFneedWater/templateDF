"""Bounded stage 05 real-data pilot; explicit actions, no automatic full training."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/05_pilot'
LOG = ROOT / 'reports/05_logs'
BINS = ((10, 29), (30, 49), (50, 99), (100, 150))
SEED = 505


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def bucket(length):
    for lo, hi in BINS:
        if lo <= length <= hi:
            return f'{lo}-{hi}'
    raise ValueError(f'Invalid peptide length: {length}')


def distribution(records):
    counts = Counter(bucket(r.length) for r in records)
    return {f'{lo}-{hi}': counts[f'{lo}-{hi}'] for lo, hi in BINS}


def quotas(counts, total):
    """Largest-remainder allocation preserves source proportions deterministically."""
    size = sum(counts.values())
    if size < total:
        raise ValueError('Insufficient records for the requested bounded sample')
    values = {k: v * total // size for k, v in counts.items()}
    order = sorted(counts, key=lambda k: (-(counts[k] * total % size), k))
    for key in order[:total - sum(values.values())]:
        values[key] += 1
    return values


def write_fasta(path, records):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(''.join(f'>{r.description or r.id}\n{r.sequence}\n' for r in records))


def prepare(validation_source):
    if validation_source != 'holdout-from-train':
        raise ValueError('Explicit confirmed validation source required')
    from templatedf.data import load_fasta, audit_splits, file_sha256
    import yaml
    cfg = yaml.safe_load((ROOT / 'configs/train.yaml').read_text())
    train_path = Path(cfg['data']['train_path'])
    test_path = Path(cfg['data']['test_path'])
    records, stats = load_fasta(train_path)
    if stats.retained_duplicate_sequences or stats.duplicate_ids:
        raise ValueError('Resolve source duplicates before sampling')
    source_audit = audit_splits({'source_train': train_path, 'heldout_test': test_path})
    if source_audit['overlaps']['source_train/heldout_test']['unique_sequences']:
        raise ValueError('Source train/test exact overlap must be resolved')
    rng = random.Random(SEED)
    source_counts = distribution(records)
    val_counts = quotas(source_counts, 128)
    train_counts = quotas({k: source_counts[k] - val_counts[k] for k in source_counts}, 512)
    selected = {'train': [], 'validation': []}
    for key in source_counts:
        group = [r for r in records if bucket(r.length) == key]
        rng.shuffle(group)
        selected['validation'].extend(group[:val_counts[key]])
        selected['train'].extend(group[val_counts[key]:val_counts[key] + train_counts[key]])
    for group in selected.values():
        rng.shuffle(group)
    ART.mkdir(parents=True, exist_ok=True)
    for name, group in selected.items():
        write_fasta(ART / f'{name}.fasta', group)
    audit = audit_splits({'train': ART/'train.fasta', 'validation': ART/'validation.fasta', 'heldout_test': test_path})
    assert all(v['unique_sequences'] == 0 for v in audit['overlaps'].values())
    report = {'seed': SEED, 'method': 'proportional length-stratified holdout from user train; largest remainder quotas',
              'validation_is_independent_of_pilot_training': True, 'homology_controlled': False,
              'source_train': str(train_path), 'source_train_sha256': file_sha256(train_path),
              'source_train_stats': asdict(stats), 'source_distribution': source_counts,
              'heldout_test_distribution': {f'{lo}-{hi}': sum(n for length,n in source_audit['splits']['heldout_test']['length_histogram'].items() if lo <= int(length) <= hi) for lo,hi in BINS},
              'selected_distribution': {k: distribution(v) for k,v in selected.items()},
              'selected': {k: [{'id':r.id,'sequence_hash':r.sequence_hash,'length':r.length} for r in v] for k,v in selected.items()},
              'audit': audit,
              'future_training_requirement': 'Exclude these validation sequence hashes from future use of the full source training file.'}
    dump(LOG/'data_selection.json', report)
    dump(LOG/'source_audit.json', source_audit)
    print(json.dumps(report['selected_distribution']), flush=True)


def extract():
    import torch
    from templatedf.data import load_fasta
    from templatedf.esm_features import ESMFeatureExtractor, local_cache_spec, DEFAULT_CHECKPOINT
    from templatedf.feature_cache import FeatureCache
    checkpoint = DEFAULT_CHECKPOINT
    regression = checkpoint.with_name(checkpoint.stem + '-contact-regression.pt')
    start = time.perf_counter()
    spec = local_cache_spec(checkpoint, regression)
    dump(ART/'cache_spec.json', asdict(spec))
    cache = FeatureCache(ART/'cache', spec)
    records = [r for name in ('train','validation') for r in load_fasta(ART/f'{name}.fasta')[0]]
    if len(records) > 640:
        raise ValueError('Stage 05 cache budget exceeded')
    extractor = ESMFeatureExtractor.from_local(checkpoint, regression, device='cuda:0')
    torch.cuda.synchronize(0)
    loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    reused = 0
    checked = []
    # Length ordering reduces padding; sampled FASTA ordering is left unchanged.
    records.sort(key=lambda r:r.length)
    for start_index in range(0,len(records),8):
        group=records[start_index:start_index+8]
        pending=[]
        for record in group:
            if cache.path_for(record).exists():
                cache.read(record,dtype=torch.float32)
                reused+=1
            else:
                pending.append(record)
        if pending:
            features=extractor.extract(pending)
            for record,feature in zip(pending,features):
                assert feature.shape == (record.length,1280) and feature.dtype == torch.float32
                assert not feature.requires_grad
                cache.write(record,feature)
                sample=cache.read(record,dtype=torch.float32)
                assert torch.equal(sample['embeddings'],feature.half().float())
                checked.append({'id':record.id,'length':record.length,'shape':list(feature.shape),'compute_dtype':str(feature.dtype)})
        if (start_index//8)%10==0:
            print(f'ESM cache {min(start_index+8,len(records))}/{len(records)}',flush=True)
    torch.cuda.synchronize(0)
    end=time.perf_counter()
    assert not extractor.model.training and all(not p.requires_grad for p in extractor.model.parameters())
    report={'data_kind':'real_cached','device':'cuda:0','gpu':torch.cuda.get_device_name(0),
            'records':len(records),'reused':reused,'new_entries':len(checked),'batch_size':8,
            'weights_hashing_and_model_load_seconds':loaded-start,'extraction_and_cache_io_seconds':end-loaded,
            'total_seconds':end-start,'frozen':True,'eval':True,'residue_slice':'row i, 1:1+L_i',
            'cache_roundtrip_checked':True,'compute_dtype':'torch.float32','storage_dtype':spec.storage_dtype,
            'max_allocated_bytes':torch.cuda.max_memory_allocated(0),'max_reserved_bytes':torch.cuda.max_memory_reserved(0),
            'spec':asdict(spec),'checked_shapes':checked}
    dump(LOG/'cache_extraction.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('spec','checked_shapes')}),flush=True)


def config_for(batch_size, steps):
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/train.yaml').read_text())
    assert (cfg['model']['dim'],cfg['model']['num_latents'],cfg['model']['encoder_blocks'],cfg['model']['decoder_blocks']) == (1280,50,4,4)
    cfg['data'].update(train_path=str(ART/'train.fasta'),validation_path=str(ART/'validation.fasta'))
    cfg['cache'].update(train_dir=str(ART/'cache'),validation_dir=str(ART/'cache'),spec_path=str(ART/'cache_spec.json'))
    cfg['training'].update(device='cuda:0',precision='bf16',batch_size=batch_size,max_steps=steps,eval_every=25,save_every=100,lambda_emb=0.0)
    return cfg


def datasets():
    import torch
    from templatedf.data import load_fasta,file_sha256
    from templatedf.feature_cache import CacheSpec,FeatureCache,CachedPeptideDataset
    spec=CacheSpec(**json.loads((ART/'cache_spec.json').read_text()))
    cache=FeatureCache(ART/'cache',spec)
    ds={name:CachedPeptideDataset(load_fasta(ART/f'{name}.fasta')[0],cache,dtype=torch.float32) for name in ('train','validation')}
    assert len(ds['train'])<=512 and len(ds['validation'])<=128
    assert not ({r.sequence_hash for r in ds['train'].records}&{r.sequence_hash for r in ds['validation'].records})
    provenance={'data_kind':'real_cached','esm_source':asdict(spec),**{name:{'file_sha256':file_sha256(ART/f'{name}.fasta'),'retained':len(ds[name])} for name in ds}}
    return ds,provenance


def runtime_info():
    import torch,platform
    return {'gpu':torch.cuda.get_device_name(0),'device':'cuda:0','total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'free_memory_bytes_before_model':torch.cuda.mem_get_info(0)[0],'torch':torch.__version__,
            'cuda':torch.version.cuda,'python':platform.python_version(),'cpu_threads':torch.get_num_threads()}


def measured_engine_class():
    import torch
    from templatedf.training import TrainingEngine
    from templatedf.checkpoint import capture_rng,restore_rng
    from templatedf.losses import MetricAccumulator
    class MeasuredEngine(TrainingEngine):
        def train_step(self):
            torch.cuda.synchronize(self.device)
            start=time.perf_counter()
            values=super().train_step()
            torch.cuda.synchronize(self.device)
            values['optimizer_step_seconds_including_cache_io']=time.perf_counter()-start
            if self.step%10==0 or self.step==1:
                print(f"step={self.step}/{self.settings.max_steps} CE={values['ce']:.5f} residue_acc={values['residue_accuracy']:.4f} seconds={values['optimizer_step_seconds_including_cache_io']:.3f}",flush=True)
            return values

        @torch.no_grad()
        def evaluate(self):
            rng=capture_rng(self.device)
            mode=self.model.training
            self.model.eval()
            total=MetricAccumulator()
            groups={f'{lo}-{hi}':[] for lo,hi in BINS}
            for i,r in enumerate(self.validation_dataset.records):
                groups[bucket(r.length)].append(i)
            result={}
            try:
                for key,indices in groups.items():
                    if not indices:
                        result[key]=None
                        continue
                    accum=MetricAccumulator()
                    for start in range(0,len(indices),self.settings.batch_size):
                        with self._autocast():
                            values=self._forward_loss(self._batch(self.validation_dataset,indices[start:start+self.settings.batch_size]))
                        accum.update(values)
                        total.update(values)
                    result[key]=accum.compute(self.settings.lambda_emb)
                return {**total.compute(self.settings.lambda_emb),'by_length':result}
            finally:
                self.model.train(mode)
                restore_rng(rng,self.device)
    return MeasuredEngine


def benchmark(batch_size):
    import torch,yaml
    from templatedf.feature_cache import CachedPeptideDataset
    ds,provenance=datasets()
    longest=sorted(ds['train'].records,key=lambda r:r.length,reverse=True)[:batch_size]
    ds['train']=CachedPeptideDataset(longest,ds['train'].cache,dtype=torch.float32)
    provenance['train'].update(benchmark_selected_ids=[r.id for r in longest],retained=batch_size)
    cfg=config_for(batch_size,40)
    path=LOG/f'benchmark_batch{batch_size}.json'
    if path.exists():
        raise FileExistsError('Benchmark already recorded; refuse duplicate budget')
    (ART/f'benchmark_batch{batch_size}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    info=runtime_info()
    torch.cuda.reset_peak_memory_stats(0)
    engine=None
    rows=[]
    try:
        engine=measured_engine_class()(cfg,ds['train'],ds['validation'],data_provenance=provenance)
        parameter=next(engine.model.parameters())
        before=parameter.detach().clone()
        for _ in range(40):
            rows.append(engine.train_step())
            if len(rows)==1:
                first_delta=float((parameter.detach()-before).abs().max())
                assert first_delta>0
        seconds=sum(r['optimizer_step_seconds_including_cache_io'] for r in rows[10:])
        report={'status':'passed','runtime':info,'model_config':engine.model_config,
                'parameter_count':sum(p.numel() for p in engine.model.parameters()),'batch_size':batch_size,
                'lengths':[r.length for r in longest],'warmup_steps':10,'measured_steps':30,
                'measured_seconds':seconds,'peptides_per_second':batch_size*30/seconds,
                'residues_per_second':sum(r['residue_count'] for r in rows[10:])/seconds,
                'max_allocated_bytes':torch.cuda.max_memory_allocated(0),'max_reserved_bytes':torch.cuda.max_memory_reserved(0),
                'first_optimizer_step_parameter_max_delta':first_delta,'precision':engine.precision_runtime,
                'optimizer_state_entries':len(engine.optimizer.state),'gradient_checkpointing':False,'rows':rows,
                'weights_discarded_after_benchmark':True}
        dump(path,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('rows','model_config')}),flush=True)
    except torch.OutOfMemoryError as exc:
        dump(path,{'status':'oom','runtime':info,'batch_size':batch_size,'model_config':cfg['model'],
                   'completed_steps':len(rows),'error':str(exc),'max_allocated_bytes':torch.cuda.max_memory_allocated(0),
                   'max_reserved_bytes':torch.cuda.max_memory_reserved(0),'rows':rows})
        raise


def train(batch_size):
    import torch,yaml
    from templatedf.data import file_sha256
    benchmark_result=json.loads((LOG/f'benchmark_batch{batch_size}.json').read_text())
    if benchmark_result['status']!='passed':
        raise ValueError('A passed full-step capacity benchmark is required')
    run=ART/'ce_run'
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('Existing pilot; do not silently restart the step budget')
    cfg=config_for(batch_size,300)
    cfg_path=ART/'ce_config.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    ds,provenance=datasets()
    info=runtime_info()
    torch.cuda.reset_peak_memory_stats(0)
    engine=measured_engine_class()(cfg,ds['train'],ds['validation'],data_provenance=provenance)
    parameter=next(engine.model.parameters())
    before=parameter.detach().clone()
    torch.cuda.synchronize(0)
    start=time.perf_counter()
    summary=engine.fit(run)
    torch.cuda.synchronize(0)
    elapsed=time.perf_counter()-start
    summary.update(runtime=info,elapsed_training_validation_checkpoint_seconds=elapsed,
                   model_config=engine.model_config,parameter_count=sum(p.numel() for p in engine.model.parameters()),
                   parameter_max_delta=float((parameter.detach()-before).abs().max()),
                   all_parameters_finite=all(bool(torch.isfinite(p).all()) for p in engine.model.parameters()),
                   max_allocated_bytes=torch.cuda.max_memory_allocated(0),max_reserved_bytes=torch.cuda.max_memory_reserved(0),
                   checkpoint_sha256=file_sha256(run/'last.pt'),checkpoint_bytes=(run/'last.pt').stat().st_size,
                   optimizer_state_entries=len(engine.optimizer.state),gradient_checkpointing=False,
                   benchmark_weights_reused=False,configuration=str(cfg_path))
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    steps=[r for r in rows if r['event']=='train']
    assert [r['step'] for r in steps]==list(range(1,301))
    assert summary['parameter_max_delta']>0 and summary['all_parameters_finite']
    seconds=sum(r['optimizer_step_seconds_including_cache_io'] for r in steps)
    summary.update(optimizer_steps_seconds=seconds,peptides_per_second=sum(r['sequence_count'] for r in steps)/seconds,
                   residues_per_second=sum(r['residue_count'] for r in steps)/seconds)
    dump(run/'summary.json',summary)
    dump(LOG/'pilot_result.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','extract','benchmark','train'))
    parser.add_argument('--validation-source',choices=('holdout-from-train',))
    parser.add_argument('--batch-size',type=int,choices=(8,4,2),default=8)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.validation_source)
    elif args.action=='extract':
        extract()
    elif args.action=='benchmark':
        benchmark(args.batch_size)
    else:
        train(args.batch_size)


if __name__=='__main__':
    main()
