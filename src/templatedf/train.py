"""Train stage 1A from existing caches with an explicit optimizer-step budget."""

import argparse
from pathlib import Path


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/train.yaml'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-steps',type=int,help='Total optimizer-step ceiling, including resumed steps')
    parser.add_argument('--device')
    parser.add_argument('--precision',choices=('fp32','bf16'))
    parser.add_argument('--train-fasta',type=Path)
    parser.add_argument('--validation-fasta',type=Path)
    parser.add_argument('--train-cache',type=Path)
    parser.add_argument('--validation-cache',type=Path)
    parser.add_argument('--cache-spec',type=Path,help='Optional expected CacheSpec; normally read from each cache manifest')
    parser.add_argument('--cache-budget-gib',type=float,help='Combined train+validation resident cache budget; default 40 GiB')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--init-checkpoint',type=Path,help='Only model weights; fresh optimizer, step and RNG')
    source.add_argument('--resume',type=Path,help='Restore optimizer, scheduler, RNG, sample order and step')
    return parser


def main():
    args=build_parser().parse_args()
    # --help and invalid CLI syntax stop before any torch/ESM imports or I/O.
    import json
    from dataclasses import asdict
    import torch
    import yaml
    from .data import AA_ORDER, audit_splits, load_fasta
    from .feature_cache import CacheSpec, CachedPeptideDataset, FeatureCache, load_caches_to_ram
    from .training import TrainingEngine, TrainingSettings, model_settings

    config=yaml.safe_load(args.config.read_text())
    if not isinstance(config,dict):
        raise ValueError('Configuration must be a YAML mapping')
    config.setdefault('training',{})
    for key in ('max_steps','device','precision'):
        value=getattr(args,key)
        if value is not None: config['training'][key]=value
    settings=TrainingSettings(**config['training'])
    model_config=model_settings(config)
    data=config.get('data',{})
    if (data.get('min_seq_len',10),data.get('max_seq_len',150),data.get('aa_order',AA_ORDER),
        data.get('padding_label',-100))!=(10,150,AA_ORDER,-100):
        raise ValueError('Unsupported data label/length contract')
    cache_config=config.get('cache',{})
    if cache_config.get('format_version',2)!=2 or cache_config.get('format','npy_shards')!='npy_shards':
        raise ValueError('Use NPY v2 cache configuration; legacy cache backends are no longer supported')
    paths={
        'train':args.train_fasta or data.get('train_path'),
        'validation':args.validation_fasta or data.get('validation_path'),
        'train_cache':args.train_cache or cache_config.get('train_dir'),
        'validation_cache':args.validation_cache or cache_config.get('validation_dir'),
    }
    if any(value is None for value in paths.values()):
        raise ValueError('Provide separate train/validation FASTA, both NPY cache directories; test_path is never used as validation')
    paths={key:Path(value).resolve() for key,value in paths.items()}
    config.setdefault('data',{}).update(train_path=str(paths['train']),validation_path=str(paths['validation']))
    config.setdefault('cache',{}).update(train_dir=str(paths['train_cache']),
                                       validation_dir=str(paths['validation_cache']))
    if data.get('test_path') and Path(data['test_path']).exists():
        if paths['validation'].samefile(Path(data['test_path'])):
            raise ValueError('Configured held-out test file cannot be used for validation')
    audit=audit_splits({name:paths[name] for name in ('train','validation')},
                       check_sequences=data.get('check_sequences',True))
    if audit['overlaps']['train/validation']['unique_sequences']:
        raise ValueError('Exact train/validation sequence overlap: '+json.dumps(audit['overlaps']['train/validation']))
    caches={name:FeatureCache(paths[name+'_cache']) for name in ('train','validation')}
    spec=caches['train'].spec
    if caches['validation'].spec != spec:
        raise ValueError('Train and validation ESM sources differ')
    expected_spec=args.cache_spec or cache_config.get('spec_path')
    if expected_spec:
        metadata=json.loads(Path(expected_spec).read_text())
        if CacheSpec(**metadata.get('spec',metadata)) != spec:
            raise ValueError('Explicit expected ESM source differs from cache manifests')
    if model_config.get('dim',1280)!=spec.embedding_dim:
        raise ValueError('Model dimension must match cached ESM dimension')
    if cache_config.get('read_dtype','float16')!='float16' or cache_config.get('storage_dtype','float16')!='float16':
        raise ValueError('NPY RAM cache must stay float16; update legacy read_dtype settings')
    esm=config.get('esm',{})
    if esm.get('model_name',spec.model_name)!=spec.model_name or esm.get('representation_layer',spec.layer)!=spec.layer:
        raise ValueError('Configuration ESM source/layer disagrees with cache spec')
    budget=args.cache_budget_gib if args.cache_budget_gib is not None else cache_config.get('ram_budget_gib',40.0)
    config['cache']['ram_budget_gib']=budget
    if args.resume is None and any((args.output_dir/name).exists() for name in ('metrics.jsonl','last.pt','summary.json')):
        raise FileExistsError('Existing run: select a fresh output directory or resume')
    loading=load_caches_to_ram(caches,budget_gib=budget)
    datasets={}
    for name in ('train','validation'):
        records,_=load_fasta(paths[name],check_sequences=data.get('check_sequences',True))
        datasets[name]=CachedPeptideDataset(records,caches[name])
    provenance={'data_kind':'synthetic' if spec.source.startswith('mock:') else 'real_cached',
                'esm_source':asdict(spec),
                'train':{'file_sha256':audit['splits']['train']['file_sha256'],
                         'retained':audit['splits']['train']['stats']['retained'],
                         'cache_manifest_sha256':caches['train'].manifest_sha256},
                'validation':{'file_sha256':audit['splits']['validation']['file_sha256'],
                              'retained':audit['splits']['validation']['stats']['retained'],
                              'cache_manifest_sha256':caches['validation'].manifest_sha256}}
    engine=TrainingEngine(config,datasets['train'],datasets['validation'],data_provenance=provenance,
                          init_checkpoint=args.init_checkpoint,resume=args.resume)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'cache_loading.json').write_text(json.dumps(loading,indent=2)+'\n')
    (args.output_dir/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (args.output_dir/'cache_spec.json').write_text(json.dumps(asdict(spec),indent=2)+'\n')
    if settings.lambda_emb==0:
        print('lambda_emb=0: embedding MSE is logged, but ESM embedding alignment is NOT supervised.')
    print(json.dumps(engine.fit(args.output_dir),indent=2))


if __name__=='__main__':
    main()
