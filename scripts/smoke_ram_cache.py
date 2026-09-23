"""One bounded mock smoke check for NPY roundtrip and two RAM-only batches."""
from pathlib import Path
import argparse
import builtins
import json
from unittest.mock import patch

import numpy as np
import torch

from templatedf.data import PeptideRecord, encode_labels
from templatedf.feature_cache import CacheSpec, FeatureCacheWriter, FeatureCache, CachedPeptideDataset, load_caches_to_ram
from templatedf.training import StatefulBatchSampler, make_data_loader, batch_to_device


def run_smoke(root, device='cpu'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    records=[PeptideRecord(f'length_{n}',('ACDEFGHIKLMNPQRSTVWY'*8)[:n]) for n in (10,50,80,150)]
    validation=[PeptideRecord('validation_50','Y'*50)]
    originals=[torch.arange(r.length*1280,dtype=torch.float32).reshape(r.length,1280)/16384 for r in records]
    spec=CacheSpec(source='mock:ram-smoke-position-values-v1')
    # Tiny target deliberately exercises shard boundaries, without a large test.
    with FeatureCacheWriter(root/'train',spec,target_shard_bytes=150*1280*2) as writer:
        for record,feature in zip(records,originals):writer.write(record,feature)
    with FeatureCacheWriter(root/'validation',spec,target_shard_bytes=150*1280*2) as writer:
        writer.write(validation[0],torch.ones(50,1280))
    caches={name:FeatureCache(root/name,spec) for name in ('train','validation')}
    with patch('numpy.load',side_effect=AssertionError('Budget failure must precede array allocation')):
        try:load_caches_to_ram(caches,budget_gib=0.000001)
        except MemoryError as exc:budget_error=str(exc)
        else:raise AssertionError('Insufficient budget accepted')
    loading=load_caches_to_ram(caches,budget_gib=40)
    train=CachedPeptideDataset(records,caches['train'])
    val=CachedPeptideDataset(validation,caches['validation'])
    assert len(train)==4 and len(val)==1
    assert not ({r.sequence for r in train.records}&{r.sequence for r in val.records})
    assert [s['offsets'].tolist() for s in caches['train'].shards]==[[0,10,60,140],[0,150]]
    for index,(record,original) in enumerate(zip(records,originals)):
        item=train[index]
        assert item['id']==record.id and item['length']==record.length and item['sequence_hash']==record.sequence_hash
        assert torch.equal(item['labels'].long(),encode_labels(record.sequence))
        assert item['embeddings'].dtype==torch.float16 and torch.equal(item['embeddings'],original.half())
        assert torch.isfinite(item['embeddings']).all() and not item['embeddings'].is_pinned()
    for cache in caches.values():
        for shard in cache.shards:
            assert type(shard['embeddings']) is np.ndarray and shard['embeddings'].dtype==np.float16
            assert shard['embeddings'].flags.owndata and not isinstance(shard['embeddings'],np.memmap)
    device=torch.device(device)
    if device.type=='cuda':
        torch.empty(1,device=device)  # Initialize CUDA before disabling Python file reads.
    sampler=StatefulBatchSampler(len(train),2,42)
    loader=make_data_loader(train,device=device,batch_sampler=sampler)
    assert loader.num_workers==0 and not loader.persistent_workers and loader.prefetch_factor is None
    summaries=[]
    def no_disk(*a,**k):raise AssertionError('Cache sampling attempted a file read')
    # All cache files are inaccessible to Python readers during both batches.
    with patch.object(builtins,'open',no_disk),patch.object(Path,'open',no_disk),patch('numpy.load',no_disk),patch('torch.load',no_disk):
        for host in loader:
            indices=list(sampler.last_indices)
            lengths=[records[i].length for i in indices]
            assert host['embeddings'].shape==(2,max(lengths),1280)
            assert host['input_mask'].sum(1).tolist()==lengths
            assert host['embeddings'].dtype==torch.float16 and host['labels'].dtype==torch.int64
            assert not host['embeddings'][~host['input_mask']].any()
            assert (host['labels'][~host['input_mask']]==-100).all()
            assert host['embeddings'].is_pinned()==(device.type=='cuda')
            for j,i in enumerate(indices):assert torch.equal(host['embeddings'][j,:records[i].length],originals[i].half())
            batch=batch_to_device(host,device)
            assert batch['embeddings'].dtype==torch.float32 and batch['embeddings'].device==device
            summaries.append({'indices':indices,'lengths':lengths,'shape':list(host['embeddings'].shape),
                              'host_dtype':str(host['embeddings'].dtype),'device_input_dtype':str(batch['embeddings'].dtype),
                              'batch_pinned':host['embeddings'].is_pinned()})
    assert len(summaries)==2
    assert any(len({caches['train'].locations[records[i].cache_key][0] for i in row['indices']})==2 for row in summaries)
    next_epoch=[i for batch in sampler for i in batch]
    first_epoch=[i for row in summaries for i in row['indices']]
    assert sorted(next_epoch)==sorted(first_epoch)==list(range(4)) and next_epoch!=first_epoch
    report={'data_kind':'mock','real_esm_extraction':False,'optimizer_steps':0,'device':str(device),
            'roundtrip_exact_after_fp16_cast':True,'offsets':[[0,10,60,140],[0,150]],
            'train_records':4,'validation_records':1,'ordinary_resident_fp16_arrays':True,
            'budget_rejection_before_allocation':budget_error,'loading':loading,'batches':summaries,
            'next_epoch_order':next_epoch,'python_file_reads_blocked_during_batches':True,
            'whole_cache_pinned':False,'whole_cache_concatenation':False,
            'not_measured':['real full-cache load time','peak RSS for real training','OS-level disk I/O during training','GPU training/benchmark']}
    (root/'smoke_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir,args.device),indent=2))
