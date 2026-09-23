"""One ESM cache format: uncompressed NPY shards, eagerly resident FP16 RAM."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import AA_ORDER, AA_TO_LABEL, PeptideRecord, file_sha256, validate_sequence

CACHE_VERSION = 2
DEFAULT_SHARD_BYTES = 256 * 1024**2
DEFAULT_RAM_BUDGET_GIB = 40.0
STORAGE_DTYPES = {'float16': torch.float16}


class CacheError(ValueError):
    """Incompatible, incomplete or damaged ESM cache."""


@dataclass(frozen=True)
class CacheSpec:
    source: str
    model_name: str = 'esm2_t33_650M_UR50D'
    layer: int = 33
    embedding_dim: int = 1280
    storage_dtype: str = 'float16'

    def __post_init__(self):
        if not self.source or self.storage_dtype != 'float16':
            raise ValueError('ESM shard caches require a source and float16 storage')
        if (self.model_name, self.layer, self.embedding_dim) != ('esm2_t33_650M_UR50D', 33, 1280):
            raise ValueError('Expected ESM2 650M, layer 33, dimension 1280')


def _array_hash(array):
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def _labels(sequence):
    return np.fromiter((AA_TO_LABEL[aa] for aa in sequence), dtype=np.int8, count=len(sequence))


class FeatureCacheWriter:
    """Buffer at most one shard; publish manifest only on successful close.

    Existing directories are never overwritten. An interrupted directory without
    manifest is incomplete and must not be used for training.
    """
    def __init__(self, root, spec, *, target_shard_bytes=DEFAULT_SHARD_BYTES, source_fasta=None):
        if type(target_shard_bytes) is not int or target_shard_bytes < 2560:
            raise ValueError('target_shard_bytes must hold at least one residue (2560 bytes)')
        self.root, self.spec = Path(root), spec
        self.root.mkdir(parents=True, exist_ok=False)
        self.target_shard_bytes = target_shard_bytes
        self.source_fasta = source_fasta
        self.features, self.records, self.shards = [], [], []
        self.residues = self.total_residues = self.total_records = 0
        self.closed = False

    def write(self, record, embeddings):
        if self.closed:
            raise CacheError('Cache writer is closed')
        validate_sequence(record.sequence)
        if isinstance(embeddings, torch.Tensor):
            feature = embeddings.detach().to(device='cpu', dtype=torch.float16).numpy().copy()
        else:
            feature = np.array(embeddings, dtype=np.float16, order='C', copy=True)
        if feature.shape != (record.length, 1280) or not np.isfinite(feature).all():
            raise CacheError('Invalid or nonfinite FP16 residue embeddings')
        if self.records and (self.residues + record.length)*2560 > self.target_shard_bytes:
            self._flush()
        self.features.append(feature)
        self.records.append(record)
        self.residues += record.length

    def _flush(self):
        if not self.records:
            return
        directory = self.root/f'shard_{len(self.shards):06d}'
        directory.mkdir()
        # Concatenation is limited to this write shard, never the resident cache.
        arrays = {'embeddings': np.concatenate(self.features, axis=0),
                  'offsets': np.array([0, *np.cumsum([r.length for r in self.records])], dtype=np.int64),
                  'labels': np.concatenate([_labels(r.sequence) for r in self.records])}
        descriptions = {}
        for name, array in arrays.items():
            np.save(directory/f'{name}.npy', array, allow_pickle=False)
            descriptions[name] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                  'nbytes': array.nbytes, 'sha256_data': _array_hash(array)}
        with (directory/'metadata.jsonl').open('w') as stream:
            for index, record in enumerate(self.records):
                item = {'id': record.id, 'sequence': record.sequence, 'sequence_hash': record.sequence_hash,
                        'cache_key': record.cache_key, 'length': record.length,
                        'record_index': self.total_records + index}
                stream.write(json.dumps(item, ensure_ascii=False)+'\n')
        self.shards.append({'directory': directory.name, 'records': len(self.records),
                            'residues': self.residues, 'arrays': descriptions,
                            'metadata_bytes': (directory/'metadata.jsonl').stat().st_size,
                            'metadata_sha256': file_sha256(directory/'metadata.jsonl')})
        self.total_records += len(self.records);self.total_residues += self.residues
        self.records.clear();self.features.clear();self.residues = 0

    def close(self):
        if self.closed:
            return
        self._flush()
        if not self.total_records:
            raise CacheError('Cannot publish an empty cache')
        manifest = {'format_version': CACHE_VERSION, 'format': 'npy_shards', 'complete': True,
                    'spec': asdict(self.spec), 'aa_order': AA_ORDER, 'labels_dtype': 'int8',
                    'target_shard_bytes': self.target_shard_bytes, 'records': self.total_records,
                    'residues': self.total_residues, 'source_fasta': self.source_fasta, 'shards': self.shards}
        temporary = self.root/'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.replace(self.root/'manifest.json')
        self.closed = True

    def __enter__(self): return self
    def __exit__(self, kind, value, traceback):
        if kind is None: self.close()


def _check_npy_header(path, shape, dtype):
    with path.open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise CacheError(f'Unsupported NPY version: {version}')
        size = stream.tell() + math.prod(shape)*np.dtype(dtype).itemsize
    if actual_shape != tuple(shape) or actual_dtype != np.dtype(dtype) or fortran or path.stat().st_size != size:
        raise CacheError(f'NPY header/size mismatch: {path}')


class FeatureCache:
    """Read manifest/header first, allocate normal arrays only after joint budget check."""
    def __init__(self, root, spec=None):
        self.root = Path(root).resolve()
        if not (self.root/'manifest.json').is_file():
            raise CacheError(f'NPY v2 manifest missing: {self.root}. Legacy per-record PT caches are not supported; regenerate into a new directory.')
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        if (m.get('format_version') != CACHE_VERSION or m.get('format') != 'npy_shards'
                or m.get('complete') is not True or m.get('aa_order') != AA_ORDER or m.get('labels_dtype') != 'int8'):
            raise CacheError('Expected a complete NPY v2 cache with the fixed AA order')
        self.spec = CacheSpec(**m['spec'])
        if spec is not None and self.spec != spec:
            raise CacheError('Cache ESM source/spec mismatch')
        self.manifest_sha256 = file_sha256(self.root/'manifest.json')
        self.shards, self.locations = [], {}
        self.loaded = False
        self.array_bytes = self.metadata_bytes = 0
        total_records = total_residues = 0
        seen = set()
        for shard in m['shards']:
            directory = self.root/shard['directory']
            if directory.resolve().parent != self.root or directory.name in seen:
                raise CacheError('Invalid/duplicate shard directory')
            seen.add(directory.name)
            n, residues = shard['records'], shard['residues']
            if type(n) is not int or n < 1 or type(residues) is not int or not 10*n <= residues <= 150*n:
                raise CacheError('Invalid shard counts')
            expected = {'embeddings': ([residues, 1280], 'float16'),
                        'offsets': ([n+1], 'int64'), 'labels': ([residues], 'int8')}
            for name, (shape, dtype) in expected.items():
                description = shard['arrays'][name]
                nbytes = math.prod(shape)*np.dtype(dtype).itemsize
                if (description['shape'], description['dtype'], description['nbytes']) != (shape, dtype, nbytes):
                    raise CacheError('Manifest array dimensions/dtype/size mismatch')
                _check_npy_header(directory/f'{name}.npy', shape, dtype)
                self.array_bytes += nbytes
            metadata_bytes = (directory/'metadata.jsonl').stat().st_size
            if metadata_bytes != shard['metadata_bytes']:
                raise CacheError('Metadata size mismatch')
            self.metadata_bytes += metadata_bytes
            total_records += n;total_residues += residues
        if not total_records or (total_records, total_residues) != (m['records'], m['residues']):
            raise CacheError('Cache total count mismatch')
        # Conservative allowance for decoded JSON strings/dicts and key index.
        self.estimated_ram_bytes = self.array_bytes + self.metadata_bytes*6 + total_records*512

    def _load(self):
        if self.loaded:
            return
        for shard_index, info in enumerate(self.manifest['shards']):
            directory = self.root/info['directory']
            arrays = {}
            for name in ('embeddings', 'offsets', 'labels'):
                array = np.load(directory/f'{name}.npy', mmap_mode=None, allow_pickle=False)
                if (array.shape != tuple(info['arrays'][name]['shape']) or str(array.dtype) != info['arrays'][name]['dtype']
                        or _array_hash(array) != info['arrays'][name]['sha256_data']):
                    raise CacheError(f'Array integrity mismatch: {directory/name}')
                arrays[name] = array
            offsets = arrays['offsets']
            if offsets[0] != 0 or offsets[-1] != info['residues'] or not np.all((np.diff(offsets) >= 10) & (np.diff(offsets) <= 150)):
                raise CacheError('Invalid residue offsets')
            for start in range(0, len(arrays['embeddings']), 4096):
                if not np.isfinite(arrays['embeddings'][start:start+4096]).all():
                    raise CacheError('Nonfinite cached embeddings')
            if file_sha256(directory/'metadata.jsonl') != info['metadata_sha256']:
                raise CacheError('Metadata checksum mismatch')
            metadata = []
            first_index = sum(s['records'] for s in self.manifest['shards'][:shard_index])
            with (directory/'metadata.jsonl').open() as stream:
                for local_index, line in enumerate(stream):
                    item = json.loads(line)
                    record = PeptideRecord(item['id'], item['sequence'])
                    validate_sequence(record.sequence)
                    if (local_index >= info['records'] or item['record_index'] != first_index+local_index
                            or item['sequence_hash'] != record.sequence_hash or item['cache_key'] != record.cache_key
                            or item['length'] != record.length or offsets[local_index+1]-offsets[local_index] != record.length
                            or not np.array_equal(arrays['labels'][offsets[local_index]:offsets[local_index+1]], _labels(record.sequence))):
                        raise CacheError('Metadata/offset/label correspondence mismatch')
                    metadata.append(item)
                    self.locations.setdefault(record.cache_key, (shard_index, local_index))
            if len(metadata) != info['records']:
                raise CacheError('Metadata record count mismatch')
            self.shards.append({**arrays, 'metadata': metadata})
        self.loaded = True

    def read(self, record, *, dtype=torch.float16):
        if not self.loaded:
            raise CacheError('Load caches with load_caches_to_ram before sampling')
        if dtype != torch.float16:
            raise ValueError('Resident ESM cache stays FP16; convert only the collated batch on device')
        try:
            shard_index, local = self.locations[record.cache_key]
        except KeyError as exc:
            raise CacheError(f'Missing cached record: {record.id}') from exc
        shard = self.shards[shard_index]
        start, stop = map(int, shard['offsets'][local:local+2])
        return {**shard['metadata'][local], 'spec': asdict(self.spec),
                'embeddings': torch.from_numpy(shard['embeddings'][start:stop]),
                'labels': torch.from_numpy(shard['labels'][start:stop])}


def load_caches_to_ram(caches, *, budget_gib=DEFAULT_RAM_BUDGET_GIB):
    """Budget train+validation together, share identical roots, then sequentially load."""
    if not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError('RAM cache budget must be positive and finite')
    unique = {}
    for cache in caches.values(): unique.setdefault(cache.root, cache)
    estimate = sum(c.estimated_ram_bytes for c in unique.values())
    array_bytes = sum(c.array_bytes for c in unique.values())
    print(f'RAM cache estimate (train + validation): {estimate/1024**3:.6f} GiB '
          f'({estimate:,} bytes; arrays {array_bytes/1024**3:.6f} GiB), '
          f'budget {budget_gib:g} GiB. Includes metadata/index allowance; excludes model, optimizer, batches and OS.', flush=True)
    if estimate > budget_gib*1024**3:
        raise MemoryError(f'RAM cache needs approximately {estimate/1024**3:.6f} GiB ({estimate:,} bytes); '
                          f'configured budget is {budget_gib:g} GiB. Increase the explicit budget or reduce the cache; no automatic fallback.')
    start = time.perf_counter()
    for cache in unique.values(): cache._load()
    for name, cache in list(caches.items()): caches[name] = unique[cache.root]
    report = {'estimated_ram_bytes': estimate, 'array_bytes': array_bytes, 'budget_gib': budget_gib,
              'unique_cache_directories': len(unique), 'load_seconds': time.perf_counter()-start,
              'mmap': False, 'resident_dtype': 'float16'}
    print(f"RAM cache loaded in {report['load_seconds']:.3f} s; all shards resident as ordinary FP16 arrays.", flush=True)
    return report


class CachedPeptideDataset(Dataset):
    def __init__(self, records, cache, *, dtype=torch.float16):
        if not cache.loaded or dtype != torch.float16:
            raise ValueError('Dataset requires a fully loaded FP16 RAM cache')
        missing = [r.id for r in records if r.cache_key not in cache.locations]
        if missing:
            raise CacheError(f'{len(missing)} records missing from cache; first IDs: {missing[:5]}')
        self.records, self.cache, self.dtype = records, cache, dtype

    def __len__(self): return len(self.records)
    def __getitem__(self, index): return self.cache.read(self.records[index], dtype=self.dtype)


def collate_features(items, *, dtype=torch.float16):
    from .training import collate_training_samples
    if dtype != torch.float16 or any(item['spec'] != items[0]['spec'] for item in items):
        raise CacheError('Expected matching FP16 ESM sources')
    batch = collate_training_samples(items)
    return {**batch, 'ids': [item['id'] for item in items], 'sequences': [item['sequence'] for item in items]}
