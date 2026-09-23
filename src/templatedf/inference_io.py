"""Streaming FASTA candidates and raw latent shards with verified provenance."""
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path

import torch

from .checkpoint import atomic_save
from .data import AA_ORDER, PeptideRecord, file_sha256, read_fasta, validate_sequence
from .inference import positive_integer


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def selected_records(path, limit):
    if limit is not None:
        positive_integer(limit, 'limit')
    for record in islice(read_fasta(path), limit):
        validate_sequence(record.sequence)  # Never filter invalid inference templates.
        yield record


def inspect_fasta(path, limit):
    count = sum(1 for _ in selected_records(path, limit))
    if not count:
        raise ValueError('No templates in selected FASTA range')
    return {'path': str(Path(path).resolve()), 'sha256': file_sha256(path), 'selected_records': count, 'limit': limit}


def write_candidates(runtime, records, output_dir, *, expected_records, input_info=None, **generation):
    positive_integer(expected_records, 'expected_records')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    seen = defaultdict(set)
    counts = defaultdict(int)
    all_sequences = set()
    count = 0
    # A manifest is published only after both files and record counts are valid.
    with (output/'candidates.fasta').open('x') as fasta, (output/'metadata.jsonl').open('x') as metadata:
        for row in runtime.iter_generate(records, **generation):
            validate_sequence(row['sequence'])
            if len(row['sequence']) != row['target_length']:
                raise ValueError('Candidate length differs from metadata')
            fasta.write(f">{row['candidate_id']}\n{row['sequence']}\n")
            metadata.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            key = row['template_index']
            seen[key].add(row['sequence']);counts[key] += 1
            all_sequences.add(row['sequence']);count += 1
    samples = generation.get('num_samples', 1)
    if count != expected_records * samples or len(counts) != expected_records or any(n != samples for n in counts.values()):
        raise ValueError('Candidate/template count mismatch; output is incomplete')
    stats = {str(k): {'count': counts[k], 'unique': len(seen[k]), 'duplicate_rate': 1-len(seen[k])/counts[k]} for k in counts}
    if input_info and file_sha256(input_info['path']) != input_info['sha256']:
        raise ValueError('Input FASTA changed during inference')
    manifest = {'format_version': 1, 'kind': 'template_candidates', 'complete': True,
                'template_count': expected_records, 'candidate_count': count, 'input': input_info,
                'generation': generation, 'provenance': runtime.provenance,
                'within_template_duplicate_rate': 1-sum(len(x) for x in seen.values())/count,
                'global_duplicate_rate': 1-len(all_sequences)/count, 'per_template': stats,
                'files': {name: file_sha256(output/name) for name in ('candidates.fasta', 'metadata.jsonl')}}
    json_write(output/'manifest.json', manifest)
    return manifest


def export_raw_latents(runtime, records, output_dir, *, expected_records, shard_size=256,
                       batch_size=4, input_info=None, condition=None):
    positive_integer(expected_records, 'expected_records')
    positive_integer(shard_size, 'shard_size');positive_integer(batch_size, 'batch_size')
    if condition is not None:
        raise NotImplementedError('Use condition=None')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    buffer, shards = [], []
    count = 0
    def flush():
        if not buffer:
            return
        name = f'shard_{len(shards):06d}.pt'
        atomic_save({'format_version': 1, 'kind': 'raw_template_latents', 'identity': runtime.identity,
                     'provenance': runtime.provenance, 'normalization': None, 'records': list(buffer)}, output/name)
        shards.append({'file': name, 'count': len(buffer), 'first_record_index': buffer[0]['record_index'],
                       'sha256': file_sha256(output/name)})
        buffer.clear()
    for batch, latent in runtime.iter_encoded(records, batch_size):
        for index, record in enumerate(batch):
            # Clone each row so serialization never keeps an entire batch storage.
            raw = latent[index].detach().cpu().clone()
            buffer.append({'record_index': count, 'id': record.id, 'sequence': record.sequence,
                           'sequence_hash': record.sequence_hash, 'sequence_length': record.length,
                           'latent': raw, 'latent_dtype': str(raw.dtype)})
            count += 1
            if len(buffer) == shard_size:
                flush()
    flush()
    if count != expected_records or sum(x['count'] for x in shards) != count:
        raise ValueError('Latent export record count mismatch; output is incomplete')
    if input_info and file_sha256(input_info['path']) != input_info['sha256']:
        raise ValueError('Input FASTA changed during export')
    manifest = {'format_version': 1, 'kind': 'raw_template_latents', 'complete': True,
                'record_count': count, 'shard_size': shard_size, 'batch_size': batch_size,
                'latent_shape': [runtime.model.num_latents, runtime.model.dim],
                'normalization': None, 'condition': None, 'input': input_info,
                'identity': runtime.identity, 'provenance': runtime.provenance, 'shards': shards}
    json_write(output/'manifest.json', manifest)
    return manifest


def iter_latent_records(directory, *, expected_identity=None):
    """Verify a complete export and yield one raw CPU [N,D] record at a time.

    Exhaust this iterator to verify the final count. Only one shard is loaded.
    Pass runtime.identity to prevent decoding with a different encoder/checkpoint.
    """
    root = Path(directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('format_version') != 1 or manifest.get('kind') != 'raw_template_latents'
            or manifest.get('complete') is not True or manifest.get('normalization') is not None
            or manifest['identity']['aa_order'] != AA_ORDER):
        raise ValueError('Invalid raw latent manifest')
    if expected_identity is not None and manifest['identity'] != expected_identity:
        raise ValueError('Latent checkpoint/model/ESM/precision identity mismatch')
    positive_integer(manifest['record_count'], 'record_count')
    positive_integer(manifest['shard_size'], 'shard_size')
    cfg = manifest['identity']['model_config']
    if manifest['latent_shape'] != [cfg.get('num_latents', 50), cfg.get('dim', 1280)]:
        raise ValueError('Latent dimensions disagree with model identity')
    count = 0
    for shard in manifest['shards']:
        positive_integer(shard['count'], 'shard count')
        if shard['count'] > manifest['shard_size']:
            raise ValueError('Shard exceeds declared size')
        name = shard['file']
        path = root/name
        if Path(name).name != name or path.resolve().parent != root:
            raise ValueError('Invalid shard path')
        if file_sha256(path) != shard['sha256']:
            raise ValueError('Latent shard checksum mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload.get('format_version') != 1 or payload.get('kind') != manifest['kind']
                or payload.get('identity') != manifest['identity'] or payload.get('provenance') != manifest['provenance']
                or payload.get('normalization') is not None or len(payload['records']) != shard['count']
                or shard['first_record_index'] != count):
            raise ValueError('Latent shard metadata/count mismatch')
        for record in payload['records']:
            peptide = PeptideRecord(record['id'], record['sequence'])
            validate_sequence(peptide.sequence)
            latent = record['latent']
            if (record['record_index'] != count or record['sequence_hash'] != peptide.sequence_hash
                    or record['sequence_length'] != peptide.length or not isinstance(latent, torch.Tensor)
                    or list(latent.shape) != manifest['latent_shape'] or not latent.is_floating_point()
                    or str(latent.dtype) != record['latent_dtype'] or not torch.isfinite(latent).all()):
                raise ValueError('Invalid raw latent record')
            count += 1
            yield record
    if count != manifest['record_count']:
        raise ValueError('Latent manifest total count mismatch')
