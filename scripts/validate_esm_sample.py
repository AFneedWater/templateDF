"""Validate a small real ESM cache against batched and individual extraction."""

import argparse
from dataclasses import asdict
import json
import time

import torch

from templatedf.data import decode_labels, load_fasta
from templatedf.esm_features import ESMFeatureExtractor, local_cache_spec
from templatedf.feature_cache import FeatureCache, collate_features


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--regression-checkpoint", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    records, stats = load_fasta(args.fasta)
    if not 1 <= len(records) <= 8:
        parser.error("Validation is bounded to 1..8 sample records")
    spec = local_cache_spec(args.checkpoint, args.regression_checkpoint)
    cache = FeatureCache(args.cache_dir, spec)
    extractor = ESMFeatureExtractor.from_local(args.checkpoint, args.regression_checkpoint, device=args.device)
    batch = extractor.extract(records)
    samples = []
    loaded = []
    for record, batched in zip(records, batch):
        single = extractor.extract([record])[0]
        torch.testing.assert_close(single, batched, rtol=1e-3, atol=1e-4)
        item = cache.read(record, dtype=torch.float32)
        torch.testing.assert_close(item["embeddings"], batched, rtol=1e-3, atol=1e-3)
        roundtrip = decode_labels(item["labels"])
        assert roundtrip == record.sequence
        loaded.append(item)
        samples.append({
            "id": record.id, "sequence": record.sequence, "length": record.length,
            "labels": item["labels"].tolist(), "decoded_sequence": roundtrip,
            "sequence_hash": record.sequence_hash, "embedding_shape": list(batched.shape),
            "storage_dtype": spec.storage_dtype, "read_dtype": str(item["embeddings"].dtype),
            "single_vs_batch_max_abs_error": (single - batched).abs().max().item(),
            "cache_vs_batch_max_abs_error": (item["embeddings"] - batched).abs().max().item(),
            "cache_path": str(cache.path_for(record).resolve()),
        })
    collated = collate_features(loaded, dtype=torch.float32)
    assert not collated["embeddings"][~collated["input_mask"]].any()
    assert (collated["labels"][~collated["input_mask"]] == -100).all()
    report = {
        "real_esm": True, "device": args.device, "spec": asdict(spec),
        "filter_stats": asdict(stats), "samples": samples,
        "model_eval": not extractor.model.training,
        "all_parameters_frozen": all(not p.requires_grad for p in extractor.model.parameters()),
        "features_require_grad": any(t.requires_grad for t in batch),
        "batch_embedding_shape": list(collated["embeddings"].shape),
        "batch_mask_counts": collated["input_mask"].sum(1).tolist(),
        "batch_labels_shape": list(collated["labels"].shape),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
