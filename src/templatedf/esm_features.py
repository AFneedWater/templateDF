"""Frozen local ESM2 extraction and bounded cache CLI (no network downloads)."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from .data import PeptideRecord, file_sha256, load_fasta, validate_sequence
from .feature_cache import CacheSpec, FeatureCache, STORAGE_DTYPES

MODEL_NAME = "esm2_t33_650M_UR50D"
DEFAULT_CHECKPOINT = Path("/home/gh/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt")


def local_cache_spec(checkpoint: str | Path, regression_checkpoint: str | Path,
                     *, storage_dtype: str = "float16") -> CacheSpec:
    checkpoint, regression_checkpoint = Path(checkpoint), Path(regression_checkpoint)
    for path in (checkpoint, regression_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Local ESM weight file missing; downloads disabled: {path}")
    source = (f"fair-esm:{importlib.metadata.version('fair-esm')};"
              f"checkpoint-sha256:{file_sha256(checkpoint)};"
              f"regression-sha256:{file_sha256(regression_checkpoint)};compute:float32")
    return CacheSpec(source=source, storage_dtype=storage_dtype)


class ESMFeatureExtractor:
    """Injected model enables offline mock tests; parameters are always frozen."""

    def __init__(self, model, alphabet, *, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        if not alphabet.prepend_bos or not alphabet.append_eos:
            raise ValueError("ESM2 BOS/EOS alphabet required")

    @classmethod
    def from_local(cls, checkpoint: str | Path, regression_checkpoint: str | Path,
                   *, device: str | torch.device = "cpu"):
        # Check both paths before importing/constructing ESM; never call a hub loader.
        checkpoint, regression_checkpoint = Path(checkpoint), Path(regression_checkpoint)
        for path in (checkpoint, regression_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"Local ESM weight file missing; downloads disabled: {path}")
        from esm.pretrained import load_model_and_alphabet_core

        # Official ESM2 checkpoint contains argparse.Namespace. Keep weights_only
        # loading, allow just that known configuration type, and use local files.
        with torch.serialization.safe_globals([argparse.Namespace]):
            model_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
            regression_data = torch.load(regression_checkpoint, map_location="cpu", weights_only=True)
        cfg = model_data["cfg"]["model"]
        if (cfg.encoder_layers, cfg.encoder_embed_dim, cfg.encoder_attention_heads) != (33, 1280, 20):
            raise ValueError("Checkpoint architecture is not ESM2 650M (33/1280/20)")
        model, alphabet = load_model_and_alphabet_core(MODEL_NAME, model_data, regression_data)
        return cls(model, alphabet, device=device)

    @torch.no_grad()
    def extract(self, records: list[PeptideRecord]) -> list[torch.Tensor]:
        if not records:
            raise ValueError("Cannot extract an empty batch")
        for record in records:
            validate_sequence(record.sequence)
        # Reassert the invariant even if a caller changed model mode/grad flags.
        self.model.requires_grad_(False)
        self.model.eval()
        _, _, tokens = self.batch_converter([(r.id, r.sequence) for r in records])
        for index, record in enumerate(records):
            if tokens[index, 0].item() != self.alphabet.cls_idx:
                raise ValueError("Missing BOS token")
            if tokens[index, record.length + 1].item() != self.alphabet.eos_idx:
                raise ValueError("EOS does not follow the true residue length")
            if not (tokens[index, record.length + 2:] == self.alphabet.padding_idx).all():
                raise ValueError("ESM batch must be right padded")
        output = self.model(tokens.to(self.device), repr_layers=[33], return_contacts=False)
        hidden = output["representations"][33]
        if hidden.shape != (*tokens.shape, 1280):
            raise ValueError(f"Unexpected ESM representation shape: {tuple(hidden.shape)}")
        # Each row has its own EOS location; a global [:, 1:-1] is incorrect.
        features = [hidden[index, 1:1 + record.length].detach().to("cpu").clone()
                    for index, record in enumerate(records)]
        if any(not torch.isfinite(feature).all() for feature in features):
            raise ValueError("ESM returned nonfinite residue embeddings")
        return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--regression-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--limit", type=int, help="Maximum retained records to process")
    budget.add_argument("--all", action="store_true", help="Explicitly request full extraction")
    parser.add_argument("--storage-dtype", choices=STORAGE_DTYPES, default="float16")
    parser.add_argument("--no-sequence-check", action="store_true",
                        help="Skip FASTA prefiltering; extraction still rejects invalid sequences")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    start = time.monotonic()
    regression = args.regression_checkpoint or args.checkpoint.with_name(args.checkpoint.stem + "-contact-regression.pt")
    records, stats = load_fasta(args.fasta, check_sequences=not args.no_sequence_check)
    selected = records if args.all else records[:args.limit]
    if not selected:
        parser.error("No retained records available for extraction")
    spec = local_cache_spec(args.checkpoint, regression, storage_dtype=args.storage_dtype)
    cache = FeatureCache(args.cache_dir, spec)
    pending = {}
    reused = 0
    for record in selected:
        validate_sequence(record.sequence)
        if cache.path_for(record).exists():
            cache.read(record, dtype=torch.float32)  # Never silently reuse a bad/stale entry.
            reused += 1
        else:
            pending.setdefault(record.cache_key, record)
    extractor = None
    if pending:
        extractor = ESMFeatureExtractor.from_local(args.checkpoint, regression, device=args.device)
        missing = list(pending.values())
        for offset in range(0, len(missing), args.batch_size):
            chunk = missing[offset:offset + args.batch_size]
            for record, feature in zip(chunk, extractor.extract(chunk)):
                cache.write(record, feature)
    report = {
        "fasta": str(args.fasta.resolve()), "fasta_sha256": file_sha256(args.fasta),
        "filter_stats": asdict(stats), "selected_records": len(selected),
        "selected_unique_cache_keys": len({r.cache_key for r in selected}),
        "new_cache_entries": len(pending), "reused_records": reused,
        "model_loaded": extractor is not None, "device": args.device,
        "spec": asdict(spec), "cache_dir": str(args.cache_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "regression_checkpoint": str(regression.resolve()),
        "elapsed_seconds": time.monotonic() - start,
        "records": [{"id": r.id, "sequence_hash": r.sequence_hash, "length": r.length,
                     "cache_path": str(cache.path_for(r).resolve())} for r in selected],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("selected_records", "new_cache_entries", "reused_records", "model_loaded", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
