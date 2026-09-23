"""Versioned, integrity-checked per-record ESM cache; tensors load on CPU."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import Dataset

from .data import AA_ORDER, PeptideRecord, collate_peptides, encode_labels

CACHE_VERSION = 1
STORAGE_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class CacheError(ValueError):
    """Cache metadata, source or contents failed validation."""


@dataclass(frozen=True)
class CacheSpec:
    source: str  # Weight content hashes + implementation, or an explicit mock identity.
    model_name: str = "esm2_t33_650M_UR50D"
    layer: int = 33
    embedding_dim: int = 1280
    storage_dtype: str = "float16"

    def __post_init__(self):
        if not self.source or self.storage_dtype not in STORAGE_DTYPES:
            raise ValueError("A model source and supported storage dtype are required")
        if (self.model_name, self.layer, self.embedding_dim) != ("esm2_t33_650M_UR50D", 33, 1280):
            raise ValueError("This stage uses ESM2 650M, layer 33, dimension 1280")


def _checksum(payload: dict) -> str:
    metadata = {key: value for key, value in payload.items() if key not in {"labels", "embeddings"}}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode())
    for key in ("labels", "embeddings"):
        tensor = payload[key].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FeatureCache:
    def __init__(self, root: str | Path, spec: CacheSpec):
        self.root = Path(root)
        self.spec = spec

    def path_for(self, record: PeptideRecord) -> Path:
        return self.root / f"{record.cache_key}.pt"

    def _validate(self, payload: dict, record: PeptideRecord):
        expected_fields = {"format_version", "id", "sequence", "sequence_hash", "cache_key",
                           "length", "aa_order", "spec", "labels", "embeddings"}
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise CacheError("Invalid cache schema")
        expected = {"format_version": CACHE_VERSION, "id": record.id,
                    "sequence": record.sequence, "sequence_hash": record.sequence_hash,
                    "cache_key": record.cache_key, "length": record.length,
                    "aa_order": AA_ORDER, "spec": asdict(self.spec)}
        for key, value in expected.items():
            if payload[key] != value:
                raise CacheError(f"Cache metadata/source mismatch: {key}")
        labels, embeddings = payload["labels"], payload["embeddings"]
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
            raise CacheError("Invalid label tensor")
        if not torch.equal(labels, encode_labels(record.sequence)):
            raise CacheError("Labels disagree with original sequence")
        if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (record.length, self.spec.embedding_dim):
            raise CacheError("Invalid embedding shape")
        if embeddings.dtype != STORAGE_DTYPES[self.spec.storage_dtype]:
            raise CacheError("Storage dtype does not match metadata")
        if not torch.isfinite(embeddings).all():
            raise CacheError("Nonfinite cache embeddings")

    def write(self, record: PeptideRecord, embeddings: torch.Tensor) -> Path:
        payload = {
            "format_version": CACHE_VERSION, "id": record.id, "sequence": record.sequence,
            "sequence_hash": record.sequence_hash, "cache_key": record.cache_key,
            "length": record.length, "aa_order": AA_ORDER, "spec": asdict(self.spec),
            "labels": encode_labels(record.sequence),
            "embeddings": embeddings.detach().to(device="cpu", dtype=STORAGE_DTYPES[self.spec.storage_dtype]).contiguous(),
        }
        self._validate(payload, record)
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(record)
        envelope = {"payload": payload, "sha256": _checksum(payload)}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(envelope, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic publication without overwriting a concurrent/existing entry.
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target

    def read(self, record: PeptideRecord, *, dtype: torch.dtype) -> dict:
        if dtype not in STORAGE_DTYPES.values():
            raise ValueError("Select an explicit floating-point read dtype")
        path = self.path_for(record)
        try:
            envelope = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(envelope, dict) or set(envelope) != {"payload", "sha256"}:
                raise CacheError("Invalid cache envelope")
            payload = envelope["payload"]
            self._validate(payload, record)
            if _checksum(payload) != envelope["sha256"]:
                raise CacheError("Cache checksum mismatch")
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise CacheError(f"Invalid cache {path}: {exc}") from exc
        return {**payload, "embeddings": payload["embeddings"].to(dtype=dtype)}


class CachedPeptideDataset(Dataset):
    """Explicit records plus source and read dtype; construction never loads ESM."""

    def __init__(self, records: list[PeptideRecord], cache: FeatureCache, *, dtype: torch.dtype):
        if dtype not in STORAGE_DTYPES.values():
            raise ValueError("Select an explicit floating-point read dtype")
        self.records, self.cache, self.dtype = records, cache, dtype

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.cache.read(self.records[index], dtype=self.dtype)


def collate_features(items: list[dict], *, dtype: torch.dtype) -> dict:
    if not items or dtype not in STORAGE_DTYPES.values():
        raise ValueError("A nonempty batch and explicit floating-point dtype are required")
    records = [PeptideRecord(item["id"], item["sequence"]) for item in items]
    batch = collate_peptides(records)
    reference = items[0]["embeddings"]
    if any(item["spec"] != items[0]["spec"] for item in items):
        raise CacheError("Cannot collate different model sources/storage specifications")
    embeddings = reference.new_zeros((*batch["labels"].shape, 1280), dtype=dtype)
    for index, (record, item) in enumerate(zip(records, items)):
        feature = item["embeddings"]
        if feature.shape != (record.length, 1280) or feature.device != reference.device:
            raise CacheError("Inconsistent cached embedding shape/device")
        embeddings[index, :record.length] = feature.to(dtype=dtype)
    batch.update({"embeddings": embeddings,
                  "labels": batch["labels"].to(reference.device),
                  "lengths": batch["lengths"].to(reference.device),
                  "input_mask": batch["input_mask"].to(reference.device)})
    return batch
