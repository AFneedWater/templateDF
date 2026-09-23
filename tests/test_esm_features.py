from dataclasses import replace

import pytest
import torch
from torch import nn

from templatedf.data import PeptideRecord, encode_labels
from templatedf.esm_features import ESMFeatureExtractor, local_cache_spec
from templatedf.feature_cache import CacheError, CacheSpec, CachedPeptideDataset, FeatureCache, collate_features


class PositionAlphabet:
    prepend_bos = append_eos = True
    cls_idx, eos_idx, padding_idx = 1000, 2000, 0

    def get_batch_converter(self):
        def convert(records):
            tokens = torch.zeros(len(records), max(len(seq) for _, seq in records) + 2, dtype=torch.long)
            for index, (_, sequence) in enumerate(records):
                length = len(sequence)
                tokens[index, 0] = self.cls_idx
                tokens[index, 1:length + 1] = torch.arange(1, length + 1)
                tokens[index, length + 1] = self.eos_idx
            return [r[0] for r in records], [r[1] for r in records], tokens
        return convert


class PositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, tokens, repr_layers, return_contacts):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert repr_layers == [33] and return_contacts is False
        self.calls += 1
        hidden = tokens.float().unsqueeze(-1).expand(-1, -1, 1280) + self.weight * 0
        return {"representations": {33: hidden}}


def test_mock_positions_single_batch_and_frozen_eval():
    model = PositionModel()
    extractor = ESMFeatureExtractor(model, PositionAlphabet())
    records = [PeptideRecord(str(n), "A" * n) for n in (10, 49, 50, 150)]
    model.train()
    model.requires_grad_(True)
    together = extractor.extract(records)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    for record, feature in zip(records, together):
        assert feature.shape == (record.length, 1280)
        assert not feature.requires_grad
        assert torch.equal(feature[:, 0], torch.arange(1, record.length + 1).float())
        assert torch.equal(feature, extractor.extract([record])[0])
        assert not (feature == 1000).any() and not (feature == 2000).any()
    calls = model.calls
    with pytest.raises(ValueError):
        extractor.extract([PeptideRecord("bad", "X" * 10)])
    assert model.calls == calls


def test_missing_local_weights_fail_before_any_hub_access(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Network loading must never be called")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fail)
    with pytest.raises(FileNotFoundError, match="downloads disabled"):
        ESMFeatureExtractor.from_local(tmp_path / "missing.pt", tmp_path / "regression.pt")
    with pytest.raises(FileNotFoundError):
        local_cache_spec(tmp_path / "missing.pt", tmp_path / "regression.pt")


@pytest.fixture
def cache_record(tmp_path):
    spec = CacheSpec(source="mock:position-v1")
    cache = FeatureCache(tmp_path / "cache", spec)
    record = PeptideRecord("same/id", "ACDEFGHIKL")
    feature = torch.arange(10).float()[:, None].expand(-1, 1280)
    cache.write(record, feature)
    return cache, record


@pytest.mark.parametrize("storage_dtype", ["float16", "bfloat16", "float32"])
def test_cache_roundtrip_dtype_and_collation(tmp_path, storage_dtype):
    cache = FeatureCache(tmp_path, CacheSpec(source="mock:position-v1", storage_dtype=storage_dtype))
    records = [PeptideRecord("same/id", "ACDEFGHIKL"), PeptideRecord("same/id", "C" * 150)]
    for record in records:
        feature = torch.arange(record.length).float()[:, None].expand(-1, 1280)
        cache.write(record, feature)
        item = cache.read(record, dtype=torch.float32)
        assert item["id"] == record.id and item["sequence_hash"] == record.sequence_hash
        assert item["length"] == record.length
        assert torch.equal(item["labels"], encode_labels(record.sequence))
        assert torch.equal(item["embeddings"], feature)
        assert item["embeddings"].dtype == torch.float32
    assert cache.path_for(records[0]) != cache.path_for(records[1])
    dataset = CachedPeptideDataset(records, cache, dtype=torch.float32)
    batch = collate_features([dataset[0], dataset[1]], dtype=torch.float32)
    assert batch["embeddings"].shape == (2, 150, 1280)
    assert torch.count_nonzero(batch["embeddings"][0, 10:]) == 0
    assert (batch["labels"][0, 10:] == -100).all()
    assert batch["input_mask"].sum(1).tolist() == [10, 150]
    with pytest.raises(FileExistsError):
        cache.write(records[0], torch.zeros(10, 1280))
    # Failed duplicate publication left original intact and no partial files.
    assert cache.read(records[0], dtype=torch.float32)["embeddings"][9, 0] == 9
    assert not list(tmp_path.glob("*.tmp"))


def test_wrong_source_and_storage_dtype_rejected(cache_record):
    cache, record = cache_record
    for spec in (replace(cache.spec, source="mock:different"), replace(cache.spec, storage_dtype="float32")):
        with pytest.raises(CacheError, match="metadata/source mismatch"):
            FeatureCache(cache.root, spec).read(record, dtype=torch.float32)


@pytest.mark.parametrize("damage", ["id", "hash", "length", "labels", "shape", "dtype", "nan", "finite_value", "layer", "version", "truncated"])
def test_corrupt_cache_rejected(cache_record, damage):
    cache, record = cache_record
    path = cache.path_for(record)
    if damage == "truncated":
        path.write_bytes(b"broken cache")
    else:
        envelope = torch.load(path, weights_only=True)
        payload = envelope["payload"]
        if damage == "id": payload["id"] = "other"
        elif damage == "hash": payload["sequence_hash"] = "bad"
        elif damage == "length": payload["length"] += 2
        elif damage == "labels": payload["labels"][0] = 19
        elif damage == "shape": payload["embeddings"] = payload["embeddings"][:-1]
        elif damage == "dtype": payload["embeddings"] = payload["embeddings"].float()
        elif damage == "nan": payload["embeddings"][0, 0] = float("nan")
        elif damage == "finite_value": payload["embeddings"][0, 0] = 123
        elif damage == "layer": payload["spec"]["layer"] = 32
        elif damage == "version": payload["format_version"] = 99
        torch.save(envelope, path)
    with pytest.raises(CacheError, match="Invalid cache"):
        cache.read(record, dtype=torch.float32)


@pytest.mark.parametrize("budget", [[], ["--limit", "0"], ["--limit", "-1"],
                                    ["--limit", "3", "--batch-size", "0"],
                                    ["--all", "--limit", "3"]])
def test_cli_requires_explicit_positive_budget_before_loading(tmp_path, monkeypatch, budget):
    import sys
    from templatedf.esm_features import main
    monkeypatch.setattr(sys, "argv", ["esm_features", "--fasta", str(tmp_path / "absent.fa"),
                        "--cache-dir", str(tmp_path / "cache"),
                        "--report", str(tmp_path / "report.json"), *budget])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert not (tmp_path / "cache").exists()
