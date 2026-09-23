import pytest
import torch
from torch import nn

from templatedf.data import PeptideRecord, encode_labels
from templatedf.esm_features import ESMFeatureExtractor, local_cache_spec


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


def test_npy_ram_cache_smoke(tmp_path):
    import runpy
    from pathlib import Path
    run=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/smoke_ram_cache.py'))['run_smoke']
    report=run(tmp_path,'cpu')
    assert report['roundtrip_exact_after_fp16_cast'] and len(report['batches'])==2


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
