from dataclasses import asdict

import pytest
import torch

from templatedf.data import (
    AA_ORDER, PeptideDataset, PeptideRecord, audit_splits, collate_peptides,
    decode_labels, encode_labels, load_fasta, read_fasta, validate_sequence,
)


def test_boundaries_filter_reasons_and_duplicates(tmp_path):
    entries = [(f"length{n}", "A" * n) for n in (9, 10, 49, 50, 150, 151)]
    entries += [("invalid", "ACDEFGHIKX"), ("internal_space", "ACDEF GHIKL"),
                ("length10", "C" * 10), ("duplicate_seq", "A" * 10),
                ("two_reasons", "X" * 9)]
    path = tmp_path / "data.fasta"
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in entries))
    records, stats = load_fasta(path)
    assert [r.length for r in records] == [10, 49, 50, 150, 10, 10]
    assert asdict(stats) == {
        "original": 11, "retained": 6, "excluded": 5, "too_short": 2,
        "too_long": 1, "nonstandard": 3, "duplicate_sequences": 1,
        "duplicate_ids": 1, "retained_duplicate_sequences": 1, "checks_enabled": True,
    }
    assert records[0].id == records[4].id
    assert records[0].cache_key != records[4].cache_key
    assert records[0].cache_key != records[5].cache_key
    assert records[0].sequence_hash == records[5].sequence_hash
    dataset = PeptideDataset(path)
    assert len(dataset) == 6 and dataset[0] == records[0]
    unchecked, unchecked_stats = load_fasta(path, check_sequences=False)
    assert len(unchecked) == 11
    assert unchecked_stats.nonstandard is None and unchecked_stats.too_short is None
    assert unchecked[7].sequence == "ACDEF GHIKL"
    with pytest.raises(ValueError, match="Nonstandard"):
        collate_peptides([unchecked[7]])


def test_wrapping_preserves_characters_and_description(tmp_path):
    path = tmp_path / "wrapped.fa"
    path.write_bytes(b">id description\r\nACDEF\r\nGHIKL\r\n\r\n>x\nACDEF GHIKL\n")
    records = list(read_fasta(path))
    assert records[0] == PeptideRecord("id", "ACDEFGHIKL", "id description")
    assert records[1].sequence == "ACDEF GHIKL"


@pytest.mark.parametrize("text", ["ACDEFGHIKL\n", ">\nACDEFGHIKL\n"])
def test_malformed_fasta_is_explicit_error(tmp_path, text):
    path = tmp_path / "bad.fa"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(read_fasta(path))


@pytest.mark.parametrize("sequence", ["A" * 9, "A" * 151, "ACDEFGHIKX", "acdefghikl", "ACDEF GHIKL"])
def test_inference_rejects_invalid_without_normalization(sequence):
    with pytest.raises(ValueError):
        validate_sequence(sequence)
    with pytest.raises(ValueError):
        encode_labels(sequence)


def test_roundtrip_padding_mask_and_real_lengths():
    records = [PeptideRecord("short", AA_ORDER[:10]), PeptideRecord("long", AA_ORDER)]
    batch = collate_peptides(records)
    assert torch.equal(encode_labels(AA_ORDER), torch.arange(20))
    assert batch["labels"].dtype == torch.long
    assert batch["input_mask"].dtype == torch.bool
    assert batch["lengths"].tolist() == [10, 20]
    assert batch["input_mask"].sum(1).tolist() == [10, 20]
    assert (batch["labels"][0, 10:] == -100).all()
    assert not batch["input_mask"][0, 10:].any()
    for i, record in enumerate(records):
        assert decode_labels(batch["labels"][i, :record.length]) == record.sequence
    with pytest.raises(ValueError):
        decode_labels(batch["labels"][0])
    with pytest.raises(ValueError):
        collate_peptides([])


def test_split_aliases_are_rejected_and_exact_overlap_reported(tmp_path):
    train, val = tmp_path / "train.fa", tmp_path / "val.fa"
    train.write_text(">train1\nACDEFGHIKL\n>train2\nAAAAAAAAAA\n")
    val.write_text(">val1\nACDEFGHIKL\n>val2\nCCCCCCCCCC\n")
    alias = tmp_path / "alias.fa"
    alias.symlink_to(train)
    hardlink = tmp_path / "hardlink.fa"
    hardlink.hardlink_to(train)
    for path in (train, alias, hardlink):
        with pytest.raises(ValueError, match="same file"):
            audit_splits({"train": train, "validation": path})
    audit = audit_splits({"train": train, "validation": val})
    assert audit["overlaps"]["train/validation"]["unique_sequences"] == 1
    assert audit["overlaps"]["train/validation"]["records"][0]["validation"] == ["val1"]
    assert audit["homology_checked"] is False
