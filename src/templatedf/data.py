"""FASTA data, fixed peptide labels and split auditing; no ESM loading."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch.utils.data import Dataset

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_LABEL = {aa: index for index, aa in enumerate(AA_ORDER)}
PAD_LABEL = -100
MIN_LENGTH = 10
MAX_LENGTH = 150


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PeptideRecord:
    id: str
    sequence: str
    description: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def sequence_hash(self) -> str:
        return hashlib.sha256(self.sequence.encode("utf-8")).hexdigest()

    @property
    def cache_key(self) -> str:
        identity = json.dumps([self.id, self.sequence_hash], ensure_ascii=False)
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def read_fasta(path: str | Path) -> Iterable[PeptideRecord]:
    """Join wrapped lines without uppercasing or removing sequence characters."""
    header = None
    pieces = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.rstrip("\r\n")
            if line.startswith(">"):
                if not line[1:].strip():
                    raise ValueError(f"Empty FASTA header at line {line_number}: {path}")
                if header is not None:
                    yield PeptideRecord(header.split()[0], "".join(pieces), header)
                header, pieces = line[1:], []
            elif line:
                if header is None:
                    raise ValueError(f"Sequence before FASTA header at line {line_number}: {path}")
                pieces.append(line)
    if header is not None:
        yield PeptideRecord(header.split()[0], "".join(pieces), header)


def validate_sequence(sequence: str) -> None:
    if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
        raise ValueError(f"Peptide length {len(sequence)} is outside {MIN_LENGTH}..{MAX_LENGTH}")
    invalid = sorted(set(sequence) - set(AA_ORDER))
    if invalid:
        raise ValueError(f"Nonstandard residue(s): {invalid!r}; input is not modified")


def encode_labels(sequence: str) -> torch.Tensor:
    validate_sequence(sequence)
    return torch.tensor([AA_TO_LABEL[aa] for aa in sequence], dtype=torch.long)


def decode_labels(labels: torch.Tensor | Iterable[int]) -> str:
    values = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
    if any(type(value) is not int or not 0 <= value < len(AA_ORDER) for value in values):
        raise ValueError("Labels must be 0..19; remove right padding using the true length")
    return "".join(AA_ORDER[value] for value in values)


@dataclass
class FilterStats:
    original: int = 0
    retained: int = 0
    excluded: int = 0
    too_short: int | None = 0
    too_long: int | None = 0
    nonstandard: int | None = 0
    duplicate_sequences: int = 0
    duplicate_ids: int = 0
    retained_duplicate_sequences: int = 0
    checks_enabled: bool = True


def load_fasta(path: str | Path, *, check_sequences: bool = True):
    """Filter records; duplicates are counted, not silently removed.

    Disable prechecks only for trusted inputs. Collation/ESM still enforce the
    20-label contract and raise on invalid input, rather than silently changing it.
    Rejection reasons may overlap; excluded counts rejected records only once.
    """
    stats = FilterStats(checks_enabled=check_sequences)
    if not check_sequences:
        stats.too_short = stats.too_long = stats.nonstandard = None
    records = []
    seen_sequences, seen_ids, retained_sequences = set(), set(), set()
    for record in read_fasta(path):
        stats.original += 1
        stats.duplicate_sequences += record.sequence in seen_sequences
        stats.duplicate_ids += record.id in seen_ids
        seen_sequences.add(record.sequence)
        seen_ids.add(record.id)
        if check_sequences:
            short = record.length < MIN_LENGTH
            long = record.length > MAX_LENGTH
            nonstandard = bool(set(record.sequence) - set(AA_ORDER))
            stats.too_short += short
            stats.too_long += long
            stats.nonstandard += nonstandard
            if short or long or nonstandard:
                stats.excluded += 1
                continue
        stats.retained_duplicate_sequences += record.sequence in retained_sequences
        retained_sequences.add(record.sequence)
        records.append(record)
    stats.retained = len(records)
    return records, stats


class PeptideDataset(Dataset):
    def __init__(self, path: str | Path, *, check_sequences: bool = True):
        self.records, self.stats = load_fasta(path, check_sequences=check_sequences)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_peptides(records: list[PeptideRecord]) -> dict:
    if not records:
        raise ValueError("Cannot collate an empty batch")
    encoded = [encode_labels(record.sequence) for record in records]
    lengths = torch.tensor([record.length for record in records], dtype=torch.long)
    labels = torch.full((len(records), int(lengths.max())), PAD_LABEL, dtype=torch.long)
    mask = torch.arange(labels.shape[1])[None, :] < lengths[:, None]
    for row, sequence_labels in enumerate(encoded):
        labels[row, :len(sequence_labels)] = sequence_labels
    return {"ids": [r.id for r in records], "sequences": [r.sequence for r in records],
            "labels": labels, "input_mask": mask, "lengths": lengths}


def audit_splits(paths: Mapping[str, str | Path], *, check_sequences: bool = True) -> dict:
    """Refuse identical files (including aliases); report exact sequence overlap."""
    normalized = {name: Path(path).resolve(strict=True) for name, path in paths.items()}
    names = list(normalized)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            if normalized[name].samefile(normalized[other]):
                raise ValueError(f"Splits {name} and {other} use the same file")
    result = {"splits": {}, "overlaps": {}, "homology_checked": False}
    sequence_ids = {}
    for name, path in normalized.items():
        records, stats = load_fasta(path, check_sequences=check_sequences)
        seq_ids = defaultdict(list)
        for record in records:
            seq_ids[record.sequence].append(record.id)
        sequence_ids[name] = seq_ids
        lengths = [record.length for record in records]
        result["splits"][name] = {
            "path": str(path), "file_sha256": file_sha256(path),
            "stats": asdict(stats), "min_length": min(lengths, default=None),
            "max_length": max(lengths, default=None),
            "length_histogram": dict(sorted(Counter(lengths).items())),
            "duplicate_sequence_groups": [
                {"sequence_hash": hashlib.sha256(seq.encode()).hexdigest(), "ids": ids}
                for seq, ids in seq_ids.items() if len(ids) > 1
            ],
        }
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            common = sorted(sequence_ids[name].keys() & sequence_ids[other].keys())
            result["overlaps"][f"{name}/{other}"] = {
                "unique_sequences": len(common),
                "records": [{"sequence_hash": hashlib.sha256(seq.encode()).hexdigest(),
                             name: sequence_ids[name][seq], other: sequence_ids[other][seq]}
                            for seq in common],
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--no-sequence-check", action="store_true",
                        help="Skip prefiltering trusted data; unassessed reason counts become null")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("train", "validation", "test")
             if getattr(args, name) is not None}
    if len(paths) < 2:
        parser.error("Provide --validation and/or --test in addition to --train")
    report = audit_splits(paths, check_sequences=not args.no_sequence_check)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: data["stats"] for name, data in report["splits"].items()}, indent=2))
    print("Exact overlap:", {name: data["unique_sequences"] for name, data in report["overlaps"].items()})


if __name__ == "__main__":
    main()
