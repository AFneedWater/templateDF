"""Pilot selection must respect budgets and keep held-out records out of train."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts/run_pilot.py'
spec = importlib.util.spec_from_file_location('pilot', MODULE_PATH)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_proportional_quotas_preserve_budget_and_source_capacity():
    counts = {'10-29': 13, '30-49': 7, '50-99': 40, '100-150': 0}
    actual = pilot.quotas(counts, 17)
    assert sum(actual.values()) == 17
    assert actual == {'10-29': 4, '30-49': 2, '50-99': 11, '100-150': 0}
    assert all(0 <= actual[key] <= counts[key] for key in counts)
    with pytest.raises(ValueError, match='Insufficient'):
        pilot.quotas(counts, 61)


def test_real_selection_holdout_is_reproducible_disjoint_and_bounded(tmp_path, monkeypatch):
    from templatedf.data import AA_ORDER, load_fasta
    (tmp_path/'configs').mkdir()
    train = tmp_path/'source.fasta'
    test = tmp_path/'test.fasta'
    lines = []
    for length in (15, 35, 65, 120):
        for index in range(200):
            code = AA_ORDER[index//20] + AA_ORDER[index%20]
            lines.append(f'>{length}_{index}\n{code + "A"*(length-2)}\n')
    train.write_text(''.join(lines))
    test.write_text('>test\nQQQQQQQQQQQQQ\n')
    (tmp_path/'configs/train.yaml').write_text(yaml.safe_dump({'data':{'train_path':str(train),'test_path':str(test)}}))
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    monkeypatch.setattr(pilot, 'ART', tmp_path/'first')
    monkeypatch.setattr(pilot, 'LOG', tmp_path/'logs')
    with pytest.raises(ValueError, match='confirmed'):
        pilot.prepare(None)
    pilot.prepare('holdout-from-train')
    first = {name:(pilot.ART/f'{name}.fasta').read_text() for name in ('train','validation')}
    groups = {name:load_fasta(pilot.ART/f'{name}.fasta')[0] for name in first}
    assert len(groups['train']) == 512 and len(groups['validation']) == 128
    assert not ({r.sequence for r in groups['train']} & {r.sequence for r in groups['validation']})
    report = json.loads((pilot.LOG/'data_selection.json').read_text())
    assert list(report['selected_distribution']['train'].values()) == [128]*4
    assert list(report['selected_distribution']['validation'].values()) == [32]*4
    monkeypatch.setattr(pilot, 'ART', tmp_path/'second')
    pilot.prepare('holdout-from-train')
    assert all((pilot.ART/f'{name}.fasta').read_text() == value for name,value in first.items())
