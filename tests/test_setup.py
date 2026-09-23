"""Stage 00 acceptance checks; no model or ESM weights are loaded."""

from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_installed_import_is_local_and_has_no_heavy_dependencies():
    # A fresh isolated interpreter also excludes cwd/PYTHONPATH as import crutches.
    code = """
import importlib.abc
from pathlib import Path
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'esm', 'raygun'}:
            raise AssertionError(f'Unexpected dependency during import: {fullname}')

sys.meta_path.insert(0, BlockHeavyImports())
import templatedf
assert Path(templatedf.__file__).resolve() == Path(sys.argv[1]).resolve()
print(templatedf.__file__)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT / "src/templatedf/__init__.py")],
        cwd="/tmp", capture_output=True, text=True, check=True,
    )
    assert str(ROOT / "src/templatedf/__init__.py") in result.stdout


def test_default_configuration_matches_contract():
    config = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
    assert config["data"] == {
        "min_seq_len": 10, "max_seq_len": 150,
        "aa_order": "ACDEFGHIKLMNPQRSTVWY", "padding_label": -100,
        "nonstandard_residues": "skip_record", "check_sequences": True,
        "train_path": str(ROOT / "AgsgeGpr3I_10-150aa_train_140000.fasta"),
        "test_path": str(ROOT / "AgsgeGpr3I_10-150aa_test_30000.fasta"),
        "validation_path": None,
    }
    model = config["model"]
    assert model == {
        "dim": 1280, "num_latents": 50, "encoder_blocks": 4,
        "decoder_blocks": 4, "num_heads": 20, "conv_kernel": 7,
        "fusion_hidden_dim": 2560, "encoder_ffn_ratio": 2,
        "pool_ffn_ratio": 4, "decoder_ffn_ratio": 4,
        "dropout": 0.1, "length_scale": 150, "condition": None,
    }
    assert model["dim"] % model["num_heads"] == 0
    assert config["esm"]["model_name"] == "esm2_t33_650M_UR50D"
    assert config["esm"]["representation_layer"] == 33
    assert config["esm"]["frozen"] is True
    assert config["esm"]["eval_mode"] is True
    assert config["training"] == {
        "stage": "1A", "precision": "bf16", "batch_size": 8, "lambda_emb": 0.0,
        "device": "cuda", "max_steps": None, "learning_rate": 0.0001,
        "weight_decay": 0.01, "grad_clip": 1.0, "seed": 42,
        "eval_every": 100, "save_every": 100, "scheduler_gamma": 1.0,
    }
