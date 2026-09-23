"""Atomic, weights-only-loadable checkpoints and complete process RNG state."""

import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng(device):
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':{
        'name':state[0],'keys':torch.from_numpy(state[1].astype(np.int64)),
        'position':state[2],'has_gauss':state[3],'cached_gaussian':state[4]},
        'torch_cpu':torch.get_rng_state(),
        'torch_cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    value=state['numpy']
    np.random.set_state((value['name'],value['keys'].cpu().numpy().astype(np.uint32),
                         value['position'],value['has_gauss'],value['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type=='cuda':
        if state['torch_cuda'] is None:
            raise ValueError("CUDA RNG state missing from resume checkpoint")
        torch.cuda.set_rng_state(state['torch_cuda'].cpu(),device)


def atomic_save(payload, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            torch.save(payload,stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    required={'format_version','stage','config','model_config','training_settings','model',
              'optimizer','scheduler','scaler','step','rng','sampler','aa_order',
              'esm_source','data_provenance','precision_runtime'}
    if not isinstance(state,dict) or set(state)!=required or state['format_version']!=1:
        raise ValueError("Unrecognized or incomplete training checkpoint")
    return state
