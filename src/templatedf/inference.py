"""Frozen template-conditioned AE inference; sampling uses one private CPU RNG."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

import torch

from .data import AA_ORDER, PeptideRecord, decode_labels, file_sha256, validate_sequence


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def validate_generation(target_length=None, num_samples=1, temperature=0.0, seed=42, condition=None):
    if condition is not None:
        raise NotImplementedError('Conditional generation is not implemented; use condition=None')
    if target_length is not None and (type(target_length) is not int or not 10 <= target_length <= 150):
        raise ValueError('target_length must be an integer in 10..150 or None')
    positive_integer(num_samples, 'num_samples')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')


def sample_logits(logits, temperature, generator):
    """CPU float64 softmax after centering supports tiny positive temperatures."""
    if logits.ndim != 2 or logits.shape[1] != 20 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite residue logits [L,20]')
    if temperature == 0:
        return logits.argmax(-1).cpu()
    scores = logits.detach().to(device='cpu', dtype=torch.float64)
    scores = (scores - scores.max(-1, keepdim=True).values) / temperature
    probabilities = scores.softmax(-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


class TemplateGenerator:
    """Bind a checkpoint/model and frozen ESM once, then process many templates.

    Direct construction supports explicitly labelled mock extractors in tests.
    Production callers should use from_checkpoint, which verifies local ESM hashes.
    """
    def __init__(self, model, extractor, *, provenance, device='cpu', precision='fp32'):
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU and CUDA are supported')
        if precision not in ('fp32', 'bf16'):
            raise ValueError('precision must be fp32 or bf16')
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable')
            self.device = torch.device('cuda', self.device.index if self.device.index is not None else torch.cuda.current_device())
        if precision == 'bf16':
            if self.device.type != 'cuda':
                raise ValueError('BF16 inference requires native CUDA BF16 support')
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise RuntimeError('Native CUDA BF16 support is required')
        if provenance.get('aa_order') != AA_ORDER or not provenance.get('checkpoint', {}).get('sha256'):
            raise ValueError('AA order and checkpoint identity are required')
        if not provenance.get('esm_source') or not provenance.get('model_config'):
            raise ValueError('Model configuration and ESM provenance are required')
        if provenance['model_config'].get('dim', 1280) != model.dim or provenance['model_config'].get('num_latents', 50) != model.num_latents:
            raise ValueError('Model dimensions disagree with provenance')
        self.model = model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.extractor = extractor
        self.precision = precision
        self.provenance = deepcopy(provenance)
        storage = provenance['esm_source'].get('storage_dtype', 'float32')
        if storage not in ('float16', 'float32', 'bfloat16'):
            raise ValueError('Unsupported checkpoint feature storage dtype')
        self.feature_dtype = getattr(torch, storage)
        self.provenance['inference'] = {
            'device': str(self.device), 'precision': precision, 'condition': None,
            'esm_compute_dtype': 'float32', 'encoder_input_dtype': 'float32',
            'feature_roundtrip_dtype': storage, 'sampling_device': 'cpu',
            'sampling_dtype': 'float64', 'latent_normalization': None,
            'torch_version': str(torch.__version__),
        }
        self.identity = {key:deepcopy(self.provenance[key]) for key in ('aa_order', 'esm_source', 'model_config')}
        self.identity.update(checkpoint_sha256=provenance['checkpoint']['sha256'],
                             precision=precision, feature_roundtrip_dtype=storage)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device='cpu', precision='fp32', esm_checkpoint=None,
                        regression_checkpoint=None):
        from .checkpoint import load_checkpoint
        from .esm_features import ESMFeatureExtractor, local_cache_spec
        from .model import ProteinAutoencoder
        from .training import model_settings
        checkpoint = Path(checkpoint).resolve(strict=True)
        checkpoint_hash = file_sha256(checkpoint)
        state = load_checkpoint(checkpoint)
        if state['stage'] != '1A' or state['aa_order'] != AA_ORDER:
            raise ValueError('Expected a stage 1A checkpoint with the fixed AA order')
        if model_settings(state['config']) != state['model_config']:
            raise ValueError('Checkpoint model configuration is inconsistent')
        if state['model_config'].get('dim', 1280) != 1280:
            raise ValueError('Local ESM650M requires a 1280-dimensional model')
        esm_cfg = state['config'].get('esm', {})
        esm_path = esm_checkpoint or esm_cfg.get('checkpoint_path')
        regression_path = regression_checkpoint or esm_cfg.get('contact_regression_path')
        if not esm_path or not regression_path:
            raise ValueError('Both local ESM and regression checkpoint paths are required')
        spec = local_cache_spec(esm_path, regression_path, storage_dtype=state['esm_source'].get('storage_dtype', 'float16'))
        if asdict(spec) != state['esm_source']:
            raise ValueError('Local ESM source differs from the AE training checkpoint')
        provenance = {'checkpoint': {'path': str(checkpoint), 'sha256': checkpoint_hash, 'step': state['step'], 'stage': state['stage']},
                      'config': deepcopy(state['config']), 'model_config': deepcopy(state['model_config']),
                      'aa_order': state['aa_order'], 'esm_source': deepcopy(state['esm_source'])}
        weights = state['model']
        del state  # Optimizer/RNG tensors are not needed by inference.
        # Model constructors consume CPU RNG; preserve the caller's state.
        with torch.random.fork_rng(devices=[]):
            model = ProteinAutoencoder(**provenance['model_config'])
            model.load_state_dict(weights, strict=True)
            del weights
            extractor = ESMFeatureExtractor.from_local(esm_path, regression_path, device=device)
        return cls(model, extractor, provenance=provenance, device=device, precision=precision)

    def _autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.precision == 'bf16' else nullcontext()

    @torch.no_grad()
    def encode_records(self, records, condition=None):
        if condition is not None:
            raise NotImplementedError('Use condition=None')
        if not records:
            raise ValueError('At least one template is required')
        for record in records:
            validate_sequence(record.sequence)
        self.model.eval().requires_grad_(False)
        features = self.extractor.extract(records)
        if len(features) != len(records):
            raise ValueError('ESM returned an incorrect record count')
        lengths = torch.tensor([r.length for r in records], device=self.device)
        embeddings = torch.zeros(len(records), int(lengths.max()), self.model.dim, device=self.device)
        for index, (record, feature) in enumerate(zip(records, features)):
            if feature.shape != (record.length, self.model.dim) or not torch.isfinite(feature).all():
                raise ValueError('Invalid ESM feature dimensions/values')
            # Reproduce the feature quantization used by cached training.
            embeddings[index, :record.length] = feature.detach().to(self.feature_dtype).float().to(self.device)
        mask = torch.arange(embeddings.shape[1], device=self.device)[None, :] < lengths[:, None]
        with self._autocast():
            latent = self.model.encode(embeddings, mask, condition=None)
        if latent.shape != (len(records), self.model.num_latents, self.model.dim) or not torch.isfinite(latent).all():
            raise ValueError('Invalid encoded latent')
        return latent.detach()

    @torch.no_grad()
    def decode_latents(self, latent, output_lengths, condition=None):
        if condition is not None:
            raise NotImplementedError('Use condition=None')
        self.model.eval().requires_grad_(False)
        if not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError('Latent must be finite floating-point values')
        with self._autocast():
            result = self.model.decode(latent.to(self.device), output_lengths.to(self.device), condition=None)
        if not torch.isfinite(result['logits'][result['output_mask']]).all():
            raise ValueError('Nonfinite decoder logits')
        return result

    def iter_encoded(self, records, batch_size=4, condition=None):
        from itertools import islice
        positive_integer(batch_size, 'batch_size')
        if condition is not None:
            raise NotImplementedError('Use condition=None')
        iterator = iter(records)
        while batch := list(islice(iterator, batch_size)):
            yield batch, self.encode_records(batch)

    def iter_generate(self, records, *, target_length=None, num_samples=1, temperature=0.0,
                      seed=42, condition=None, batch_size=4):
        validate_generation(target_length, num_samples, temperature, seed, condition)
        positive_integer(batch_size, 'batch_size')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        template_index = 0
        for batch, latent in self.iter_encoded(records, batch_size):
            lengths = torch.tensor([r.length if target_length is None else target_length for r in batch], device=self.device)
            output = self.decode_latents(latent, lengths)
            scores = output['logits'].detach().cpu()
            for index, record in enumerate(batch):
                length = int(lengths[index])
                for sample_index in range(num_samples):
                    labels = sample_logits(scores[index, :length], temperature, generator)
                    yield {'candidate_id': f'template_{template_index:08d}_sample_{sample_index:06d}',
                           'template_index': template_index, 'template_id': record.id,
                           'template_sequence_hash': record.sequence_hash, 'template_length': record.length,
                           'target_length': length, 'sample_index': sample_index, 'seed': seed,
                           'temperature': temperature, 'condition': None, 'sequence': decode_labels(labels),
                           'checkpoint': deepcopy(self.provenance['checkpoint']),
                           'method': 'argmax' if temperature == 0 else 'temperature_sampling'}
                template_index += 1

    def generate_from_template(self, template_sequence, target_length=None, num_samples=1,
                               temperature=0.0, seed=42, condition=None):
        """Return candidate dictionaries for a single standard 10–150 aa template."""
        validate_generation(target_length, num_samples, temperature, seed, condition)
        validate_sequence(template_sequence)
        return list(self.iter_generate([PeptideRecord('template', template_sequence)], target_length=target_length,
                                       num_samples=num_samples, temperature=temperature, seed=seed, condition=condition))
