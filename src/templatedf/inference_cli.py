"""Shared inference CLI parser; --help imports no model/ESM or tensor libraries."""
import argparse
import math
from pathlib import Path


def parser_for(mode):
    parser = argparse.ArgumentParser(description={
        'generate': 'Generate temperature-sampled candidates from template latents (not diffusion).',
        'reconstruct': 'Reconstruct each template at its original length with argmax.',
        'export_latents': 'Export raw, unnormalized template latents in bounded shards.',
    }[mode])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fasta', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=('fp32','bf16'), default='fp32')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--esm-checkpoint', type=Path)
    parser.add_argument('--regression-checkpoint', type=Path)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--limit', type=int, help='Maximum templates; invalid templates are rejected, not skipped')
    budget.add_argument('--all', action='store_true', help='Explicitly process every template')
    if mode == 'generate':
        parser.add_argument('--target-length', type=int)
        parser.add_argument('--num-samples', type=int, default=1)
        parser.add_argument('--temperature', type=float, default=0.0)
        parser.add_argument('--seed', type=int, default=42)
    if mode == 'export_latents':
        parser.add_argument('--shard-size', type=int, default=256)
    return parser


def main(mode):
    parser = parser_for(mode)
    args = parser.parse_args()
    for name in ('batch_size','limit','num_samples','shard_size'):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f'{name} must be positive')
    if mode == 'generate':
        if args.target_length is not None and not 10 <= args.target_length <= 150:
            parser.error('target-length must be in 10..150')
        if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 <= args.seed < 2**63:
            parser.error('temperature must be finite/nonnegative; seed must be in [0,2**63)')
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new directory')
    # Parse and validate CLI arguments before heavy imports or model construction.
    import json
    from .inference import TemplateGenerator
    from .inference_io import inspect_fasta, selected_records, write_candidates, export_raw_latents
    info = inspect_fasta(args.fasta, args.limit)
    runtime = TemplateGenerator.from_checkpoint(args.checkpoint, device=args.device, precision=args.precision,
                                               esm_checkpoint=args.esm_checkpoint, regression_checkpoint=args.regression_checkpoint)
    records = selected_records(args.fasta, args.limit)
    if mode == 'export_latents':
        result = export_raw_latents(runtime, records, args.output_dir, expected_records=info['selected_records'],
                                    shard_size=args.shard_size, batch_size=args.batch_size, input_info=info)
    else:
        settings = {'target_length':args.target_length, 'num_samples':args.num_samples,
                    'temperature':args.temperature, 'seed':args.seed} if mode == 'generate' else {
                    'target_length':None, 'num_samples':1, 'temperature':0.0, 'seed':42}
        result = write_candidates(runtime, records, args.output_dir, expected_records=info['selected_records'],
                                  input_info=info, batch_size=args.batch_size, **settings)
    print(json.dumps({'output_dir':str(args.output_dir.resolve()),
                      **{k:result[k] for k in ('template_count','candidate_count','record_count','within_template_duplicate_rate') if k in result}}, indent=2))
