"""Command-line entry point for proximal matching."""
import argparse
import json
from pathlib import Path
from pgm.config import dump_yaml
from .config import load_config
from .engine import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    for source, section, field in ((args.data, config.data, 'train_dir'),
            (args.validation, config.data, 'validation_dir'), (args.checkpoint, config.model, 'checkpoint'),
            (args.out, config.output, 'directory')):
        if source is not None:
            setattr(section, field, str(source.resolve()))
    if args.device:
        config.runtime.device = args.device
    config.validate()
    if args.dry_run:
        print(dump_yaml(config.to_dict()))
        return
    print(json.dumps(run_training(config, args.resume, args.stop_after), indent=2))
