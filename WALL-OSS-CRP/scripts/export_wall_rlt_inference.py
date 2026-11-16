#!/usr/bin/env python3
"""Export a completed RLinf actor checkpoint as WALL + encoder inference weights."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='actor/ directory with full_weights.pt and WALL-RLT sidecar')
    parser.add_argument('--output', required=True, help='New output directory; must be outside the source checkpoint')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wall_rlt.inference.artifact import export_inference_artifact
    manifest = export_inference_artifact(args.checkpoint, args.output)
    print(json.dumps({'status': manifest['validation_status'], 'output': str(Path(args.output).resolve()),
                      'artifact_id': manifest['artifact_id'], 'training_step': manifest['training_step'],
                      'includes_decoder': False, 'includes_actor': False}, indent=2))


if __name__ == '__main__':
    main()
