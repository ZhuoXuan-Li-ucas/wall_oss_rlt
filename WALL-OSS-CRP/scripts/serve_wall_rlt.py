#!/usr/bin/env python3
"""Serve trained WALL reference actions and optional RLT features on localhost."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--wall-source', help='Optional isolated patched WALL tree; otherwise use installed WALL')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--vision-attention', choices=['flash_attention_2','eager'], default='flash_attention_2')
    parser.add_argument('--inference-steps', type=int, default=10)
    parser.add_argument('--check-only', action='store_true', help='Verify artifact integrity; do not construct a model or open a port')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be in [0,65535]')
    sys.path.insert(0, str(root))
    if args.wall_source:
        source = Path(args.wall_source).resolve()
        if not (source/'wall_x/__init__.py').is_file():
            parser.error('--wall-source must contain the patched wall_x package')
        sys.path.insert(0, str(source))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    if args.check_only:
        from wall_rlt.inference.artifact import verify_inference_artifact
        manifest = verify_inference_artifact(args.artifact)
        if manifest.get('data_purpose') != 'target':
            raise ValueError('serving requires a trained target artifact, not synthetic test weights')
        print(json.dumps({'status': 'ARTIFACT_INTEGRITY_PASS', 'artifact_id': manifest['artifact_id'],
                          'model_loaded': False, 'gpu_inference': 'NOT_RUN'}, indent=2))
        return
    from wall_rlt.inference.runtime import load_backend
    from wall_rlt.inference.server import create_server
    backend = load_backend(args.artifact, device=args.device, vision_attention=args.vision_attention,
                           inference_steps=args.inference_steps)
    server = create_server(backend, host=args.host, port=args.port)
    print(json.dumps({'status': 'LISTENING', 'address': server.server_address,
                      'metadata': backend.metadata(), 'full_inference_acceptance': 'NOT_RUN'},
                     ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
