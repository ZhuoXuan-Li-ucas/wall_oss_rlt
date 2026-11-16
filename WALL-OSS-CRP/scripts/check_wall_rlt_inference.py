#!/usr/bin/env python3
"""Replay one existing observation through the inference service; no robot I/O."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


RUNTIME_SETTINGS = (
    'backend', 'training_step', 'device', 'inference_steps', 'image_size', 'max_length',
    'vision_attention', 'wall_attention', 'feature_dim', 'runtime_model_sources_sha256',
)


def _sha_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def bind_observation(observation):
    """Bind exactly the decoded RGB pixels/state/instruction sent to the service."""
    images = {}
    for name, image in observation['images'].items():
        if image.mode != 'RGB':
            raise ValueError('replay observation images must be RGB')
        images[name] = {'size': list(image.size), 'mode': image.mode,
                        'pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest()}
    components = {'instruction_sha256': _sha_json(observation['instruction']),
                  'state_sha256': _sha_json(observation['state']), 'images': images}
    return _sha_json(components), components


def validate_service(health, contract):
    metadata = health['metadata']
    if (metadata['model_origin'] != 'stage1_checkpoint'
            or metadata['backend'] != 'wall_rlt_stage1'
            or metadata['data_contract_hash'] != contract.fingerprint()):
        raise ValueError('service must use the same trained target contract')
    for key in ('artifact_id', 'runtime_model_sources_sha256'):
        if not isinstance(metadata.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', metadata[key]):
            raise ValueError(f'service requires a valid {key}')
    instance_id = health.get('server_instance_id')
    if not isinstance(instance_id, str) or not re.fullmatch(r'[0-9a-f]{32}', instance_id):
        raise ValueError('service requires a valid server_instance_id')
    expected = {key: getattr(contract, key) for key in (
        'action_horizon', 'action_names', 'action_units', 'action_convention',
        'coordinate_frame', 'camera_mapping', 'state_names', 'state_units')}
    expected.update(action_dim=len(contract.action_names),
                    includes_actor=False, includes_decoder=False, robot_execution_supported=False,
                    action_source='WALL_reference_prediction',
                    gripper_output='continuous_prediction_of_binary_target')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'service metadata differs from target contract/scope: {key}')
    for key, minimum, maximum in (
        ('training_step', 1, 2**63 - 1), ('inference_steps', 1, 100),
        ('image_size', 1, 448), ('max_length', 1, 2048), ('feature_dim', 1, 65536),
    ):
        if type(metadata.get(key)) is not int or not minimum <= metadata[key] <= maximum:
            raise ValueError(f'invalid runtime setting: {key}')
    if metadata.get('vision_attention') not in ('eager', 'flash_attention_2'):
        raise ValueError('unsupported vision attention setting')
    if metadata.get('wall_attention') != 'sdpa':
        raise ValueError('unsupported WALL attention setting')
    if not isinstance(metadata.get('device'), str) or not metadata['device']:
        raise ValueError('service requires an explicit device')
    return {key: metadata[key] for key in RUNTIME_SETTINGS}


def validate_prediction(response, metadata):
    from wall_rlt.inference.protocol import validate_response
    request_id = response.get('request_id')
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('replay prediction requires a request ID')
    validate_response(response, request_id=request_id, return_features=True)
    for key in ('backend', 'model_origin', 'data_contract_hash', 'action_units', 'action_convention'):
        if response[key] != metadata[key]:
            raise ValueError(f'prediction provenance differs from service: {key}')
    if len(response['z_rl']) != metadata['feature_dim']:
        raise ValueError('prediction feature dimension differs from service')


def difference(baseline, other):
    import numpy as np
    result = {}
    for key, output_key in (('actions', 'action_max_abs'), ('z_rl', 'feature_max_abs')):
        first = np.asarray(baseline[key], dtype=np.float64)
        second = np.asarray(other[key], dtype=np.float64)
        if first.shape != second.shape or not first.size:
            raise ValueError('replay result shapes differ or are empty')
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError('replay results must be finite')
        result[output_key] = float(np.max(np.abs(first - second)))
    return result


def compare_restart(previous, current):
    if previous.get('schema_version') != 2 or previous.get('status') != 'PASS':
        raise ValueError('restart comparison requires a successful version 2 report')
    for key in ('artifact_id', 'contract_hash', 'episode_id', 'frame_index', 'seed',
                'observation_sha256', 'runtime_settings', 'atol'):
        if previous.get(key) != current[key]:
            raise ValueError(f'restart comparison requires matching {key}')
    previous_instance = previous.get('server_instance_id')
    if not isinstance(previous_instance, str) or not re.fullmatch(r'[0-9a-f]{32}', previous_instance):
        raise ValueError('previous report requires a valid server_instance_id')
    if previous_instance == current['server_instance_id']:
        raise ValueError('restart comparison requires a different server_instance_id; restart the service')
    previous_response = previous['requests'][0]['response']
    validate_prediction(previous_response, current['metadata'])
    return difference(current['requests'][0]['response'], previous_response)


def _validate_output_location(path):
    for parent in path.parents:
        if any((parent / name).exists() for name in (
                'BUNDLE_READY.json', 'TRAINING_CANDIDATE.json', 'wall_rlt_data.json')):
            raise ValueError('--output must be outside immutable deployment/training artifacts')
        for name in ('manifest.json', 'MANIFEST.json'):
            marker = parent / name
            if not marker.is_file():
                continue
            try:
                manifest = json.loads(marker.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(manifest, dict) and (
                    manifest.get('kind') == 'wall-rlt-stage1-inference' or
                    manifest.get('status') == 'SOURCE_PACKAGE_NOT_GPU_VALIDATED'):
                raise ValueError('--output must be outside inference/source artifacts')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080')
    parser.add_argument('--contract', required=True, help='Existing dataset contract on this replay machine')
    parser.add_argument('--episode-id', required=True, type=int)
    parser.add_argument('--frame-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--compare-report', help='Previous successful report for a service restart comparison')
    parser.add_argument('--output', required=True, help='New JSON report outside artifact directories')
    args = parser.parse_args(argv)
    if (not 1 <= args.repeats <= 10 or args.episode_id < 0 or args.frame_index < 0
            or not 0 <= args.seed < 2**63 or not math.isfinite(args.atol) or args.atol < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('invalid repeat count, episode/frame index, seed, timeout or tolerance')
    output = Path(args.output).resolve()
    if output.exists():
        parser.error('--output must be a new report; preserve previous evidence')
    try:
        _validate_output_location(output)
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    report = {'schema_version': 2, 'status': 'FAIL',
              'scope': 'existing observation to WALL actions and RLT features through HTTP; no robot execution',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'robot_executed': False,
              'repeat_checked': args.repeats > 1, 'restart_checked': False}
    try:
        import numpy as np
        from PIL import Image
        from wall_rlt.data.contract import DataContract
        from wall_rlt.data.npz_source import build_source
        from wall_rlt.inference.client import InferenceClient
        from wall_rlt.inference.protocol import strict_json_loads
        contract = DataContract.load(args.contract)
        contract.require_trainable()
        client = InferenceClient(args.endpoint, timeout=args.timeout)
        health = client.health()
        metadata = health['metadata']
        settings = validate_service(health, contract)
        source = build_source(contract, episode_ids=[args.episode_id])
        indices = np.flatnonzero((np.asarray(source.episode_indices) == args.episode_id)
                                 & (np.asarray(source.frame_indices) == args.frame_index))
        if len(indices) != 1:
            raise ValueError('episode/frame must identify exactly one usable observation')
        row = source.observation(int(indices[0]))
        observation = {'instruction': row['instruction'], 'state': row['state'].tolist(),
            'images': {name: Image.fromarray(value.permute(1, 2, 0).numpy())
                       for name, value in row['images'].items()}}
        observation_hash, observation_components = bind_observation(observation)
        results = []
        report.update(artifact_id=metadata['artifact_id'], contract_hash=contract.fingerprint(),
            episode_id=args.episode_id, frame_index=args.frame_index, seed=args.seed,
            server_instance_id=health['server_instance_id'], runtime_settings=settings,
            observation_sha256=observation_hash, observation_components=observation_components,
            metadata=metadata, requests=results, atol=args.atol)
        for index in range(args.repeats):
            started = time.perf_counter()
            response = client.infer(observation, request_id=f'replay-{args.episode_id}-{args.frame_index}-{index}',
                                    return_features=True, seed=args.seed)
            validate_prediction(response, metadata)
            results.append({'elapsed_seconds': time.perf_counter() - started, 'response': response})
        after = client.health()
        report['metadata_after'] = after['metadata']
        if (after['server_instance_id'] != health['server_instance_id']
                or validate_service(after, contract) != settings):
            raise ValueError('service instance or runtime settings changed during replay')
        baseline = results[0]['response']
        comparisons = [difference(baseline, item['response']) for item in results[1:]]
        report['repeat_differences'] = comparisons
        if args.compare_report:
            previous = strict_json_loads(Path(args.compare_report).read_bytes())
            delta = compare_restart(previous, report)
            report['restart_difference'] = delta
            report['restart_checked'] = True
            comparisons = comparisons + [delta]
        if any(value > args.atol for item in comparisons for value in item.values()):
            raise ValueError('fixed-seed repeat/restart difference exceeds the declared tolerance')
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(output), 'robot_executed': False}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
