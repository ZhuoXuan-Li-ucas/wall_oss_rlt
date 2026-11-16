"""A bounded observation-only wire contract for offline WALL-RLT inference.

The contract transports model predictions, never robot execution commands.
Image names deliberately follow the frozen dataset: cam_high is the external
camera and cam_front is the wrist camera, despite the latter's historical name.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import io
import json
import math
import numbers
import re
import warnings

from PIL import Image, UnidentifiedImageError


SCHEMA_VERSION = 1
CAMERA_KEYS = ('cam_high', 'cam_front')
ACTION_HORIZON = 32
ACTION_DIM = STATE_DIM = 8          # AR5 joint8 contract
SUPPORTED_DIMS = (8, 10)            # joint8, Wall-OSS-0.5 right-arm EE (pos, 6D, gripper)
ABSOLUTE_TARGET_DIM = 7             # EE only: flange pos, rotation vector, gripper
MAX_BODY_BYTES = 16 * 1024**2
MAX_IMAGE_BYTES = 8 * 1024**2
MAX_IMAGE_SIDE = 2048
MAX_FEATURE_DIM = 65536


class ProtocolError(ValueError):
    def __init__(self, message, *, code='invalid_request', status=400):
        super().__init__(message)
        self.code, self.status = code, status


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    observation: dict
    return_features: bool = False
    seed: int | None = None
    rtc: dict | None = None             # EE only: previous chunk's leftover for RTC guidance


def _object(value, allowed, required, label):
    if not isinstance(value, dict):
        raise ProtocolError(f'{label} must be an object')
    if set(value) - set(allowed) or not set(required).issubset(value):
        raise ProtocolError(f'{label} has missing or unknown fields')


def _text(value, label, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ProtocolError(f'{label} must contain 1 to {maximum} characters')
    if any(ord(character) < 32 and character not in '\n\r\t' for character in value):
        raise ProtocolError(f'{label} contains unsupported control characters')
    try:
        value.encode('utf-8')
    except UnicodeError:
        raise ProtocolError(f'{label} must contain valid Unicode text') from None
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ProtocolError(f'{label} must contain finite real numbers')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ProtocolError(f'{label} must contain finite real numbers') from None
    if not math.isfinite(result):
        raise ProtocolError(f'{label} must contain finite real numbers')
    return result


def _vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ProtocolError(f'{label} must have length {length}')
    return [_finite(item, label) for item in value]


def strict_json_loads(data):
    """Reject duplicate fields and JSON's nonstandard NaN/Infinity extensions."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError('JSON object contains duplicate fields')
            result[key] = value
        return result
    def constant(_):
        raise ProtocolError('JSON must not contain NaN or Infinity')
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        if isinstance(error, ProtocolError):
            raise
        raise ProtocolError('request must be valid UTF-8 JSON') from None


def json_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, OverflowError, UnicodeError):
        raise ProtocolError('value cannot be represented as finite JSON') from None


def decode_image(value):
    _object(value, {'encoding', 'data'}, {'encoding', 'data'}, 'image')
    encoding, encoded = value['encoding'], value['data']
    if encoding not in ('png', 'jpeg') or not isinstance(encoded, str):
        raise ProtocolError('image requires png/jpeg encoding and base64 data')
    if not encoded or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ProtocolError('encoded image exceeds the image byte limit', status=413,
                            code='image_too_large')
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ProtocolError('image data must be valid base64') from None
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ProtocolError('image exceeds the image byte limit', status=413,
                            code='image_too_large')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if source.format != {'png': 'PNG', 'jpeg': 'JPEG'}[encoding]:
                    raise ProtocolError('image bytes disagree with declared encoding')
                if source.mode != 'RGB':
                    raise ProtocolError('images must be RGB, without alpha or grayscale channels')
                if not all(1 <= side <= MAX_IMAGE_SIDE for side in source.size):
                    raise ProtocolError('image dimensions exceed the supported limit',
                                        code='image_too_large', status=413)
                if getattr(source, 'n_frames', 1) != 1:
                    raise ProtocolError('each image must contain exactly one frame')
                source.load()
                return source.copy()
    except ProtocolError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning, SyntaxError):
        raise ProtocolError('image cannot be decoded as a supported RGB image') from None


def encode_image(image, encoding='png'):
    """Encode an already aligned RGB PIL image; no rotation/resize is inferred."""
    if not isinstance(image, Image.Image) or image.mode != 'RGB':
        raise ProtocolError('images must be RGB PIL images')
    if encoding not in ('png', 'jpeg'):
        raise ProtocolError('image encoding must be png or jpeg')
    if not all(1 <= side <= MAX_IMAGE_SIDE for side in image.size):
        raise ProtocolError('image dimensions exceed the supported limit', status=413,
                            code='image_too_large')
    buffer = io.BytesIO()
    image.save(buffer, format={'png': 'PNG', 'jpeg': 'JPEG'}[encoding])
    payload = buffer.getvalue()
    if len(payload) > MAX_IMAGE_BYTES:
        raise ProtocolError('image exceeds the image byte limit', status=413,
                            code='image_too_large')
    return {'encoding': encoding, 'data': base64.b64encode(payload).decode('ascii')}


def _layout_dim(value, label):
    if type(value) is not int or value not in SUPPORTED_DIMS:
        raise ProtocolError(f'{label} must be one of {SUPPORTED_DIMS}')
    return value


RTC_METHODS = ('inpaint', 'guidance')


def _rtc(value):
    """Previous chunk's not-yet-superseded absolute EE targets, one row per future
    step from this observation, the inference delay in steps and the method."""
    _object(value, {'prefix_targets', 'inference_delay', 'method'},
            {'prefix_targets', 'inference_delay'}, 'rtc')
    method = value.get('method', 'inpaint')
    if method not in RTC_METHODS:
        raise ProtocolError(f'rtc method must be one of {RTC_METHODS}')
    targets = value['prefix_targets']
    if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= ACTION_HORIZON:
        raise ProtocolError(f'rtc prefix_targets must have 1 to {ACTION_HORIZON} steps')
    delay = value['inference_delay']
    if type(delay) is not int or not 0 <= delay <= ACTION_HORIZON:
        raise ProtocolError(f'rtc inference_delay must be an integer in [0, {ACTION_HORIZON}]')
    return {'prefix_targets': [_vector(row, ABSOLUTE_TARGET_DIM, 'rtc prefix_targets') for row in targets],
            'inference_delay': delay, 'method': method}


def decode_request(value, *, state_dim=None):
    """``state_dim`` comes from the backend contract; None accepts any supported layout."""
    required = {'schema_version', 'request_id', 'instruction', 'state', 'images'}
    _object(value, required | {'return_features', 'seed', 'rtc'}, required, 'request')
    if type(value['schema_version']) is not int or value['schema_version'] != SCHEMA_VERSION:
        raise ProtocolError('unsupported schema_version')
    request_id = _text(value['request_id'], 'request_id', 128)
    if any(character in request_id for character in '\r\n\t'):
        raise ProtocolError('request_id must be a single line')
    instruction = _text(value['instruction'], 'instruction', 4096)
    if state_dim is None:
        given = len(value['state']) if isinstance(value['state'], (list, tuple)) else None
        state_dim = given if given in SUPPORTED_DIMS else STATE_DIM
    state = _vector(value['state'], _layout_dim(state_dim, 'state_dim'), 'state')
    _object(value['images'], CAMERA_KEYS, CAMERA_KEYS, 'images')
    return_features = value.get('return_features', False)
    if type(return_features) is not bool:
        raise ProtocolError('return_features must be boolean')
    seed = value.get('seed')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ProtocolError('seed must be an integer in [0, 2^63)')
    rtc = value.get('rtc')
    if rtc is not None:
        if state_dim != 10:
            raise ProtocolError('rtc requires the EE (10-dimensional state) contract')
        rtc = _rtc(rtc)
    images = {key: decode_image(value['images'][key]) for key in CAMERA_KEYS}
    return InferenceRequest(request_id, {'instruction': instruction, 'state': state,
                                        'images': images}, return_features, seed, rtc)


def make_request(observation, *, request_id, return_features=False, seed=None,
                 image_encoding='png', rtc=None):
    """Build and validate the wire request from an observation with PIL images."""
    _object(observation, {'instruction', 'state', 'images'},
            {'instruction', 'state', 'images'}, 'observation')
    _object(observation['images'], CAMERA_KEYS, CAMERA_KEYS, 'images')
    value = {'schema_version': SCHEMA_VERSION, 'request_id': request_id,
             'instruction': observation['instruction'], 'state': observation['state'],
             'images': {key: encode_image(observation['images'][key], image_encoding)
                        for key in CAMERA_KEYS}, 'return_features': return_features,
             'seed': seed}
    if rtc is not None:
        value['rtc'] = rtc
    decode_request(value)
    if len(json_bytes(value)) > MAX_BODY_BYTES:
        raise ProtocolError('request exceeds the body byte limit', status=413,
                            code='request_too_large')
    return value


def validate_metadata(metadata):
    required = {'backend', 'data_contract_hash', 'action_horizon', 'action_dim',
                'action_units', 'action_convention', 'model_origin'}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ProtocolError('backend metadata is incomplete')
    for key in ('backend', 'model_origin', 'action_convention'):
        _text(metadata[key], key, 256)
    if not isinstance(metadata['data_contract_hash'], str) or not re.fullmatch(
            '[0-9a-f]{64}', metadata['data_contract_hash']):
        raise ProtocolError('backend requires a SHA256 data contract hash')
    if (type(metadata['action_horizon']) is not int or metadata['action_horizon'] != ACTION_HORIZON or
            type(metadata['action_dim']) is not int or metadata['action_dim'] not in SUPPORTED_DIMS):
        raise ProtocolError('backend action layout must be 32 by 8 (joint8) or 32 by 10 (EE)')
    units = metadata['action_units']
    if not isinstance(units, list) or len(units) != metadata['action_dim']:
        raise ProtocolError('backend must declare every physical action unit')
    for unit in units:
        _text(unit, 'action unit', 128)
    if 'state_dim' in metadata:
        _layout_dim(metadata['state_dim'], 'state_dim')
    if 'absolute_target_dim' in metadata and metadata['absolute_target_dim'] != ABSOLUTE_TARGET_DIM:
        raise ProtocolError('absolute EE targets must have 7 values per step')
    # Copy through strict JSON so optional backend diagnostics must also be finite.
    return strict_json_loads(json_bytes(metadata))


def make_response(result, request, metadata):
    """Validate predictions and attach provenance without an execution claim."""
    _object(result, {'actions', 'absolute_targets', 'z_rl', 'timing_seconds'}, {'actions'},
            'backend response')
    actions = result['actions']
    if not isinstance(actions, (list, tuple)) or len(actions) != ACTION_HORIZON:
        raise ProtocolError('backend actions must have 32 steps')
    action_dim = metadata.get('action_dim', ACTION_DIM)
    response = {'schema_version': SCHEMA_VERSION, 'status': 'ok',
                'request_id': request.request_id,
                'actions': [_vector(row, action_dim, 'actions') for row in actions],
                'data_contract_hash': metadata['data_contract_hash'],
                'model_origin': metadata['model_origin'], 'backend': metadata['backend'],
                'action_space': 'physical', 'action_units': metadata['action_units'],
                'action_convention': metadata['action_convention'], 'robot_executed': False}
    if ('absolute_targets' in result) != ('absolute_target_dim' in metadata):
        raise ProtocolError('absolute EE targets must match the backend declaration')
    if 'absolute_targets' in result:
        targets = result['absolute_targets']
        if not isinstance(targets, (list, tuple)) or len(targets) != ACTION_HORIZON:
            raise ProtocolError('absolute EE targets must have 32 steps')
        response['absolute_targets'] = [_vector(row, ABSOLUTE_TARGET_DIM, 'absolute_targets')
                                        for row in targets]
    if request.return_features:
        features = result.get('z_rl')
        if not isinstance(features, (list, tuple)) or not 1 <= len(features) <= MAX_FEATURE_DIM:
            raise ProtocolError('backend must return a bounded nonempty z_rl vector when requested')
        response['z_rl'] = _vector(features, len(features), 'z_rl')
    elif 'z_rl' in result:
        raise ProtocolError('backend returned unrequested features')
    if 'timing_seconds' in result:
        timing = _finite(result['timing_seconds'], 'timing_seconds')
        if timing < 0:
            raise ProtocolError('timing_seconds must not be negative')
        response['timing_seconds'] = timing
    return response


def validate_response(value, *, request_id, return_features=False):
    """Validate server results again at the client before exposing predictions."""
    required = {'schema_version', 'status', 'request_id', 'actions', 'data_contract_hash',
                'model_origin', 'backend', 'action_space', 'action_units',
                'action_convention', 'robot_executed'}
    _object(value, required | {'absolute_targets', 'z_rl', 'timing_seconds'}, required, 'response')
    if (type(value['schema_version']) is not int or value['schema_version'] != SCHEMA_VERSION or
            value['status'] != 'ok' or value['request_id'] != request_id or
            value['action_space'] != 'physical' or value['robot_executed'] is not False):
        raise ProtocolError('response identity, schema or execution claim is invalid')
    units = value['action_units']
    metadata = validate_metadata({key: value[key] for key in
        ('backend', 'data_contract_hash', 'action_units', 'action_convention', 'model_origin')} |
        {'action_horizon': ACTION_HORIZON,
         'action_dim': len(units) if isinstance(units, list) else ACTION_DIM} |
        ({'absolute_target_dim': ABSOLUTE_TARGET_DIM} if 'absolute_targets' in value else {}))
    result = {key: value[key] for key in ('actions', 'absolute_targets', 'z_rl', 'timing_seconds')
              if key in value}
    return make_response(result, InferenceRequest(request_id, {}, return_features), metadata)
