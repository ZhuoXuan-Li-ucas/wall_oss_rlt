"""Wire-contract checks use tiny generated RGB images, never model weights."""
import base64
import io

import pytest
from PIL import Image

from wall_rlt.inference.protocol import (
    CAMERA_KEYS, InferenceRequest, ProtocolError, decode_image, decode_request,
    encode_image, json_bytes, make_request, make_response, strict_json_loads,
    validate_metadata, validate_response,
)


def observation():
    return {'instruction': '抓起杯子', 'state': [0.] * 8,
            'images': {key: Image.new('RGB', (28, 28), (30, 60, 90)) for key in CAMERA_KEYS}}


def metadata():
    return {'backend': 'fixture', 'model_origin': 'synthetic_fixture',
            'data_contract_hash': 'a' * 64, 'action_horizon': 32, 'action_dim': 8,
            'action_units': ['rad'] * 7 + ['binary_0_close_1_open'],
            'action_convention': 'absolute'}


def wire():
    return make_request(observation(), request_id='test-1', return_features=True, seed=17)


@pytest.mark.parametrize('encoding', ['png', 'jpeg'])
def test_observation_roundtrip_uses_frozen_camera_names_and_rgb(encoding):
    value = make_request(observation(), request_id='obs-1', seed=0, image_encoding=encoding)
    request = decode_request(strict_json_loads(json_bytes(value)))
    assert request.request_id == 'obs-1' and request.seed == 0
    assert request.observation['instruction'] == '抓起杯子'
    assert list(request.observation['images']) == ['cam_high', 'cam_front']
    assert all(image.mode == 'RGB' and image.size == (28, 28)
               for image in request.observation['images'].values())
    assert set(request.observation) == {'images', 'state', 'instruction'}


@pytest.mark.parametrize('field', ['action', 'actions', 'reward', 'next_obs', 'execute', 'checkpoint'])
def test_future_labels_and_execution_requests_cannot_enter_observation(field):
    value = wire(); value[field] = [0]
    with pytest.raises(ProtocolError, match='unknown fields'):
        decode_request(value)


@pytest.mark.parametrize('state', [[0.] * 7, [0.] * 9, [False] * 8,
                                  [float('nan')] * 8, [float('inf')] * 8,
                                  ['0'] * 8, [10**1000] * 8])
def test_state_rejects_wrong_dimensions_types_and_nonfinite_values(state):
    value = wire(); value['state'] = state
    with pytest.raises(ProtocolError):
        decode_request(value)


@pytest.mark.parametrize('key,value', [('schema_version', True), ('schema_version', 2),
    ('request_id', ''), ('request_id', 'a\nb'), ('request_id', '\ud800'),
    ('instruction', ''), ('instruction', 'a' * 4097), ('instruction', '\x00'),
    ('return_features', 1), ('seed', True), ('seed', -1), ('seed', 2**63), ('seed', 0.5)])
def test_invalid_request_metadata_is_rejected(key, value):
    request = wire(); request[key] = value
    with pytest.raises(ProtocolError):
        decode_request(request)


@pytest.mark.parametrize('text', ['{"state":NaN}', '{"x":1,"x":2}', '{', b'\xff'])
def test_json_refuses_nonstandard_numbers_duplicate_keys_and_invalid_bytes(text):
    with pytest.raises(ProtocolError):
        strict_json_loads(text)


def test_camera_aliases_are_not_guessed():
    value = wire()
    value['images']['cam_wrist'] = value['images'].pop('cam_front')
    with pytest.raises(ProtocolError, match='unknown fields'):
        decode_request(value)


@pytest.mark.parametrize('mode', ['L', 'RGBA'])
def test_source_image_must_already_have_three_rgb_channels(mode):
    output = io.BytesIO(); Image.new(mode, (8, 8)).save(output, format='PNG')
    value = {'encoding': 'png', 'data': base64.b64encode(output.getvalue()).decode()}
    with pytest.raises(ProtocolError, match='RGB'):
        decode_image(value)


def test_bad_image_encoding_corruption_and_oversized_header_are_rejected():
    good = encode_image(Image.new('RGB', (28, 28)))
    for value in ({**good, 'encoding': 'jpeg'}, {**good, 'data': 'not valid!'},
                  {**good, 'data': base64.b64encode(b'not an image').decode()},
                  {**good, 'ignored_extra': 1}):
        with pytest.raises(ProtocolError):
            decode_image(value)
    output = io.BytesIO(); Image.new('RGB', (2049, 1)).save(output, format='PNG')
    with pytest.raises(ProtocolError) as caught:
        decode_image({'encoding': 'png', 'data': base64.b64encode(output.getvalue()).decode()})
    assert caught.value.status == 413


def test_response_attaches_provenance_and_never_claims_execution():
    request = InferenceRequest('id', {}, True)
    value = make_response({'actions': [[0.] * 8 for _ in range(32)], 'z_rl': [1., 2.],
                           'timing_seconds': .02}, request, metadata())
    checked = validate_response(value, request_id='id', return_features=True)
    assert checked['model_origin'] == 'synthetic_fixture'
    assert checked['action_space'] == 'physical' and checked['robot_executed'] is False
    assert checked['z_rl'] == [1., 2.] and checked['timing_seconds'] == .02


@pytest.mark.parametrize('fault', ['nan_action', 'short_chunk', 'wide_action', 'no_features',
                                  'nan_features', 'negative_time', 'execution_claim'])
def test_backend_output_is_validated_before_becoming_a_response(fault):
    value = {'actions': [[0.] * 8 for _ in range(32)], 'z_rl': [1.]}
    if fault == 'nan_action': value['actions'][0][0] = float('nan')
    elif fault == 'short_chunk': value['actions'].pop()
    elif fault == 'wide_action': value['actions'][0].append(0.)
    elif fault == 'no_features': value.pop('z_rl')
    elif fault == 'nan_features': value['z_rl'] = [float('inf')]
    elif fault == 'negative_time': value['timing_seconds'] = -1
    elif fault == 'execution_claim': value['robot_executed'] = True
    with pytest.raises(ProtocolError):
        make_response(value, InferenceRequest('id', {}, True), metadata())


def test_features_are_only_returned_when_explicitly_requested():
    with pytest.raises(ProtocolError, match='unrequested'):
        make_response({'actions': [[0.] * 8] * 32, 'z_rl': [1.]},
                      InferenceRequest('id', {}), metadata())


@pytest.mark.parametrize('fault', ['request_id', 'robot_executed', 'data_contract_hash', 'schema_version'])
def test_client_rejects_wrong_identity_execution_claim_or_metadata(fault):
    value = make_response({'actions': [[0.] * 8] * 32}, InferenceRequest('id', {}), metadata())
    value[fault] = {'request_id': 'different', 'robot_executed': True,
                    'data_contract_hash': 'bad', 'schema_version': True}[fault]
    with pytest.raises(ProtocolError):
        validate_response(value, request_id='id')


def test_health_metadata_validates_dimensions_and_optional_finite_values():
    assert validate_metadata(metadata())['model_origin'] == 'synthetic_fixture'
    for key, value in (('action_dim', 26), ('action_horizon', True),
                       ('data_contract_hash', 'not a hash'), ('peak_bytes', float('nan'))):
        changed = metadata(); changed[key] = value
        with pytest.raises(ProtocolError):
            validate_metadata(changed)
