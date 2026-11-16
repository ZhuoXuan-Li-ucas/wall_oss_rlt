"""Real localhost requests exercise transport with an explicitly synthetic backend."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import http.client
import json
import socket
import threading
import time

import pytest
from PIL import Image

from wall_rlt.inference.client import InferenceClient, RemoteInferenceError
from wall_rlt.inference.protocol import CAMERA_KEYS, json_bytes, make_request
from wall_rlt.inference.server import create_server


class SyntheticFixtureBackend:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.maximum_active = 0
        self.fault = None

    def metadata(self):
        return {'backend': 'test-fixture', 'model_origin': 'synthetic_fixture',
                'data_contract_hash': 'b' * 64, 'action_horizon': 32, 'action_dim': 8,
                'action_units': ['rad'] * 7 + ['binary_0_close_1_open'],
                'action_convention': 'absolute'}

    def infer(self, observation, *, return_features=False, seed=None):
        self.active += 1
        self.maximum_active = max(self.active, self.maximum_active)
        try:
            self.calls.append((observation, return_features, seed))
            assert set(observation['images']) == set(CAMERA_KEYS)
            assert all(isinstance(image, Image.Image) and image.mode == 'RGB'
                       for image in observation['images'].values())
            time.sleep(.01)
            if self.fault == 'exception':
                raise RuntimeError('private checkpoint /private/path/secret.pt')
            actions = [[float(index)] * 8 for index in range(32)]
            if self.fault == 'nonfinite':
                actions[0][0] = float('nan')
            result = {'actions': actions, 'timing_seconds': .01}
            if return_features:
                result['z_rl'] = [float(seed or 0), 1.]
            return result
        finally:
            self.active -= 1


def observation():
    return {'instruction': 'pick the object', 'state': [0.] * 8,
            'images': {key: Image.new('RGB', (28, 28), (1, 2, 3)) for key in CAMERA_KEYS}}


@contextmanager
def running_server(**kwargs):
    backend = SyntheticFixtureBackend()
    server = create_server(backend, port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    thread.start()
    try:
        yield server, backend, InferenceClient(f'http://127.0.0.1:{server.server_port}', timeout=2.)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
        assert not thread.is_alive()


def raw_request(server, *, body=b'{}', path='/infer', content_type='application/json', headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2.)
    try:
        connection.request('POST', path, body=body,
                           headers={'Content-Type': content_type, **(headers or {})})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_real_socket_health_and_prediction_roundtrip():
    with running_server() as (server, backend, client):
        assert server.server_address[0] == '127.0.0.1'
        health = client.health()
        assert health['model_acceptance'] == 'not_asserted_by_health_check'
        assert health['metadata']['model_origin'] == 'synthetic_fixture'
        result = client.infer(observation(), request_id='roundtrip', return_features=True, seed=19)
        assert result['request_id'] == 'roundtrip'
        assert len(result['actions']) == 32 and len(result['actions'][0]) == 8
        assert result['z_rl'] == [19., 1.] and result['robot_executed'] is False
        assert backend.calls[0][2] == 19
        result = client.infer(observation(), request_id='without-features')
        assert 'z_rl' not in result


@pytest.mark.parametrize('fault', ['exception', 'nonfinite'])
def test_backend_failure_cannot_leak_traceback_or_nonfinite_predictions(fault):
    with running_server() as (_, backend, client):
        backend.fault = fault
        with pytest.raises(RemoteInferenceError) as caught:
            client.infer(observation(), request_id='bad-model')
        assert caught.value.status == 500 and caught.value.code == 'backend_failure'
        assert caught.value.request_id == 'bad-model'
        assert 'private' not in str(caught.value) and 'Traceback' not in str(caught.value)


def test_unknown_future_label_is_rejected_before_backend_runs():
    with running_server() as (server, backend, _):
        value = make_request(observation(), request_id='labels')
        value['action'] = [0] * 8
        status, result = raw_request(server, body=json_bytes(value))
        assert status == 400 and result['error']['code'] == 'invalid_request'
        assert not backend.calls and result['robot_executed'] is False


@pytest.mark.parametrize('body', [b'{', b'{"x":NaN}', b'{"schema_version":1,"schema_version":1}', b'\xff'])
def test_http_malformed_json_is_structured_and_never_calls_backend(body):
    with running_server() as (server, backend, _):
        status, result = raw_request(server, body=body)
        assert status == 400 and result['status'] == 'error' and not backend.calls


def test_http_headers_and_routes_are_bounded():
    with running_server(max_body_bytes=128) as (server, backend, _):
        for kwargs, expected in (({'body': b'x' * 129}, 413),
                                  ({'content_type': 'text/plain'}, 415),
                                  ({'content_type': 'application/json; charset=latin1'}, 415),
                                  ({'path': '/execute'}, 404),
                                  ({'headers': {'Transfer-Encoding': 'chunked'}}, 400)):
            status, result = raw_request(server, **kwargs)
            assert status == expected and result['status'] == 'error'
        assert not backend.calls


@pytest.mark.parametrize('length_count', [0, 2])
def test_missing_or_duplicate_content_length_is_rejected(length_count):
    with running_server() as (server, backend, _):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2.)
        try:
            connection.putrequest('POST', '/infer')
            connection.putheader('Content-Type', 'application/json')
            for _ in range(length_count):
                connection.putheader('Content-Length', '2')
            connection.endheaders(b'{}')
            response = connection.getresponse()
            result = json.loads(response.read())
            assert response.status == 411
            assert result['error']['code'] == 'content_length_required'
            assert not backend.calls
        finally:
            connection.close()


def test_incomplete_request_times_out_and_releases_serial_service():
    with running_server(request_timeout=.1) as (server, backend, client):
        with socket.create_connection(('127.0.0.1', server.server_port), timeout=2.) as connection:
            connection.sendall(b'POST /infer HTTP/1.1\r\nHost: localhost\r\n'
                               b'Content-Type: application/json\r\nContent-Length: 20\r\n\r\n{')
            chunks = []
            while True:
                chunk = connection.recv(4096)
                if not chunk: break
                chunks.append(chunk)
            response = b''.join(chunks)
        assert b'408' in response and b'request_timeout' in response
        assert not backend.calls and client.health()['status'] == 'ready'


def test_concurrent_clients_do_not_run_backend_in_parallel():
    with running_server() as (_, backend, client):
        def run(index):
            # Independent HTTP clients emulate separate requesting processes.
            return InferenceClient(client.endpoint, timeout=3).infer(
                observation(), request_id=f'request-{index}', seed=index)
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(run, range(3)))
        assert {row['request_id'] for row in results} == {'request-0', 'request-1', 'request-2'}
        assert backend.maximum_active == 1 and len(backend.calls) == 3


def test_client_limit_and_unreachable_server_are_explicit_errors():
    with running_server() as (_, _, client):
        tiny = InferenceClient(client.endpoint, max_response_bytes=16)
        with pytest.raises(RemoteInferenceError) as caught:
            tiny.health()
        assert caught.value.code == 'response_too_large'
        endpoint = client.endpoint
    with pytest.raises(RemoteInferenceError, match='transport failed'):
        InferenceClient(endpoint, timeout=.2).health()


def test_server_cannot_start_with_an_implicit_or_invalid_backend():
    with pytest.raises(ValueError, match='explicit backend'):
        create_server(None, port=0)
    backend = SyntheticFixtureBackend()
    backend.metadata = lambda: {'model_origin': 'synthetic_fixture'}
    with pytest.raises(ValueError, match='metadata'):
        create_server(backend, port=0)


def test_health_refreshes_diagnostics_but_keeps_identity_for_this_instance():
    with running_server() as (_, backend, client):
        original = backend.metadata
        backend.metadata = lambda: {**original(), 'completed_requests': len(backend.calls),
            'peak_allocated_bytes': len(backend.calls) * 1024}
        before = client.health()
        client.infer(observation(), request_id='diagnostics')
        after = client.health()
        assert before['server_instance_id'] == after['server_instance_id']
        assert len(before['server_instance_id']) == 32
        assert before['metadata']['completed_requests'] == 0
        assert after['metadata']['completed_requests'] == 1
        assert after['metadata']['peak_allocated_bytes'] == 1024
    with running_server() as (_, _, other_client):
        assert other_client.health()['server_instance_id'] != before['server_instance_id']


def test_health_rejects_changed_identity_and_nonfinite_diagnostics():
    with running_server() as (_, backend, client):
        original = backend.metadata
        for extra in ({'data_contract_hash': 'a' * 64}, {'peak_allocated_bytes': float('nan')}):
            backend.metadata = lambda: {**original(), **extra}
            with pytest.raises(RemoteInferenceError) as caught:
                client.health()
            assert caught.value.code == 'backend_failure'
