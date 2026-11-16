"""Serial HTTP transport for an explicitly supplied inference backend.

No backend is created by this module. A caller must load a concrete artifact
and supply metadata()/infer(). Health means that the service is reachable, not
that a full model, target GPU, or robot deployment has passed acceptance.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import math
import re
import socket
import uuid

from .protocol import (MAX_BODY_BYTES, SCHEMA_VERSION, STATE_DIM, ProtocolError, decode_request,
                       json_bytes, make_response, strict_json_loads, validate_metadata)


LOGGER = logging.getLogger(__name__)

IDENTITY_METADATA_KEYS = (
    'backend', 'model_origin', 'artifact_id', 'training_step', 'data_contract_hash',
    'action_horizon', 'action_dim', 'action_names', 'action_units', 'action_convention',
    'coordinate_frame', 'camera_mapping', 'state_names', 'state_units', 'feature_dim',
    'device', 'inference_steps', 'image_size', 'max_length', 'vision_attention',
    'wall_attention', 'runtime_model_sources_sha256',
)


class InferenceHTTPServer(HTTPServer):
    """One active request at a time, so backend RNG/model access stays serial."""
    allow_reuse_address = True

    def __init__(self, address, backend, *, request_timeout=10.0, max_body_bytes=MAX_BODY_BYTES):
        if (not isinstance(request_timeout, (int, float)) or not math.isfinite(request_timeout)
                or request_timeout <= 0):
            raise ValueError('request_timeout must be a positive finite number')
        if type(max_body_bytes) is not int or not 0 < max_body_bytes <= MAX_BODY_BYTES:
            raise ValueError('max_body_bytes must be within the protocol limit')
        if not callable(getattr(backend, 'infer', None)) or not callable(getattr(backend, 'metadata', None)):
            raise ValueError('an explicit backend with metadata() and infer() is required')
        self.backend = backend
        self.backend_metadata = validate_metadata(backend.metadata())
        self.instance_id = uuid.uuid4().hex
        self.request_timeout = float(request_timeout)
        self.max_body_bytes = max_body_bytes
        super().__init__(address, _Handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(self.request_timeout)
        return connection, address


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'WallRLTInference/1'
    sys_version = ''

    def log_message(self, format, *args):
        # No observation bodies, instructions, state vectors, or image bytes.
        LOGGER.debug('inference HTTP %s', args[1] if len(args) > 1 else 'request')

    def _reply(self, status, payload):
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        if self.command != 'HEAD':
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass

    def _error(self, error, request_id=None):
        self._reply(error.status, {'schema_version': SCHEMA_VERSION, 'status': 'error',
            'request_id': request_id, 'error': {'code': error.code, 'message': str(error)},
            'robot_executed': False})

    def send_error(self, code, message=None, explain=None):
        # BaseHTTPRequestHandler also uses this for malformed headers/methods.
        self._error(ProtocolError('HTTP request rejected', status=code, code='http_request_rejected'))

    def do_GET(self):
        if self.path != '/health':
            self._error(ProtocolError('unknown endpoint', code='not_found', status=404))
            return
        try:
            metadata = validate_metadata(self.server.backend.metadata())
            if any(metadata.get(key) != self.server.backend_metadata.get(key)
                   for key in IDENTITY_METADATA_KEYS):
                raise ValueError('backend identity changed after service startup')
        except Exception:
            LOGGER.exception('inference backend health metadata failed')
            self._error(ProtocolError('inference backend failed', status=500,
                                     code='backend_failure'))
            return
        self._reply(200, {'schema_version': SCHEMA_VERSION, 'status': 'ready',
            'server_instance_id': self.server.instance_id,
            'scope': 'inference transport; no robot execution',
            'model_acceptance': 'not_asserted_by_health_check',
            'metadata': metadata, 'robot_executed': False})

    def do_POST(self):
        request = None
        try:
            if self.path != '/infer':
                raise ProtocolError('unknown endpoint', code='not_found', status=404)
            if self.headers.get('Transfer-Encoding') is not None:
                raise ProtocolError('Transfer-Encoding is unsupported; send Content-Length')
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or not re.fullmatch(r'[0-9]{1,10}', lengths[0]):
                raise ProtocolError('exactly one numeric Content-Length is required', status=411,
                                    code='content_length_required')
            length = int(lengths[0])
            if not 0 < length <= self.server.max_body_bytes:
                raise ProtocolError('request exceeds the body byte limit', status=413,
                                    code='request_too_large')
            if self.headers.get_content_type() != 'application/json':
                raise ProtocolError('Content-Type must be application/json', status=415,
                                    code='unsupported_content_type')
            charset = self.headers.get_content_charset()
            if charset is not None and charset.lower() != 'utf-8':
                raise ProtocolError('JSON encoding must be UTF-8', status=415,
                                    code='unsupported_content_type')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ProtocolError('request body is incomplete')
            request = decode_request(strict_json_loads(raw),
                                     state_dim=self.server.backend_metadata.get('state_dim', STATE_DIM))
        except socket.timeout:
            self._error(ProtocolError('request body timed out', status=408, code='request_timeout'))
            return
        except ProtocolError as error:
            self._error(error)
            return
        try:
            extra = {} if request.rtc is None else {'rtc': request.rtc}
            result = self.server.backend.infer(request.observation,
                return_features=request.return_features, seed=request.seed, **extra)
            response = make_response(result, request, self.server.backend_metadata)
        except Exception:
            LOGGER.exception('inference backend failed; no result was returned')
            self._error(ProtocolError('inference backend failed', status=500,
                                     code='backend_failure'), request.request_id)
            return
        self._reply(200, response)


def create_server(backend, *, host='127.0.0.1', port=8080, request_timeout=10.0,
                  max_body_bytes=MAX_BODY_BYTES):
    """Create a local server; explicit host overrides are caller decisions."""
    if not isinstance(host, str) or not host or type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('invalid host or port')
    return InferenceHTTPServer((host, port), backend, request_timeout=request_timeout,
                               max_body_bytes=max_body_bytes)
