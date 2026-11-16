"""Bounded standard-library client for prediction-only WALL-RLT HTTP serving."""
from __future__ import annotations

import math
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from .protocol import (MAX_BODY_BYTES, ProtocolError, json_bytes, make_request,
                       strict_json_loads, validate_metadata, validate_response)


class RemoteInferenceError(RuntimeError):
    def __init__(self, message, *, status=None, code='transport_error', request_id=None):
        super().__init__(message)
        self.status, self.code, self.request_id = status, code, request_id


class InferenceClient:
    def __init__(self, endpoint='http://127.0.0.1:8080', *, timeout=30.0,
                 max_response_bytes=MAX_BODY_BYTES):
        parts = urlsplit(endpoint)
        if (parts.scheme not in ('http', 'https') or not parts.hostname or
                parts.username or parts.password or parts.query or parts.fragment or
                parts.path not in ('', '/')):
            raise ValueError('endpoint must be an HTTP(S) origin without credentials or a path')
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        if type(max_response_bytes) is not int or not 0 < max_response_bytes <= MAX_BODY_BYTES:
            raise ValueError('invalid response byte limit')
        self.endpoint, self.timeout = endpoint.rstrip('/'), float(timeout)
        self.max_response_bytes = max_response_bytes
        # Local robot-host transport must not inherit an unrelated HTTP proxy.
        self.opener = build_opener(ProxyHandler({}))

    def _read(self, response):
        raw = response.read(self.max_response_bytes + 1)
        if len(raw) > self.max_response_bytes:
            raise RemoteInferenceError('response exceeds the byte limit', code='response_too_large')
        try:
            return strict_json_loads(raw)
        except ProtocolError as error:
            raise RemoteInferenceError(str(error), code='invalid_response') from None

    def _request(self, path, value=None):
        request = Request(self.endpoint + path,
                          data=None if value is None else json_bytes(value),
                          headers={'Accept': 'application/json', 'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                if response.headers.get_content_type() != 'application/json':
                    raise RemoteInferenceError('response must be application/json', code='invalid_response')
                return self._read(response)
        except HTTPError as error:
            with error:
                value = self._read(error)
            if not isinstance(value, dict) or value.get('status') != 'error' or not isinstance(value.get('error'), dict):
                raise RemoteInferenceError('server returned an invalid error response',
                                           status=error.code, code='invalid_response') from None
            details = value['error']
            message = details.get('message')
            code = details.get('code')
            if not isinstance(message, str) or not isinstance(code, str):
                raise RemoteInferenceError('server returned an invalid error response',
                                           status=error.code, code='invalid_response') from None
            raise RemoteInferenceError(message, status=error.code, code=code,
                                       request_id=value.get('request_id')) from None
        except (URLError, TimeoutError, ConnectionError, OSError) as error:
            raise RemoteInferenceError(f'inference transport failed: {type(error).__name__}') from None

    def health(self):
        value = self._request('/health')
        if (not isinstance(value, dict) or type(value.get('schema_version')) is not int or
                value['schema_version'] != 1 or value.get('status') != 'ready' or
                not isinstance(value.get('server_instance_id'), str) or
                not re.fullmatch(r'[0-9a-f]{32}', value['server_instance_id']) or
                value.get('model_acceptance') != 'not_asserted_by_health_check' or
                value.get('robot_executed') is not False):
            raise RemoteInferenceError('invalid health response', code='invalid_response')
        try:
            validate_metadata(value.get('metadata'))
        except ProtocolError as error:
            raise RemoteInferenceError(str(error), code='invalid_response') from None
        return value

    def infer(self, observation, *, request_id=None, return_features=False, seed=None,
              image_encoding='png', rtc=None):
        request_id = uuid.uuid4().hex if request_id is None else request_id
        value = make_request(observation, request_id=request_id,
                             return_features=return_features, seed=seed,
                             image_encoding=image_encoding, rtc=rtc)
        response = self._request('/infer', value)
        try:
            return validate_response(response, request_id=request_id,
                                     return_features=return_features)
        except ProtocolError as error:
            raise RemoteInferenceError(str(error), code='invalid_response',
                                       request_id=request_id) from None
