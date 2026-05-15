# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Unit tests for ``RequestSizeLimitMiddleware``.

These tests verify the properties that determine correctness of the OOM
fix:

  * a body within the limit is accepted (no false positives);
  * a body exceeding the limit via honest ``Content-Length`` is rejected
    before the handler runs (Stage 1, fast path);
  * a body exceeding the limit via ``Transfer-Encoding: chunked`` (no
    ``Content-Length``) is rejected by the byte counter (Stage 2, slow
    path);
  * a malformed or negative ``Content-Length`` is rejected with HTTP 400
    rather than crashing the worker (RFC 9112 §6.3).

They also assert that the middleware is endpoint-agnostic: a single
registration on the ``FastAPI`` app protects every route - chat,
completions, embeddings, model load/unload - without per-router wiring.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

sys.path.append(
    os.path.join(str(Path(__file__).resolve().parent.parent), "openai_frontend")
)

from frontend.fastapi.middleware.request_size import (  # noqa: E402
    RequestSizeLimitMiddleware,
)


# Endpoints exercised by the OpenAI frontend that accept a request body.
# Mirroring real route paths makes it explicit that the middleware applies
# globally, not just to ``/v1/completions``.
_BODY_ENDPOINTS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models/test-model/load",
    "/v1/models/test-model/unload",
)


def _build_app(http_max_input_size: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        RequestSizeLimitMiddleware, http_max_input_size=http_max_input_size
    )

    async def _echo(request: Request):
        body = await request.body()
        return {"length": len(body)}

    for endpoint in _BODY_ENDPOINTS:
        app.add_api_route(endpoint, _echo, methods=["POST"])

    return app


class TestRequestSizeLimitMiddleware:
    LIMIT = 128

    @pytest.mark.parametrize("endpoint", _BODY_ENDPOINTS)
    def test_body_at_limit_is_accepted_on_every_endpoint(self, endpoint):
        client = TestClient(_build_app(self.LIMIT))
        response = client.post(endpoint, content=b"x" * self.LIMIT)
        assert response.status_code == 200
        assert response.json() == {"length": self.LIMIT}

    def _assert_content_too_large(self, response):
        # Both stages must reject through the same OpenAI-style error
        # envelope as ``APIRestrictionMiddleware`` so clients can rely on
        # a single error shape for every middleware-level rejection.
        assert response.status_code == 413
        body = response.json()
        assert set(body) == {"error"}
        error = body["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "content_too_large"
        # The message must name the configured limit so operators can
        # diagnose the rejection without correlating logs.
        assert str(self.LIMIT) in error["message"]
        assert "--http-max-input-size" in error["message"]

    @pytest.mark.parametrize("endpoint", _BODY_ENDPOINTS)
    def test_body_one_byte_over_limit_is_rejected_on_every_endpoint(
        self, endpoint
    ):
        # Stage 1: declared Content-Length exceeds the limit by one byte.
        # The rejection is a pure header check; no body bytes are read.
        client = TestClient(_build_app(self.LIMIT))
        response = client.post(endpoint, content=b"x" * (self.LIMIT + 1))
        self._assert_content_too_large(response)

    @pytest.mark.parametrize("endpoint", _BODY_ENDPOINTS)
    def test_chunked_body_over_limit_is_rejected_on_every_endpoint(
        self, endpoint
    ):
        # Stage 2: Transfer-Encoding: chunked carries no Content-Length, so
        # Stage 1 is bypassed and the byte counter must catch it. httpx
        # sends chunked when ``content`` is an Iterable[bytes].
        def chunks():
            yield b"x" * 80
            yield b"x" * 80  # cumulative 160 > 128

        client = TestClient(_build_app(self.LIMIT))
        response = client.post(endpoint, content=chunks())
        self._assert_content_too_large(response)

    def test_get_without_body_is_unaffected(self):
        # The middleware must not add overhead or false positives to
        # bodyless methods. The endpoints above are POST-only, so a GET
        # produces the framework's normal 405 - nothing from our middleware.
        client = TestClient(_build_app(self.LIMIT))
        response = client.get(_BODY_ENDPOINTS[0])
        assert response.status_code == 405

    @pytest.mark.parametrize("invalid", [0, -1, -1024])
    def test_constructor_rejects_non_positive_limit(self, invalid):
        # Mirrors core Triton's CLI validation in
        # server/src/command_line_parser.cc: --http-max-input-size must be
        # greater than 0. Validating in the middleware too prevents
        # programmatic misuse from silently disabling the limit.
        with pytest.raises(ValueError, match="must be greater than 0"):
            RequestSizeLimitMiddleware(app=None, http_max_input_size=invalid)


class TestContentLengthValidation:
    """Stage 1 header validation. RFC 9112 §6.3 requires Content-Length to
    be a non-negative integer; anything else is an unrecoverable framing
    error and must be rejected with HTTP 400, not silently bubbled up as
    an unhandled ``ValueError`` that becomes an HTTP 500.

    These cases use direct ASGI invocation because httpx's ``TestClient``
    validates and rewrites Content-Length, which would prevent us from
    sending the malformed values we need to exercise.
    """

    LIMIT = 128

    def _run_with_content_length(
        self, raw_value: bytes
    ) -> tuple[int, dict]:
        """Drive the middleware with one raw Content-Length value and
        return ``(status, parsed_body)``. The wrapped app must never be
        invoked - all validation must reject before Stage 2.
        """
        captured: dict = {"status": None, "body": b""}

        async def app(scope, receive, send):
            raise AssertionError(
                "Wrapped app must not be reached for invalid Content-Length"
            )

        async def receive():
            # Stage 1 rejects without reading the body. If receive is
            # called, the middleware did the wrong thing.
            raise AssertionError(
                "receive() must not be called when Stage 1 rejects"
            )

        async def send(message):
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
            elif message["type"] == "http.response.body":
                captured["body"] += message.get("body", b"")

        middleware = RequestSizeLimitMiddleware(
            app=app, http_max_input_size=self.LIMIT
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-length", raw_value)],
        }

        asyncio.run(middleware(scope, receive, send))

        assert captured["status"] is not None, "no response was sent"
        return captured["status"], json.loads(captured["body"])

    def _assert_invalid_content_length(self, status: int, body: dict):
        assert status == 400
        assert set(body) == {"error"}
        error = body["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_content_length"

    def test_non_integer_content_length_rejected_with_400(self):
        status, body = self._run_with_content_length(b"not-a-number")
        self._assert_invalid_content_length(status, body)
        assert "not an integer" in body["error"]["message"]

    def test_empty_content_length_rejected_with_400(self):
        # ``int(b"")`` raises ``ValueError``; an empty value is also
        # invalid per RFC 9112 §6.3.
        status, body = self._run_with_content_length(b"")
        self._assert_invalid_content_length(status, body)
        assert "not an integer" in body["error"]["message"]

    @pytest.mark.parametrize("raw", [b"-1", b"-1024", b"-9999999"])
    def test_negative_content_length_rejected_with_400(self, raw):
        # ``int(b"-1")`` succeeds and returns -1; the framing error must
        # still be caught explicitly so this case becomes a 400 rather
        # than falling through Stage 1 and being read as a 0-byte body.
        status, body = self._run_with_content_length(raw)
        self._assert_invalid_content_length(status, body)
        assert "non-negative" in body["error"]["message"]
