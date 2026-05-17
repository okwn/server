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


import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.append(
    os.path.join(str(Path(__file__).resolve().parent.parent), "openai_frontend")
)

from frontend.fastapi.middleware.request_size import RequestSizeLimitMiddleware
from frontend.fastapi_frontend import FastApiFrontend
from utils.utils import HTTP_DEFAULT_MAX_INPUT_SIZE

_LIMIT = HTTP_DEFAULT_MAX_INPUT_SIZE  # 64 MiB
_OVERFLOW = 16

# All POST endpoints; used for rejection tests (middleware short-circuits
# before the mock-engine-backed handler is invoked).
_ENDPOINTS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models/test-model/load",
    "/v1/models/test-model/unload",
)

# Subset that parses the body via Pydantic; used for the at-limit acceptance
# test where the handler fails validation cleanly on raw bytes instead of
# invoking the mock engine.
_BODY_PARSING_ENDPOINTS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
)


@pytest.fixture(scope="module")
def client():
    """FastApiFrontend wired with a MagicMock engine (no Triton/GPU needed)."""
    frontend = FastApiFrontend(engine=MagicMock(), http_max_input_size=_LIMIT)
    return TestClient(frontend.app)


def _assert_content_too_large(response) -> None:
    assert response.status_code == 413
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "content_too_large"
    assert str(_LIMIT) in error["message"]
    assert "--http-max-input-size" in error["message"]


class TestRequestSizeLimitMiddleware:
    """End-to-end coverage through FastApiFrontend with a TestClient."""

    @pytest.mark.parametrize("endpoint", _BODY_PARSING_ENDPOINTS)
    def test_body_at_limit_is_not_rejected(self, client, endpoint):
        response = client.post(endpoint, content=b"x" * _LIMIT)
        assert response.status_code != 413

    @pytest.mark.parametrize("endpoint", _ENDPOINTS)
    def test_body_over_limit_is_rejected(self, client, endpoint):
        response = client.post(endpoint, content=b"x" * (_LIMIT + _OVERFLOW))
        _assert_content_too_large(response)

    @pytest.mark.parametrize("endpoint", _ENDPOINTS)
    def test_chunked_body_over_limit_is_rejected(self, client, endpoint):
        # httpx switches to chunked transfer when content is an Iterable[bytes].
        def chunks():
            yield b"x" * _LIMIT
            yield b"x" * _OVERFLOW

        response = client.post(endpoint, content=chunks())
        _assert_content_too_large(response)

    def test_get_without_body_is_unaffected(self, client):
        response = client.get(_ENDPOINTS[0])
        assert response.status_code == 405


class TestContentLengthValidation:
    """Stage 1 rejects malformed Content-Length with 400 (direct ASGI: httpx rewrites the header)."""

    def _run_with_content_length(self, raw_value: bytes) -> tuple[int, dict]:
        captured: dict = {"status": None, "body": b""}

        async def app(scope, receive, send):
            raise AssertionError("app must not be reached for invalid Content-Length")

        async def receive():
            raise AssertionError("receive() must not be called when Stage 1 rejects")

        async def send(message):
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
            elif message["type"] == "http.response.body":
                captured["body"] += message.get("body", b"")

        middleware = RequestSizeLimitMiddleware(app=app, http_max_input_size=_LIMIT)
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

    @pytest.mark.parametrize("raw", [b"not-a-number", b""])
    def test_non_integer_content_length_rejected_with_400(self, raw):
        status, body = self._run_with_content_length(raw)
        self._assert_invalid_content_length(status, body)
        assert "not an integer" in body["error"]["message"]

    @pytest.mark.parametrize("raw", [b"-1", b"-1024"])
    def test_negative_content_length_rejected_with_400(self, raw):
        status, body = self._run_with_content_length(raw)
        self._assert_invalid_content_length(status, body)
        assert "non-negative" in body["error"]["message"]


class TestStreamingDisconnectListener:
    """
    After the body is delivered, replay_receive must delegate to the original
    receive() so streaming responses (SSE chat/completions) can wait for the
    real client disconnect instead of being aborted by a premature
    http.disconnect from the middleware.
    """

    def test_subsequent_receive_delegates_to_original(self):
        seen: list[str] = []
        upstream_queue = [
            {"type": "http.request", "body": b"hello", "more_body": False},
            {"type": "http.disconnect"},  # arrives later, simulating client hang-up
        ]

        async def receive():
            assert upstream_queue, "middleware exhausted original receive() queue"
            return upstream_queue.pop(0)

        async def send(_message):
            pass

        async def app(scope, app_receive, app_send):
            seen.append((await app_receive())["type"])
            # Second call must delegate to original receive() and return the
            # queued disconnect, not a synthetic disconnect from the closure.
            seen.append((await app_receive())["type"])

        middleware = RequestSizeLimitMiddleware(app=app, http_max_input_size=_LIMIT)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-length", b"5")],
        }
        asyncio.run(middleware(scope, receive, send))

        assert seen == ["http.request", "http.disconnect"]
        assert upstream_queue == [], "original receive() was not delegated to"


class TestStreamTermination:
    """Stage 2 must abort silently (no app invocation, no response) on disconnect or unexpected message types."""

    def _run(self, messages: list) -> tuple[bool, bool]:
        app_invoked = False
        response_sent = False

        async def app(scope, receive, send):
            nonlocal app_invoked
            app_invoked = True

        queue = list(messages)

        async def receive():
            assert queue, "middleware called receive() more times than expected"
            return queue.pop(0)

        async def send(message):
            nonlocal response_sent
            response_sent = True

        middleware = RequestSizeLimitMiddleware(app=app, http_max_input_size=_LIMIT)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        }
        asyncio.run(middleware(scope, receive, send))
        return app_invoked, response_sent

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([{"type": "http.disconnect"}], id="disconnect_before_body"),
            pytest.param(
                [
                    {"type": "http.request", "body": b"x" * 100, "more_body": True},
                    {"type": "http.disconnect"},
                ],
                id="disconnect_mid_body",
            ),
            pytest.param([{"type": "http.unexpected"}], id="unknown_message_type"),
        ],
    )
    def test_terminates_silently(self, messages):
        app_invoked, response_sent = self._run(messages)
        assert not app_invoked
        assert not response_sent
