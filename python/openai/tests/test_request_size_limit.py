#!/usr/bin/env python3

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

"""Unit tests for the request body size limit middleware.

These tests do not require a backend model to be loaded — they construct a
minimal FastAPI app that mounts the same middleware used by the OpenAI
frontend and exercise it via Starlette's TestClient.
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Make the openai_frontend package importable regardless of where pytest is
# launched from.
sys.path.append(
    os.path.join(Path(__file__).resolve().parent, "..", "openai_frontend")
)

from frontend.fastapi.middleware.request_size_limit import (  # noqa: E402
    DEFAULT_MAX_INPUT_SIZE,
    RequestSizeLimitMiddleware,
)


def _build_app(max_input_size: int) -> FastAPI:
    """Build a minimal FastAPI app with the size limit middleware mounted."""
    app = FastAPI()

    @app.post("/echo")
    async def echo(payload: dict) -> dict:
        return {"received_keys": list(payload.keys())}

    @app.get("/ping")
    async def ping() -> dict:
        return {"status": "ok"}

    app.add_middleware(
        RequestSizeLimitMiddleware, max_input_size=max_input_size
    )
    return app


class TestRequestSizeLimitMiddleware:
    SMALL_LIMIT = 1024  # 1 KiB, easy to exceed without large allocations.

    @pytest.fixture(scope="class")
    def client(self) -> TestClient:
        app = _build_app(max_input_size=self.SMALL_LIMIT)
        with TestClient(app) as test_client:
            yield test_client

    def test_request_within_limit_succeeds(self, client: TestClient):
        body = {"prompt": "x" * 100}
        response = client.post("/echo", json=body)
        assert response.status_code == 200
        assert response.json() == {"received_keys": ["prompt"]}

    def test_oversized_request_rejected_via_content_length(
        self, client: TestClient
    ):
        # Construct a JSON payload comfortably larger than the configured limit.
        body = {"prompt": "x" * (self.SMALL_LIMIT * 2)}

        response = client.post("/echo", json=body)
        assert response.status_code == 413, response.text

        payload = response.json()
        assert "error" in payload
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["code"] == "request_too_large"
        assert "exceeds the maximum allowed input size" in payload["error"][
            "message"
        ]
        assert "--openai-max-input-size" in payload["error"]["message"]

    def test_oversized_raw_body_rejected_via_content_length(
        self, client: TestClient
    ):
        raw_body = b"a" * (self.SMALL_LIMIT * 2)
        response = client.post(
            "/echo",
            content=raw_body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413, response.text

    def test_request_exactly_at_limit_succeeds(self, client: TestClient):
        # Build a JSON payload whose serialized size is the limit exactly.
        # We pad the value so that the resulting body length matches
        # SMALL_LIMIT exactly. The shape is {"prompt": "<padded>"}; the
        # surrounding JSON syntax accounts for 14 characters.
        envelope_overhead = len('{"prompt": ""}')
        padding_length = self.SMALL_LIMIT - envelope_overhead
        assert padding_length > 0
        body = {"prompt": "x" * padding_length}

        # Sanity check: ensure the encoded body is actually exactly the limit.
        import json as _json

        encoded = _json.dumps(body, separators=(", ", ": ")).encode("utf-8")
        assert len(encoded) == self.SMALL_LIMIT

        response = client.post(
            "/echo",
            content=encoded,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 200, response.text

    def test_request_one_byte_over_limit_rejected(self, client: TestClient):
        # Mirrors the "1 byte above the nominal core Triton limit" scenario
        # described in the issue, scaled down to keep the test fast.
        envelope_overhead = len('{"prompt": ""}')
        padding_length = self.SMALL_LIMIT - envelope_overhead + 1
        body = {"prompt": "x" * padding_length}

        import json as _json

        encoded = _json.dumps(body, separators=(", ", ": ")).encode("utf-8")
        assert len(encoded) == self.SMALL_LIMIT + 1

        response = client.post(
            "/echo",
            content=encoded,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413, response.text

    def test_get_request_unaffected(self, client: TestClient):
        # Ensure non-body requests are passed through unchanged.
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_oversized_streaming_body_rejected(self, client: TestClient):
        # Generator-based bodies exercise the streaming path that does not
        # rely on the Content-Length header check. The body is well over
        # the limit so the middleware must short-circuit during receive().
        def big_chunks():
            yield b'{"prompt": "'
            for _ in range(8):
                yield b"x" * 512
            yield b'"}'

        response = client.post(
            "/echo",
            content=big_chunks(),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413, response.text
        payload = response.json()
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["code"] == "request_too_large"
        assert "--openai-max-input-size" in payload["error"]["message"]

    def test_streaming_body_within_limit_succeeds(self, client: TestClient):
        # Sanity check: streaming bodies under the limit are not affected.
        def small_chunks():
            yield b'{"prompt": "hi"}'

        response = client.post(
            "/echo",
            content=small_chunks(),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 200, response.text
        assert response.json() == {"received_keys": ["prompt"]}


class TestRequestSizeLimitConfiguration:
    def test_invalid_zero_limit_raises(self):
        app = FastAPI()
        with pytest.raises(ValueError):
            RequestSizeLimitMiddleware(app, max_input_size=0)

    def test_invalid_negative_limit_raises(self):
        app = FastAPI()
        with pytest.raises(ValueError):
            RequestSizeLimitMiddleware(app, max_input_size=-1)

    def test_default_limit_matches_core_triton(self):
        # Core Triton's HTTP_DEFAULT_MAX_INPUT_SIZE is 64 MiB (1 << 26).
        assert DEFAULT_MAX_INPUT_SIZE == 1 << 26
