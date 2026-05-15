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

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from utils.utils import StatusCode


class RequestSizeLimitMiddleware:
    """Pure ASGI middleware that bounds the size of HTTP request bodies.

    Registered once on the FastAPI application, this middleware is
    endpoint-agnostic: every route - ``/v1/chat/completions``,
    ``/v1/completions``, ``/v1/embeddings``, ``/v1/models``,
    ``/v1/models/{name}/load``, ``/v1/models/{name}/unload``, ``/metrics``,
    ``/health/ready`` and any future addition - is protected without
    per-router wiring.

    Two stages cover every body framing permitted by RFC 9112 over
    HTTP/1.1 (the only protocol Uvicorn/h11 speaks):

      1. Reject up front when the declared ``Content-Length`` exceeds the
         limit. The header is parsed defensively (RFC 9112 §6.3): a
         non-integer value or a negative value is itself an unrecoverable
         framing error and is rejected with HTTP 400. We do not rely on
         the upstream ASGI server (h11, hypercorn, daphne) to have
         validated the header for us.
      2. Drain the body in the middleware, counting bytes as they arrive,
         so requests using ``Transfer-Encoding: chunked`` (which carry no
         ``Content-Length``) are also bounded. Buffering once here adds no
         extra memory: every OpenAI route handler reads the full body
         anyway (Pydantic JSON parsing is atomic), and we cap at the same
         limit. The buffered body is then handed to the app via a one-shot
         replay receive.

    All rejection paths funnel through :meth:`_send_error_response`, which
    uses the same :class:`~fastapi.responses.JSONResponse` machinery and
    OpenAI-style error envelope as :class:`APIRestrictionMiddleware`. The
    whole frontend therefore speaks one consistent error shape for every
    middleware-level rejection.

    Oversized requests are rejected with HTTP 413 Content Too Large *before*
    FastAPI deserializes the body into Python objects, preventing the
    ~8x JSON-to-Python memory amplification that lets a single oversized
    request OOM-kill the frontend.
    """

    def __init__(self, app: ASGIApp, http_max_input_size: int) -> None:
        # Mirror core Triton's CLI validation: a non-positive limit would
        # silently disable the protection. argparse already enforces this
        # for --http-max-input-size, but programmatic construction also
        # needs to fail loudly.
        if http_max_input_size <= 0:
            raise ValueError(
                f"http_max_input_size must be greater than 0, got "
                f"{http_max_input_size}"
            )
        self.app = app
        self.http_max_input_size = http_max_input_size

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Stage 1: reject before reading any body bytes.
        # RFC 9112 §6.3: Content-Length must be a non-negative integer;
        # anything else is an unrecoverable framing error. We validate
        # defensively rather than trusting the upstream ASGI server.
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    await self._send_error_response(
                        scope,
                        receive,
                        send,
                        StatusCode.CLIENT_ERROR,
                        "invalid_content_length",
                        "Invalid Content-Length header: not an integer.",
                    )
                    return
                if declared < 0:
                    await self._send_error_response(
                        scope,
                        receive,
                        send,
                        StatusCode.CLIENT_ERROR,
                        "invalid_content_length",
                        "Invalid Content-Length header: must be non-negative.",
                    )
                    return
                if declared > self.http_max_input_size:
                    await self._send_content_too_large_response(
                        scope, receive, send
                    )
                    return
                break

        # Stage 2: drain the body, enforcing the cap as bytes arrive so
        # Transfer-Encoding: chunked (no Content-Length) is also bounded.
        body_chunks: list[bytes] = []
        received: int = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # http.disconnect: client gave up before sending the body.
                # We never invoked the app, so there is no state to clean
                # up - just drop the request.
                return
            received += len(message["body"])
            if received > self.http_max_input_size:
                await self._send_content_too_large_response(
                    scope, receive, send
                )
                return
            body_chunks.append(message["body"])
            if not message.get("more_body", False):
                break

        # Hand the buffered body to the app via a one-shot replay receive.
        body: bytes = b"".join(body_chunks)
        body_delivered: bool = False

        async def replay_receive() -> Message:
            nonlocal body_delivered
            if not body_delivered:
                body_delivered = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)

    async def _send_content_too_large_response(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Emit the canonical HTTP 413 Content Too Large response.

        The message includes the configured limit so operators can debug
        the rejection without correlating logs.
        """
        await self._send_error_response(
            scope,
            receive,
            send,
            StatusCode.CONTENT_TOO_LARGE,
            "content_too_large",
            (
                f"Request content size exceeds the maximum allowed input "
                f"size of {self.http_max_input_size} bytes. Use "
                f"--http-max-input-size to increase the limit."
            ),
        )

    @staticmethod
    async def _send_error_response(
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: StatusCode,
        code: str,
        message: str,
    ) -> None:
        """Emit a middleware-level error using the OpenAI-style envelope.

        Mirrors the rejection style used by
        :class:`APIRestrictionMiddleware` - a :class:`JSONResponse`
        carrying ``{"error": {"message": ..., "type": ..., "code": ...}}``
        - so the whole frontend rejects middleware-level errors with one
        consistent shape on the wire.
        """
        response = JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": code,
                }
            },
        )
        await response(scope, receive, send)
