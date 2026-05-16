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

import logging

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from utils.utils import StatusCode, validate_positive_int

logger = logging.getLogger(__name__)


class _BodyTooLargeError(Exception):
    pass


class RequestSizeLimitMiddleware:
    """ASGI middleware that enforces a hard cap on HTTP request body size.

    Stage 1 — Content-Length present: reject before reading any body bytes.
    Stage 2 — Chunked/HTTP2/no Content-Length: count bytes as they arrive
               and reject as soon as the running total exceeds the limit.
    """

    def __init__(self, app: ASGIApp, http_max_input_size: int) -> None:
        self.app = app
        self.http_max_input_size = validate_positive_int(http_max_input_size)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Stage 1: reject on Content-Length before reading any body bytes.
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    content_length = int(value)
                except ValueError:
                    await self._send_error(
                        scope, send,
                        status_code=StatusCode.CLIENT_ERROR,
                        code="invalid_content_length",
                        message="Invalid Content-Length header: not an integer.",
                    )
                    return
                if content_length < 0:
                    await self._send_error(
                        scope, send,
                        status_code=StatusCode.CLIENT_ERROR,
                        code="invalid_content_length",
                        message="Invalid Content-Length header: must be non-negative.",
                    )
                    return
                if content_length > self.http_max_input_size:
                    await self._send_error(
                        scope, send,
                        status_code=StatusCode.CONTENT_TOO_LARGE,
                        code="content_too_large",
                        message=(
                            f"Request content size exceeds the maximum allowed "
                            f"input size of {self.http_max_input_size} bytes. "
                            f"Use --http-max-input-size to increase the limit."
                        ),
                    )
                    return
                break

        # Stage 2: count streaming bytes for chunked / no Content-Length.
        received: int = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] != "http.request":
                return message
            received += len(message.get("body", b""))
            if received > self.http_max_input_size:
                raise _BodyTooLargeError()
            return message

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, send_wrapper)
        except _BodyTooLargeError:
            if not response_started:
                await self._send_error(
                    scope, send,
                    status_code=StatusCode.CONTENT_TOO_LARGE,
                    code="content_too_large",
                    message=(
                        f"Request content size exceeds the maximum allowed "
                        f"input size of {self.http_max_input_size} bytes. "
                        f"Use --http-max-input-size to increase the limit."
                    ),
                )
            else:
                # Response headers already sent; cannot issue 413.
                logger.error(
                    "Body limit exceeded after response already started; "
                    "cannot send 413. path=%s",
                    scope.get("path", "?"),
                )

    async def _send_error(
        self,
        scope: Scope,
        send: Send,
        *,
        status_code: StatusCode,
        code: str,
        message: str,
    ) -> None:
        async def noop_receive() -> Message:
            return {"type": "http.disconnect"}

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
        await response(scope, noop_receive, send)
