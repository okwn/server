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

from utils.utils import StatusCode, validate_positive_int

_DISCONNECT_MESSAGE: Message = {"type": "http.disconnect"}


async def _disconnect_receive() -> Message:
    return _DISCONNECT_MESSAGE


class RequestSizeLimitMiddleware:
    """
    ASGI middleware that enforces a hard cap on HTTP request body size.

    Stage 1: reject immediately if Content-Length exceeds the limit
             (zero body bytes read).
    Stage 2: drain the receive() stream from the middleware, counting bytes
             as they arrive. Reject as soon as the running total exceeds the
             limit. Once the full body is buffered within the limit, replay
             it to the application as a single http.request message and drop
             the middleware's reference so the body is collectable as soon
             as the framework finishes consuming it.

    Driving receive() unconditionally guarantees the limit is enforced for
    every endpoint, including handlers that never read the body. Releasing
    the buffered body on hand-off keeps the steady-state middleware overhead
    at ~0 bytes during request processing (the body lives only in the layer
    that actually parses it).
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
                        StatusCode.CLIENT_ERROR,
                        "invalid_content_length",
                        "Invalid Content-Length header: not an integer.",
                    )
                    return
                if content_length < 0:
                    await self._send_error(
                        scope, send,
                        StatusCode.CLIENT_ERROR,
                        "invalid_content_length",
                        "Invalid Content-Length header: must be non-negative.",
                    )
                    return
                if content_length > self.http_max_input_size:
                    await self._send_too_large(scope, send)
                    return
                break

        # Stage 2: buffer body chunks, enforce limit, then replay to app.
        body_chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # http.disconnect (client gone) or an unexpected message
                # type. Abort without invoking the app on an incomplete
                # body and without sending a response — the client is not
                # there to receive it, and treating unknown types as
                # terminal prevents an infinite receive() loop.
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.http_max_input_size:
                await self._send_too_large(scope, send)
                return
            body_chunks.append(chunk)
            if not message.get("more_body", False):
                break

        # Single-chunk fast path avoids a copy in the common case where the
        # ASGI server delivers the whole body in one message.
        body_message: Message = {
            "type": "http.request",
            "body": body_chunks[0] if len(body_chunks) == 1 else b"".join(body_chunks),
            "more_body": False,
        }
        body_chunks = None  # release chunk list before app runs

        async def replay_receive() -> Message:
            nonlocal body_message
            if body_message is None:
                return _DISCONNECT_MESSAGE
            # Hand the message to the app and drop our reference so the body
            # bytes can be freed by the framework as soon as it has finished
            # buffering them, instead of being held alive by this closure for
            # the full duration of the request.
            message, body_message = body_message, None
            return message

        await self.app(scope, replay_receive, send)

    async def _send_too_large(self, scope: Scope, send: Send) -> None:
        await self._send_error(
            scope, send,
            StatusCode.CONTENT_TOO_LARGE,
            "content_too_large",
            f"Request content size exceeds the maximum allowed input size of "
            f"{self.http_max_input_size} bytes. Use --http-max-input-size to "
            f"increase the limit.",
        )

    async def _send_error(
        self,
        scope: Scope,
        send: Send,
        status_code: StatusCode,
        code: str,
        message: str,
    ) -> None:
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
        await response(scope, _disconnect_receive, send)
