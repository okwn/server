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

import json

from utils.utils import StatusCode

# 64 MiB default, matching core Triton's HTTP_DEFAULT_MAX_INPUT_SIZE.
DEFAULT_MAX_INPUT_SIZE: int = 1 << 26


class RequestSizeLimitMiddleware:
    """
    Reject requests with a Content-Length header greater than max_input_size.
    """

    def __init__(self, app, max_input_size: int = DEFAULT_MAX_INPUT_SIZE):
        if max_input_size <= 0:
            raise ValueError(
                f"max_input_size must be greater than 0, got {max_input_size}"
            )
        self.app = app
        self.max_input_size = max_input_size

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            for name, value in scope.get("headers", []):
                if name != b"content-length":
                    continue
                try:
                    declared = int(value)
                except (ValueError, TypeError):
                    break
                if declared > self.max_input_size:
                    await self._reject(send, declared)
                    return
                break
        await self.app(scope, receive, send)

    async def _reject(self, send, byte_size: int) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": (
                        f"Request body size of {byte_size} bytes exceeds the "
                        f"maximum allowed input size of {self.max_input_size} "
                        f"bytes. Use --http-max-input-size to increase the "
                        f"limit."
                    ),
                    "type": "invalid_request_error",
                    "code": "request_too_large",
                }
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": int(StatusCode.PAYLOAD_TOO_LARGE),
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
