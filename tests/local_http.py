import ssl
from typing import Any

import httpx

from tests.shared_owner import observe_request_attempt

_LOOPBACK_SSL_CONTEXT = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
_LOOPBACK_SSL_CONTEXT.check_hostname = False
_LOOPBACK_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


class _BuckshotAwareAsyncClient(httpx.AsyncClient):
    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        await observe_request_attempt()
        return await super().send(request, **kwargs)


def local_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """Build a client for the suite's plaintext loopback-only HTTP server."""
    return _BuckshotAwareAsyncClient(
        verify=_LOOPBACK_SSL_CONTEXT,
        trust_env=False,
        **kwargs,
    )
