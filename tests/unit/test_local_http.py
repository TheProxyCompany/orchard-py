import ssl

import pytest

from tests.local_http import local_async_client


@pytest.mark.asyncio
async def test_local_http_client_does_not_load_certificate_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_ca_load(*args, **kwargs):
        raise AssertionError("loopback HTTP clients must not load TLS authorities")

    monkeypatch.setattr(ssl, "create_default_context", unexpected_ca_load)

    async with local_async_client():
        pass
