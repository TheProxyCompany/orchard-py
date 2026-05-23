from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from orchard.clients.client import Client
from orchard.engine import ClientDelta


class OpenAIPrivacyFilterClient(Client):
    model_id = "openai/privacy-filter"
    max_input_bytes = 16 * 1024

    async def aanalyze(
        self,
        text: str,
        *,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[ClientDelta]:
        self._check_input_size(text)
        result = await self.aprefill_task(
            self.model_id,
            text,
            "privacy_filter",
            stream=stream,
        )
        if stream:
            return result
        assert isinstance(result, list)
        return self._first_payload(result)

    async def aanalyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        for text in texts:
            self._check_input_size(text)
        deltas_by_prompt = await self.aprefill_task_batch(
            self.model_id,
            texts,
            "privacy_filter",
        )
        return self._payloads_by_prompt(deltas_by_prompt)

    def analyze(
        self,
        text: str,
        *,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[ClientDelta]:
        self._check_input_size(text)
        result = self.prefill_task(
            self.model_id,
            text,
            "privacy_filter",
            stream=stream,
        )
        if stream:
            return result
        assert isinstance(result, list)
        return self._first_payload(result)

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        for text in texts:
            self._check_input_size(text)
        return self._payloads_by_prompt(
            self.prefill_task_batch(self.model_id, texts, "privacy_filter")
        )

    @classmethod
    def _check_input_size(cls, text: str) -> None:
        byte_count = len(text.encode("utf-8"))
        if byte_count > cls.max_input_bytes:
            raise ValueError(
                f"privacy filter input exceeds {cls.max_input_bytes} bytes; callers must chunk larger inputs"
            )

    @staticmethod
    def _first_payload(deltas: list[ClientDelta]) -> dict[str, Any]:
        for delta in deltas:
            if delta.modal_decoder_id == "privacy_filter" and delta.modal_bytes_b64:
                payload = base64.b64decode(delta.modal_bytes_b64)
                return json.loads(payload.decode("utf-8"))
        raise RuntimeError("privacy filter response did not include a result payload")

    @classmethod
    def _payloads_by_prompt(
        cls, deltas_by_prompt: list[list[ClientDelta]]
    ) -> list[dict[str, Any]]:
        return [cls._first_payload(deltas) for deltas in deltas_by_prompt]
