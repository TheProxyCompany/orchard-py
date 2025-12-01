import logging
from typing import Any

logger = logging.getLogger(__name__)

ResponseDeltaDict = dict[str, Any]


def release_delta_resources(delta_item: ResponseDeltaDict) -> None:
    """Release any shared-memory resources associated with a delta."""
    if not delta_item:
        return

    seen_views: set[int] = set()
    for key in ("bulk_content_view", "bulk_content_bytes", "embedding_bytes"):
        value = delta_item.get(key)
        if isinstance(value, memoryview):
            view_id = id(value)
            if view_id not in seen_views:
                seen_views.add(view_id)
                try:
                    value.release()
                except (AttributeError, BufferError):  # pragma: no cover - defensive
                    pass
            delta_item[key] = None


def normalise_delta_payload(delta_item: ResponseDeltaDict) -> None:
    if not delta_item:
        return

    existing_text = delta_item.get("content") or ""
    if isinstance(existing_text, str) and existing_text:
        return

    buffer_value = delta_item.get("bulk_content_view") or delta_item.get(
        "bulk_content_bytes"
    )
    if isinstance(buffer_value, memoryview):
        try:
            delta_item["content"] = buffer_value.tobytes().decode(
                "utf-8", errors="replace"
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Dispatcher: Failed to decode bulk memoryview for request %s: %s",
                delta_item.get("request_id"),
                exc,
                exc_info=True,
            )
    elif isinstance(buffer_value, bytes | bytearray):
        try:
            delta_item["content"] = bytes(buffer_value).decode("utf-8")
        except UnicodeDecodeError:
            delta_item["content"] = bytes(buffer_value).decode(
                "utf-8", errors="replace"
            )
