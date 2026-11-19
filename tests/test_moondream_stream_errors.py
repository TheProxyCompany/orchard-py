import pytest
from PIL import Image

from orchard.clients.moondream import MoondreamClient
from orchard.engine import ClientDelta
from orchard.server.exceptions import InferenceError


@pytest.mark.parametrize(
    ("deltas", "error"),
    [
        ([ClientDelta(request_id=1, sequence_id=1, is_final=True, finish_reason="error",
                      error_message="workspace refused")], "workspace refused"),
        ([], "stream ended before its final delta"),
        ([ClientDelta(request_id=1, sequence_id=1, is_final=True, finish_reason="stop")], None),
    ],
)
def test_point_distinguishes_a_clean_empty_result_from_interrupted_inference(monkeypatch, deltas, error):
    client = object.__new__(MoondreamClient)
    monkeypatch.setattr(client, "chat", lambda *args, **kwargs: iter(deltas))
    image = Image.new("RGB", (8, 8))
    if error:
        with pytest.raises(InferenceError, match=error):
            client.point(image, "apple")
    else:
        assert client.point(image, "apple") == {"points": []}
