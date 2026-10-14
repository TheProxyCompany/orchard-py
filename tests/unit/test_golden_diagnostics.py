"""Run without engine startup: python -m unittest tests.unit.test_golden_diagnostics."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from tests.golden import golden_io


class Event(BaseModel):
    type: str = "response.output_text.delta"
    sequence_number: int = 0
    item_id: str = "msg_live"
    created_at: int = 123
    delta: str


class GoldenDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "data"
        self.output = self.root / "diagnostics"
        self.snapshot = self.data / "gemma4" / "tool_chaining.json"
        self.snapshot.parent.mkdir(parents=True)
        self.events = [Event(delta="Response"), Event(sequence_number=1, delta="done")]
        self.expected = golden_io.normalize([Event(delta="Output")])
        self.variant = golden_io.normalize([Event(delta="Other")])
        self.snapshot.write_text(json.dumps({"turn1": [self.expected, self.variant]}))
        self.original = self.snapshot.read_bytes()
        self.enterContext(patch.object(golden_io, "DATA_DIR", self.data))
        self.enterContext(
            patch.dict(
                os.environ, {"ORCHARD_GOLDEN_DIFF_DIR": "", "GOLDEN_ADD_VARIANT": ""}
            )
        )
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        golden_io.discard_pending()
        self.addCleanup(golden_io.discard_pending)

    def assert_drift(self):
        with self.assertRaisesRegex(
            AssertionError, "golden drift gemma4/tool_chaining/turn1"
        ):
            golden_io.assert_or_record("gemma4", "tool_chaining", "turn1", self.events)
        self.assertEqual(self.snapshot.read_bytes(), self.original)
        self.assertEqual(golden_io.pending_paths(), [])

    def test_disabled_capture_keeps_failure_and_writes_nothing(self):
        self.assert_drift()
        self.assertFalse(self.output.exists())

    def test_full_stream_and_all_variants_saved_without_accepting_drift(self):
        os.environ["ORCHARD_GOLDEN_DIFF_DIR"] = str(self.output)
        self.assert_drift()
        (directory,) = self.output.iterdir()
        self.assertEqual(
            json.loads((directory / "actual.json").read_text()),
            golden_io.normalize(self.events),
        )
        self.assertEqual(
            json.loads((directory / "expected.json").read_text()), self.expected
        )
        self.assertEqual(
            json.loads((directory / "expected-variants.json").read_text()),
            [self.expected, self.variant],
        )
        self.assertEqual(
            json.loads((directory / "case.json").read_text())["variant_count"], 2
        )
        # Recording hooks must not acquire anything to flush from diagnostics.
        golden_io.flush_pending()
        self.assertEqual(self.snapshot.read_bytes(), self.original)
        previous = {p.name: p.read_bytes() for p in directory.iterdir()}
        self.assert_drift()
        self.assertEqual(len(list(self.output.iterdir())), 2)
        self.assertEqual(
            {p.name: p.read_bytes() for p in directory.iterdir()}, previous
        )

    def test_capture_failure_does_not_replace_golden_failure(self):
        self.output.write_text("not a directory")
        os.environ["ORCHARD_GOLDEN_DIFF_DIR"] = str(self.output)
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assert_drift()
        self.assertIn("could not save drift evidence", errors.getvalue())

    def test_matching_variant_does_not_capture(self):
        os.environ["ORCHARD_GOLDEN_DIFF_DIR"] = str(self.output)
        golden_io.assert_or_record(
            "gemma4", "tool_chaining", "turn1", [Event(delta="Other")]
        )
        self.assertFalse(self.output.exists())
        self.assertEqual(self.snapshot.read_bytes(), self.original)

    def test_explicit_variant_still_stages_without_diagnostic_failure(self):
        os.environ["ORCHARD_GOLDEN_DIFF_DIR"] = str(self.output)
        os.environ["GOLDEN_ADD_VARIANT"] = "1"
        golden_io.assert_or_record("gemma4", "tool_chaining", "turn1", self.events)
        self.assertEqual(self.snapshot.read_bytes(), self.original)
        self.assertEqual(golden_io.pending_paths(), [self.snapshot])
        self.assertFalse(self.output.exists())
        golden_io.discard_pending()
        golden_io.flush_pending()
        self.assertEqual(self.snapshot.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
