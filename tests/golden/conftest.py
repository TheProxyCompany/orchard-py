"""Golden-tier conftest: persist staged baselines only when a test passes.

`assert_or_record` stages new baselines in memory rather than writing them. The
hook below records each test's outcome; the autouse fixture then flushes the
staged baselines to disk on pass, or discards them on fail — so a failing test
never leaves a buggy golden behind.
"""

import pytest

from tests.golden import golden_io


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"_rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _golden_record_guard(request):
    golden_io.discard_pending()
    yield
    rep = getattr(request.node, "_rep_call", None)
    if rep is not None and rep.passed:
        golden_io.flush_pending()
    else:
        golden_io.discard_pending()
