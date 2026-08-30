from __future__ import annotations

import asyncio
import json
import os
import socket
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token

SHARED_OWNER_ENV = "ORCHARD_TEST_SHARED_OWNER"
START_BARRIER_ENV = "ORCHARD_TEST_BARRIER_SOCKET"
EXPECTED_PLAN_ENV = "ORCHARD_TEST_EXPECTED_PY_REQUEST_PLAN_JSON"
START_RELEASE = b"G"
PROTOCOL_VERSION = 3

_CURRENT_CASE: ContextVar[str | None] = ContextVar(
    "orchard_buckshot_case", default=None
)
_CURRENT_PHASE: ContextVar[int] = ContextVar("orchard_buckshot_phase", default=0)
_ACTIVE_GATE: BuckshotAdmissionGate | None = None


def shared_owner_enabled() -> bool:
    return os.getenv(SHARED_OWNER_ENV) == "1"


@contextmanager
def buckshot_phase(phase: int):
    """Label a causally dependent request phase inside one canonical case."""
    if phase < 0:
        raise ValueError("Buckshot request phases must be non-negative")
    token = _CURRENT_PHASE.set(phase)
    try:
        yield
    finally:
        _CURRENT_PHASE.reset(token)


async def observe_request_attempt() -> None:
    """Observe one real client transport attempt, immediately before sending it."""
    gate = _ACTIVE_GATE
    case_id = _CURRENT_CASE.get()
    if gate is None or case_id is None:
        return
    if not any(gate.request_plan[case_id]):
        return
    await gate.observe_request(case_id, _CURRENT_PHASE.get())


class BuckshotAdmissionGate:
    """Park every canonical case on the shared cross-process release gate."""

    def __init__(
        self,
        model_ids: Sequence[str],
        request_plan: Mapping[str, Sequence[int]],
    ) -> None:
        global _ACTIVE_GATE

        self.model_ids = list(model_ids)
        self.request_plan = {
            case_id: tuple(int(count) for count in phase_counts)
            for case_id, phase_counts in sorted(request_plan.items())
        }
        if not self.request_plan:
            raise ValueError("Buckshot requires at least one Python case")
        for case_id, phase_counts in self.request_plan.items():
            if any(count < 0 for count in phase_counts):
                raise ValueError(f"Negative request count in {case_id}: {phase_counts}")
            if phase_counts and phase_counts[0] == 0 and any(phase_counts[1:]):
                raise ValueError(f"{case_id} has dependent requests without phase 0")

        self.case_ids = tuple(self.request_plan)
        self._expected_cases = set(self.case_ids)
        self._started_cases: set[str] = set()
        self._completed_cases: set[str] = set()
        self._skipped = 0
        self._attempt_counts: Counter[tuple[str, int]] = Counter()
        self._attempts_total = 0
        self._retries = 0
        self._phase0_expected = sum(
            phase_counts[0] if phase_counts else 0
            for phase_counts in self.request_plan.values()
        )
        if self._phase0_expected == 0:
            raise ValueError("Buckshot requires at least one phase-0 inference request")

        self._lock = asyncio.Lock()
        self._released = asyncio.Event()
        self._release_task: asyncio.Task[socket.socket] | None = None
        self._release_error: Exception | None = None
        self._session: socket.socket | None = None
        self._context_tokens: dict[str, Token[str | None]] = {}

        if shared_owner_enabled():
            encoded_expected = os.getenv(EXPECTED_PLAN_ENV)
            if not encoded_expected:
                raise RuntimeError(
                    f"{EXPECTED_PLAN_ENV} is required when {SHARED_OWNER_ENV}=1"
                )
            try:
                decoded_plan = json.loads(encoded_expected)
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"{EXPECTED_PLAN_ENV} is not valid JSON") from error
            parent_plan = self._normalize_wire_plan(decoded_plan)
            if parent_plan != self.request_plan:
                raise RuntimeError(
                    "Python Buckshot request plan disagrees with the root owner's manifest"
                )

        if _ACTIVE_GATE is not None:
            raise RuntimeError("A Python Buckshot request gate is already active")
        _ACTIVE_GATE = self

    async def admit(self, case_id: str) -> None:
        """Park one canonical case until both clients have admitted their suites."""
        if case_id not in self._expected_cases:
            raise RuntimeError(f"Unmanifested Python Buckshot case: {case_id}")
        release_owner = False
        async with self._lock:
            if case_id in self._started_cases:
                raise RuntimeError(f"Duplicate Python Buckshot case start: {case_id}")
            self._started_cases.add(case_id)
            release_owner = self._started_cases == self._expected_cases
            if release_owner:
                if shared_owner_enabled():
                    self._release_task = asyncio.create_task(
                        asyncio.to_thread(self._announce_and_wait_for_release)
                    )
                else:
                    self._released.set()
        self._context_tokens[case_id] = _CURRENT_CASE.set(case_id)

        if release_owner and self._release_task is not None:
            try:
                self._session = await self._release_task
            except Exception as error:  # noqa: BLE001 - wake every parked case
                self._release_error = error
            finally:
                self._released.set()
        await self._released.wait()
        if self._release_error is not None:
            raise RuntimeError(
                f"Python Buckshot case release failed: {self._release_error}"
            ) from self._release_error

    async def observe_request(self, case_id: str, phase: int) -> None:
        if not self._released.is_set():
            raise RuntimeError(
                f"Python Buckshot request started before case release: {case_id}"
            )
        phase_counts = self.request_plan[case_id]
        async with self._lock:
            self._attempts_total += 1
            key = (case_id, phase)
            self._attempt_counts[key] += 1
            expected = phase_counts[phase] if phase < len(phase_counts) else 0
            if self._attempt_counts[key] > expected:
                self._retries += 1
                raise RuntimeError(
                    "Unplanned Python Buckshot request attempt "
                    f"for {case_id} phase {phase}: "
                    f"attempt={self._attempt_counts[key]} expected={expected}"
                )

    def complete(self, case_id: str, *, skipped: bool = False) -> None:
        if case_id not in self._started_cases:
            raise RuntimeError(
                f"Python Buckshot case completed before start: {case_id}"
            )
        if case_id in self._completed_cases:
            raise RuntimeError(
                f"Python Buckshot case completed more than once: {case_id}"
            )
        self._completed_cases.add(case_id)
        self._skipped += int(skipped)
        token = self._context_tokens.pop(case_id)
        _CURRENT_CASE.reset(token)

    async def finish(self, *, success: bool) -> None:
        global _ACTIVE_GATE

        missing_cases = sorted(self._expected_cases - self._completed_cases)
        missing_requests = self._missing_requests()
        errors: list[str] = []
        if missing_cases:
            errors.append(f"missing cases: {missing_cases}")
        if missing_requests:
            errors.append(f"missing request attempts: {missing_requests}")
        if self._retries:
            errors.append(f"observed {self._retries} excess request attempts")
        try:
            if self._session is not None:
                frame = {
                    "type": "done",
                    "protocol": PROTOCOL_VERSION,
                    "role": "orchard-py",
                    "cases_completed": len(self._completed_cases),
                    "request_attempts": self._attempts_total,
                    "phase_counts": self._actual_phase_counts(),
                    "filtered": 0,
                    "skipped": self._skipped,
                    "inference_retries": self._retries,
                    "success": success and not errors,
                }
                session = self._session
                self._session = None
                await asyncio.to_thread(self._send_and_close, session, frame)
            if errors:
                raise RuntimeError("Python Buckshot completion " + "; ".join(errors))
        finally:
            _ACTIVE_GATE = None

    def _announce_and_wait_for_release(self) -> socket.socket:
        barrier_path = os.getenv(START_BARRIER_ENV)
        if not barrier_path:
            raise RuntimeError(
                f"{START_BARRIER_ENV} is required when {SHARED_OWNER_ENV}=1"
            )

        cases = sorted(self._started_cases)
        if set(cases) != self._expected_cases:
            raise RuntimeError(
                "Python Buckshot announced before every case was admitted"
            )
        if self._attempts_total != 0:
            raise RuntimeError("Python Buckshot executed requests before case release")

        barrier = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            barrier.connect(barrier_path)
            self._send_frame(
                barrier,
                {
                    "type": "ready",
                    "protocol": PROTOCOL_VERSION,
                    "role": "orchard-py",
                    "models": self.model_ids,
                    "cases": cases,
                    "cases_admitted": len(cases),
                    "request_plan": self._wire_plan(),
                    "phase_counts": self._expected_phase_counts(),
                    "phase0_requests": self._phase0_expected,
                    "request_attempts": 0,
                    "filtered": 0,
                    "skipped": 0,
                    "inference_retries": self._retries,
                },
            )
            release = barrier.recv(1)
            if release != START_RELEASE:
                raise RuntimeError(
                    "Combined Buckshot owner closed without releasing the volley"
                )
            return barrier
        except BaseException:
            barrier.close()
            raise

    def _phase_attempt_count(self, phase: int) -> int:
        return sum(
            count
            for (_, observed_phase), count in self._attempt_counts.items()
            if observed_phase == phase
        )

    def _expected_phase_counts(self) -> list[int]:
        width = max((len(phases) for phases in self.request_plan.values()), default=0)
        return [
            sum(
                phases[phase] if phase < len(phases) else 0
                for phases in self.request_plan.values()
            )
            for phase in range(width)
        ]

    def _actual_phase_counts(self) -> list[int]:
        return [
            self._phase_attempt_count(phase)
            for phase in range(len(self._expected_phase_counts()))
        ]

    def _missing_requests(self) -> list[str]:
        missing: list[str] = []
        for case_id, phase_counts in self.request_plan.items():
            for phase, expected in enumerate(phase_counts):
                actual = self._attempt_counts[(case_id, phase)]
                if actual != expected:
                    missing.append(
                        f"{case_id}:phase-{phase} expected={expected} actual={actual}"
                    )
        return missing

    def _wire_plan(self) -> dict[str, list[int]]:
        return {case_id: list(phases) for case_id, phases in self.request_plan.items()}

    @staticmethod
    def _normalize_wire_plan(value: object) -> dict[str, tuple[int, ...]]:
        if not isinstance(value, dict):
            raise TypeError(f"{EXPECTED_PLAN_ENV} must be a JSON object")
        normalized: dict[str, tuple[int, ...]] = {}
        for case_id, phases in value.items():
            if (
                not isinstance(case_id, str)
                or not isinstance(phases, list)
                or not all(isinstance(count, int) for count in phases)
            ):
                raise RuntimeError(
                    f"{EXPECTED_PLAN_ENV} must map case strings to integer lists"
                )
            normalized[case_id] = tuple(phases)
        return dict(sorted(normalized.items()))

    @staticmethod
    def _send_frame(connection: socket.socket, frame: dict[str, object]) -> None:
        payload = json.dumps(frame, separators=(",", ":"), sort_keys=True).encode()
        connection.sendall(payload + b"\n")

    @classmethod
    def _send_and_close(
        cls, connection: socket.socket, frame: dict[str, object]
    ) -> None:
        try:
            cls._send_frame(connection, frame)
        finally:
            connection.close()
