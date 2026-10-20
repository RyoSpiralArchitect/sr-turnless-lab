"""One serial inference worker per experiment, with epoch-based interruption."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import ProtocolError, Settings, make_prompt, parse_decision
from .annotations import annotation_payload, observation_targets
from .providers import Provider
from . import __version__
from .catalog import constraint_mode
from .provenance import source_identity

TYPING_LEASE_SECONDS = 3.0


@dataclass
class Cycle:
    epoch: int
    trigger: str
    cancel: threading.Event = field(default_factory=threading.Event)
    emitted: int = 0
    started: float = field(default_factory=time.monotonic)


class Session:
    def __init__(self, settings: Settings, provider: Provider, log_dir: Path | None = None):
        self.id = uuid.uuid4().hex
        self.settings, self.provider = settings, provider
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.events: list[dict[str, Any]] = []
        self.started = time.monotonic()
        self.last_access = time.monotonic()
        self.epoch = 0
        self.calls = 0
        self.status = "waiting"
        self.closed = False
        self.inflight = False
        self._human_typing = False
        self._typing_seen = 0.0
        self._current: Cycle | None = None
        self._pending: Cycle | None = None
        self.log_path: Path | None = None
        self.provenance = source_identity()
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.log_path = log_dir / f"{self.id}.jsonl"
            self.log_path.touch(mode=0o600, exist_ok=False)
        with self.lock:
            self._event("session_open", source="harness", settings=settings.public(),
                        version=__version__, provenance=self.provenance,
                        constraint_mode=constraint_mode(settings.provider, settings.structured, settings.anthropic_mode),
                        fixture=settings.provider == "fixture")
        self.worker = threading.Thread(target=self._worker, name=f"turnless-{self.id[:8]}", daemon=True)
        self.worker.start()
        if settings.proactive:
            self.wake("session_open")

    def _event(self, kind: str, **data: Any) -> dict[str, Any]:
        # Caller owns the RLock. Persist before making an event visible.
        event = {"seq": len(self.events) + 1, "session_id": self.id, "kind": kind,
                 "at_ms": round((time.monotonic() - self.started) * 1000),
                 "timestamp": datetime.now(timezone.utc).isoformat(), **data}
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.events.append(event)
        return event

    def _valid(self, cycle: Cycle) -> bool:
        return not self.closed and cycle.epoch == self.epoch and not cycle.cancel.is_set()

    def _invalidate(self, reason: str) -> None:
        interrupted = self.status in ("queued", "generating", "delaying")
        self.epoch += 1
        if self._current:
            self._current.cancel.set()
        if self._pending:
            self._pending.cancel.set()
        self._pending = None
        if interrupted:
            self._event("interrupt", source="harness", reason=reason, epoch=self.epoch,
                        upstream_cancelled=False)
        self.status = "waiting"

    def _queue(self, trigger: str) -> None:
        cycle = Cycle(self.epoch, trigger)
        self._pending = cycle
        self.status = "queued"
        self._event("wake", source="harness", reason=trigger, epoch=cycle.epoch)
        self.condition.notify_all()

    def wake(self, trigger: str = "operator_wake") -> None:
        with self.condition:
            if self.closed:
                raise ValueError("experiment is closed")
            if self.status in ("queued", "generating", "delaying"):
                raise ValueError("The experiment is active. Interrupt it before waking again.")
            self.epoch += 1
            self._queue(trigger)

    def human(self, text: str, auto_reply: bool = True) -> None:
        if not isinstance(text, str) or not text.strip() or len(text) > 8000:
            raise ValueError("Enter a message between 1 and 8000 characters.")
        if type(auto_reply) is not bool:
            raise ValueError("auto_reply must be boolean")
        with self.condition:
            if self.closed:
                raise ValueError("experiment is closed")
            self._set_typing(False, "message_sent")
            self._invalidate("human_message")
            self._event("utterance", source="human", speaker="human", text=text.strip(), epoch=self.epoch,
                        auto_reply_requested=auto_reply)
            if auto_reply:
                self._queue("human_message")

    def interrupt(self, reason: str = "operator_stop") -> None:
        if reason not in ("operator_stop", "human_typing"):
            raise ValueError("invalid interrupt reason")
        with self.condition:
            if not self.closed:
                if reason == "human_typing":
                    self.typing(True)
                    return
                self._invalidate(reason)
                self._event("pause", source="harness", reason=reason, epoch=self.epoch)

    def _set_typing(self, active: bool, reason: str) -> None:
        if active != self._human_typing:
            self._event("typing", source="harness", actor="human", active=active,
                        reason=reason, mode=self.settings.typing_mode, epoch=self.epoch,
                        lease_ms=round(TYPING_LEASE_SECONDS * 1000))
            self._human_typing = active

    def _expire_typing(self) -> None:
        if self._human_typing and time.monotonic() - self._typing_seen >= TYPING_LEASE_SECONDS:
            self._set_typing(False, "idle_timeout")

    def typing(self, active: bool) -> None:
        """Accept activity only, never draft text. A heartbeat renews the lease."""
        if type(active) is not bool:
            raise ValueError("active must be boolean")
        with self.condition:
            if self.closed:
                raise ValueError("experiment is closed")
            self._expire_typing()
            self._typing_seen = time.monotonic()
            self._set_typing(active, "input_activity" if active else "input_stopped")
            if active and self.settings.typing_mode == "interrupt" and self.status in ("queued", "generating", "delaying"):
                self._invalidate("human_typing")
                self._event("pause", source="harness", reason="human_typing", epoch=self.epoch)
            self.condition.notify_all()

    def _presence(self) -> dict[str, Any]:
        self._expire_typing()
        return {"human_typing": self._human_typing,
                "observed_at_ms": round((time.monotonic() - self.started) * 1000),
                "lease_ms": round(TYPING_LEASE_SECONDS * 1000)}

    def close(self) -> None:
        with self.condition:
            if not self.closed:
                self._set_typing(False, "session_close")
                self._invalidate("session_close")
                self.closed, self.status = True, "closed"
                self._event("session_close", source="harness", upstream_pending=self.inflight)
                self.condition.notify_all()

    def annotate(self, target_seq: int, tag: str | None = None, note: str = "", *,
                 scope: str = "event", judgment: str = "labeled",
                 supersedes: int | None = None) -> None:
        with self.lock:
            payload = annotation_payload(self.events, target_seq, tag, note, scope=scope,
                                         judgment=judgment, supersedes=supersedes)
            self._event("annotation", source="observer", **payload)

    def _finish(self, cycle: Cycle, kind: str, reason: str) -> None:
        if self._valid(cycle):
            self.status = "waiting"
            self._event(kind, source="harness", reason=reason, epoch=cycle.epoch)

    def _worker(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self._pending is not None, timeout=0.5)
                self._expire_typing()
                if self.closed:
                    return
                if self._pending is None:
                    continue
                cycle = self._pending
                self._pending = None
                self._current = cycle
            try:
                self._run(cycle)
            except Exception as exc:
                # Fail closed even on unexpected implementation or filesystem errors.
                with self.lock:
                    self.inflight = False
                    if self._valid(cycle):
                        self.status = "waiting"
                        try:
                            self._event("harness_error", source="harness", epoch=cycle.epoch,
                                        error=f"{type(exc).__name__}: {exc}")
                        except OSError:
                            self.status = "log_error"
            finally:
                with self.lock:
                    if self._current is cycle:
                        self._current = None

    def _run(self, cycle: Cycle) -> None:
        while True:
            with self.lock:
                if not self._valid(cycle):
                    return
                if cycle.emitted >= self.settings.max_utterances:
                    self._finish(cycle, "forced_stop", "utterance_cap")
                    return
                if self.calls >= self.settings.call_budget:
                    self._finish(cycle, "forced_stop", "session_call_budget")
                    return
                if time.monotonic() - cycle.started > 120:
                    self._finish(cycle, "forced_stop", "cycle_time_budget")
                    return
                presence = self._presence()
                instructions, payload = make_prompt(self.settings, self.events, cycle.trigger, cycle.emitted, presence)
                if len(instructions) + len(payload) > 80_000:
                    self._finish(cycle, "forced_stop", "context_character_budget")
                    return
                self.calls += 1
                call_id = self.calls
                self.status, self.inflight = "generating", True
                describe = getattr(self.provider, "describe_request", None)
                wire = describe(instructions, payload, self.settings) if describe else None
                self._event("request", source="harness", epoch=cycle.epoch, call_id=call_id,
                            instructions=instructions, input=payload,
                            prompt_sha256=hashlib.sha256((instructions + "\n" + payload).encode()).hexdigest(),
                            provider=self.settings.provider, model=self.settings.model,
                            structured=self.settings.structured, max_output_tokens=self.settings.max_output_tokens,
                            wire_request=wire)
            began = time.monotonic()
            try:
                result = self.provider.generate(instructions, payload, self.settings, cycle.cancel)
            except Exception as exc:
                with self.lock:
                    self.inflight = False
                    obsolete = not self._valid(cycle)
                    self._event("provider_error", source="provider", epoch=cycle.epoch, call_id=call_id,
                                discarded=obsolete, error=f"{type(exc).__name__}: {exc}",
                                latency_ms=round((time.monotonic() - began) * 1000))
                    if not obsolete:
                        self._finish(cycle, "forced_stop", "provider_error")
                return
            with self.lock:
                self.inflight = False
                obsolete = not self._valid(cycle)
                raw_event = self._event("raw_output", source="provider", epoch=cycle.epoch, call_id=call_id,
                                       raw=result.raw, usage=result.usage, request_id=result.request_id,
                                       actual_model=result.model, provider_status=result.status,
                                       finish_reason=result.finish_reason, public_response=result.public_response,
                                       metadata=result.metadata,
                                       latency_ms=round((time.monotonic() - began) * 1000), discarded=obsolete)
                if obsolete:
                    self._event("discard", source="harness", epoch=cycle.epoch, call_id=call_id,
                                reason="stale_generation", target_seq=raw_event["seq"])
                    return
                if result.error or result.status != "completed":
                    self._event("provider_error", source="provider", epoch=cycle.epoch, call_id=call_id,
                                error=result.error or result.status, target_seq=raw_event["seq"], discarded=False)
                    self._finish(cycle, "forced_stop", "provider_status")
                    return
                try:
                    decision = parse_decision(result.raw)
                except ProtocolError as exc:
                    self._event("protocol_error", source="harness", epoch=cycle.epoch, call_id=call_id,
                                error=str(exc), target_seq=raw_event["seq"])
                    self._finish(cycle, "forced_stop", "protocol_error")
                    return
                if decision.action == "yield":
                    self.status = "waiting"
                    self._event("yield", source="model", epoch=cycle.epoch, call_id=call_id,
                                emitted_in_cycle=cycle.emitted)
                    return
                delay = max(decision.delay_ms, self.settings.min_gap_ms if cycle.emitted else 0)
                self.status = "delaying"
                self._event("decision", source="model", epoch=cycle.epoch, call_id=call_id,
                            action="emit", proposed_delay_ms=decision.delay_ms, applied_delay_ms=delay,
                            target_seq=raw_event["seq"])
            if cycle.cancel.wait(delay / 1000):
                with self.lock:
                    self._event("discard", source="harness", epoch=cycle.epoch, call_id=call_id,
                                reason="interrupted_before_emit", target_seq=raw_event["seq"])
                return
            with self.lock:
                if not self._valid(cycle):
                    self._event("discard", source="harness", epoch=cycle.epoch, call_id=call_id,
                                reason="stale_before_commit", target_seq=raw_event["seq"])
                    return
                cycle.emitted += 1
                previous = [e for e in self.events if e["kind"] == "utterance" and e["speaker"] == "assistant"]
                emitted = self._event("utterance", source="model", speaker="assistant", text=decision.text,
                                      epoch=cycle.epoch, call_id=call_id, index_in_cycle=cycle.emitted)
                normal = lambda s: re.sub(r"\s+", "", s).casefold()
                if previous and normal(previous[-1]["text"]) == normal(decision.text):
                    self._event("flag", source="heuristic", tag="exact-repeat", target_seq=emitted["seq"],
                                note="Normalized string match; not a semantic failure judgment.")
                if re.search(r"(?im)^\s*(?:<\|user\|>|user\s*:|human\s*:|人間\s*[:：])", decision.text):
                    self._event("flag", source="heuristic", tag="possible-fake-user", target_seq=emitted["seq"],
                                note="An observer must distinguish quotation from fabrication. Not a human event.")

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        with self.lock:
            self.last_access = time.monotonic()
            presence = self._presence()
            presence["assistant_composing"] = self.status in ("queued", "generating", "delaying")
            count = lambda kind: sum(e["kind"] == kind for e in self.events)
            token_in = token_out = 0
            for e in self.events:
                if e["kind"] == "raw_output":
                    usage = e.get("usage") or {}
                    token_in += usage.get("input_tokens", 0) or 0
                    token_out += usage.get("output_tokens", 0) or 0
            return {
                "session_id": self.id, "status": self.status, "inflight": self.inflight,
                "presence": presence,
                "settings": self.settings.public(), "cursor": len(self.events),
                "events": [e.copy() for e in self.events if e["seq"] > after],
                "annotation_targets": observation_targets(self.events),
                "metrics": {
                    "calls": self.calls, "budget": self.settings.call_budget,
                    "assistant": sum(e["kind"] == "utterance" and e["speaker"] == "assistant" for e in self.events),
                    "human": sum(e["kind"] == "utterance" and e["speaker"] == "human" for e in self.events),
                    "yields": count("yield"), "forced_stops": count("forced_stop"),
                    "interrupts": count("interrupt"), "discards": count("discard"),
                    "flags": count("flag"), "annotations": count("annotation"),
                    "input_tokens": token_in, "output_tokens": token_out,
                },
            }

    def export_jsonl(self) -> str:
        with self.lock:
            self._expire_typing()
            data = [*self.events, {
                "kind": "export_metadata", "session_id": self.id, "version": __version__,
                "source_sha256": self.provenance["source_sha256"],
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
                "upstream_pending": self.inflight, "closed": self.closed,
                "warning": "Fixture traces are not LLM evidence. Raw traces are not FT-ready datasets.",
            }]
            return "\n".join(json.dumps(e, ensure_ascii=False) for e in data) + "\n"
