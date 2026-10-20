"""Summarize traces without treating automatic flags as semantic ground truth."""
from __future__ import annotations
import argparse
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from .annotations import annotation_summary


def cycle_summaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe published sequences, including interrupted/unfinished ones.

    Interrupt records carry the new epoch, so they end the active cycle rather
    than the cycle matching their epoch. Late discarded results are not speech.
    """
    cycles: dict[int, dict[str, Any]] = {}
    active = None
    preceding_human = None

    def end(cycle: dict[str, Any], event: dict[str, Any], outcome: str) -> None:
        cycle.update(outcome=outcome, end_seq=event.get("seq"), reason=event.get("reason"))

    for event in events:
        kind, epoch = event.get("kind"), event.get("epoch")
        if kind == "utterance" and event.get("speaker") == "human":
            preceding_human = event.get("seq")
        if kind == "wake" and type(epoch) is int:
            if active is not None:
                end(active, event, "superseded_by_wake")
            active = {"epoch": epoch, "wake_seq": event.get("seq"),
                      "trigger": event.get("reason"), "preceding_human_seq": preceding_human,
                      "requests": 0, "assistant_seqs": [], "assistant_utterances": 0,
                      "outcome": "open_at_snapshot", "end_seq": None, "reason": None}
            cycles[epoch] = active
        cycle = cycles.get(epoch) if type(epoch) is int else None
        if cycle is not None:
            if kind == "request":
                cycle["requests"] += 1
            elif kind == "utterance" and event.get("speaker") == "assistant":
                cycle["assistant_seqs"].append(event["seq"])
                cycle["assistant_utterances"] += 1
            elif kind in ("yield", "forced_stop", "harness_error"):
                end(cycle, event, {"yield": "model_yield", "forced_stop": "harness_stop",
                                   "harness_error": "harness_error"}[kind])
                if active is cycle:
                    active = None
        if active is not None and kind in ("interrupt", "session_close"):
            end(active, event, "interrupted" if kind == "interrupt" else "session_closed")
            active = None
    return list(cycles.values())


def presence_observations(events: list[dict[str, Any]]) -> dict[str, int]:
    """Count the activity state delivered in requests, not UI typing events."""
    counts = {"human_typing_true": 0, "human_typing_false": 0, "missing_or_invalid": 0}
    for event in events:
        if event.get("kind") != "request":
            continue
        try:
            payload = json.loads(event["input"])
            value = payload["presence"]["human_typing"]
        except (KeyError, ValueError, TypeError):
            value = None
        key = "human_typing_true" if value is True else "human_typing_false" if value is False else "missing_or_invalid"
        counts[key] += 1
    return counts


def summarize(path: Path) -> dict[str, Any]:
    events = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON; partial logs must be reviewed") from exc
    start = next((e for e in events if e.get("kind") == "session_open"), None)
    if start is None:
        raise ValueError(f"{path}: no session_open record")
    settings = start["settings"]
    kinds = Counter(e.get("kind") for e in events)
    assistant = [e for e in events if e.get("kind") == "utterance" and e.get("speaker") == "assistant"]
    yields = [e for e in events if e.get("kind") == "yield"]
    raw = [e for e in events if e.get("kind") == "raw_output"]
    requests = kinds["request"]
    return {
        "file": str(path), "run": start["session_id"],
        "fixture_not_llm_evidence": settings["provider"] == "fixture",
        "provider": settings["provider"], "requested_model": settings["model"],
        "actual_models": sorted({e["actual_model"] for e in raw if e.get("actual_model")}),
        "version": start.get("version"),
        "protocol_version": start.get("provenance", {}).get("protocol_version", "legacy_unspecified"),
        "source_sha256": start.get("provenance", {}).get("source_sha256"),
        "constraint_mode": start.get("constraint_mode", "legacy_unspecified"),
        "conversation_language": settings.get("conversation_language", "legacy_unspecified"),
        "typing_mode": settings.get("typing_mode", "legacy_unspecified"),
        "local_runtime": settings.get("local_runtime", "legacy_unspecified"),
        "local_prompt_mode": settings.get("local_prompt_mode", "legacy_unspecified"),
        "actual_local_runtimes": sorted({(e.get("metadata") or {}).get("runtime") for e in raw
                                        if (e.get("metadata") or {}).get("runtime")}),
        "actual_local_prompt_modes": sorted({(e.get("metadata") or {}).get("prompt_mode") for e in raw
                                            if (e.get("metadata") or {}).get("prompt_mode")}),
        "policy": settings["policy"], "structured": settings["structured"],
        "max_utterances": settings["max_utterances"], "min_gap_ms": settings["min_gap_ms"],
        "proactive": settings["proactive"], "requests": requests,
        "cycles": cycle_summaries(events),
        "presence_observations_at_request": presence_observations(events),
        "assistant_utterances": len(assistant), "model_yields": len(yields),
        "zero_emit_yields_not_necessarily_failures": sum(e.get("emitted_in_cycle") == 0 for e in yields),
        "forced_stop_reasons": dict(Counter(e.get("reason") for e in events if e.get("kind") == "forced_stop")),
        "interrupts": kinds["interrupt"], "discarded_decisions": kinds["discard"],
        "protocol_error_count": kinds["protocol_error"],
        "protocol_error_per_request": kinds["protocol_error"] / requests if requests else None,
        "provider_error_count": kinds["provider_error"],
        "finish_reasons": dict(Counter(e.get("finish_reason", "legacy_unspecified") for e in raw)),
        "latency_median_ms": statistics.median(e["latency_ms"] for e in raw) if raw else None,
        "automatic_flags_not_ground_truth": dict(Counter(e.get("tag") for e in events if e.get("kind") == "flag")),
        # Retained for older consumers: raw label counts, including superseded
        # and legacy observations. Use annotation_review for current judgments.
        "human_annotations": dict(Counter(e["tag"] for e in events
                                          if e.get("kind") == "annotation" and e.get("tag"))),
        "annotation_review": annotation_summary(events),
        "trace_has_unresolved_requests": requests > len(raw) + sum(e.get("kind") == "provider_error" and "latency_ms" in e for e in events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize turnless JSONL traces, one JSON summary per file")
    parser.add_argument("paths", nargs="+", help="trace paths or wildcard patterns")
    args = parser.parse_args()
    files = sorted({Path(p) for pattern in args.paths for p in glob.glob(pattern)})
    if not files:
        parser.error("no matching trace files")
    failed = False
    for path in files:
        try:
            print(json.dumps(summarize(path), ensure_ascii=False))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(str(exc), file=sys.stderr)
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
