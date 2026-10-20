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
        "source_sha256": start.get("provenance", {}).get("source_sha256"),
        "constraint_mode": start.get("constraint_mode", "legacy_unspecified"),
        "conversation_language": settings.get("conversation_language", "legacy_unspecified"),
        "policy": settings["policy"], "structured": settings["structured"],
        "max_utterances": settings["max_utterances"], "min_gap_ms": settings["min_gap_ms"],
        "proactive": settings["proactive"], "requests": requests,
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
