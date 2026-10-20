"""Observer judgments with explicit targets; never part of model input."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

ANNOTATION_VERSION = "1"
TAGS = (
    "unnecessary-fragmentation", "redundant-continuation", "overtalk",
    "premature-yield", "fake-user", "monologue-collapse", "turn-reversion",
    "interruption-failure", "good-continuation", "good-silence", "other",
)
TERMINALS = {"yield", "forced_stop"}
OBSERVABLE = {"utterance", "yield", "forced_stop", "interrupt", "protocol_error", "provider_error"}
ASSISTANT_TAGS = {
    "redundant-continuation", "overtalk", "fake-user", "monologue-collapse",
    "turn-reversion", "interruption-failure", "good-continuation",
}


def observation_targets(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Freeze sequences at their terminal event, excluding requests/raw/human text.

    An interrupted cycle has no model terminal; its individual public events and
    interrupt remain reviewable. A forced stop never qualifies as model silence.
    """
    targets = []
    assistant_by_epoch: dict[int, list[int]] = defaultdict(list)
    for event in events:
        kind, seq = event.get("kind"), event.get("seq")
        if type(seq) is not int or kind not in OBSERVABLE:
            continue
        assistant = kind == "utterance" and event.get("speaker") == "assistant"
        epoch = event.get("epoch")
        if assistant and type(epoch) is int:
            assistant_by_epoch[epoch].append(seq)
        allowed = {"other"}
        if assistant:
            allowed |= ASSISTANT_TAGS
        if kind == "yield":
            allowed |= {"premature-yield", "good-silence"}
        if kind == "interrupt":
            allowed.add("interruption-failure")
        targets.append({"target_seq": seq, "scope": "event", "target_seqs": [seq],
                        "allowed_tags": [tag for tag in TAGS if tag in allowed]})
        if kind in TERMINALS and type(epoch) is int:
            utterances = assistant_by_epoch[epoch]
            if utterances:
                allowed |= ASSISTANT_TAGS
            if len(utterances) >= 2:
                allowed.add("unnecessary-fragmentation")
            targets.append({"target_seq": seq, "scope": "cycle", "target_seqs": [*utterances, seq],
                            "allowed_tags": [tag for tag in TAGS if tag in allowed]})
    return targets


def annotation_payload(events: list[dict[str, Any]], target_seq: int, tag: str | None,
                       note: str = "", *, scope: str = "event", judgment: str = "labeled",
                       supersedes: int | None = None) -> dict[str, Any]:
    if type(target_seq) is not int or scope not in ("event", "cycle"):
        raise ValueError("Select a public event or a completed sequence to observe.")
    target = next((item for item in observation_targets(events)
                   if item["target_seq"] == target_seq and item["scope"] == scope), None)
    if target is None:
        raise ValueError("Select a public event or a completed sequence to observe.")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("Observation notes must be at most 2000 characters.")
    if judgment == "labeled":
        if not isinstance(tag, str) or tag not in target["allowed_tags"]:
            raise ValueError("Choose a label that applies to the selected target.")
    elif judgment in ("unsure", "note"):
        if tag is not None:
            raise ValueError("An unsure or note-only observation cannot carry a label.")
        if judgment == "note" and not note.strip():
            raise ValueError("A note-only observation needs a note.")
    else:
        raise ValueError("Choose a label, unsure, or note-only.")
    if supersedes is not None:
        if type(supersedes) is not int:
            raise ValueError("Invalid observation revision.")
        previous = next((e for e in events if e.get("seq") == supersedes and e.get("kind") == "annotation"), None)
        if (previous is None or previous.get("target_seq") != target_seq
                or previous.get("scope", "event") != scope
                or any(e.get("kind") == "annotation" and e.get("supersedes") == supersedes for e in events)):
            raise ValueError("The observation changed or has a different target. Refresh before revising.")
    return {"annotation_version": ANNOTATION_VERSION,
            "target_seq": target_seq, "scope": scope, "target_seqs": target["target_seqs"],
            "judgment": judgment, "tag": tag, "note": note, "supersedes": supersedes}


def annotation_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Report legacy target problems without guessing new labels or rewriting data."""
    annotations = [e for e in events if e.get("kind") == "annotation"]
    superseded: set[int] = set()
    valid = []
    issues = []
    groups: Counter = Counter()
    for event in annotations:
        groups[(event.get("target_seq"), str(event.get("scope", "event")))] += 1
        # Validate against the prefix actually available when the observer saved.
        prefix = [e for e in events if type(e.get("seq")) is int and e["seq"] < event["seq"]]
        try:
            payload = annotation_payload(prefix, event.get("target_seq"), event.get("tag"),
                                         event.get("note", ""), scope=event.get("scope", "event"),
                                         judgment=event.get("judgment", "labeled"),
                                         supersedes=event.get("supersedes"))
            if event.get("annotation_version") is not None:
                if (event["annotation_version"] != ANNOTATION_VERSION
                        or event.get("target_seqs") != payload["target_seqs"]):
                    raise ValueError("Unknown annotation version or mismatched target sequence.")
            valid.append(event)
            if payload["supersedes"] is not None:
                superseded.add(payload["supersedes"])
        except (ValueError, TypeError) as exc:
            issues.append({"annotation_seq": event.get("seq"), "target_seq": event.get("target_seq"),
                           "reason": str(exc)})
    active = [e for e in valid if e.get("seq") not in superseded]
    return {
        "raw_events": len(annotations),
        "legacy_events": sum("annotation_version" not in e for e in annotations),
        "superseded_events": len(superseded),
        "active_labeled": dict(Counter(e["tag"] for e in active if e.get("judgment", "labeled") == "labeled")),
        "active_unsure": sum(e.get("judgment") == "unsure" for e in active),
        "active_note_only": sum(e.get("judgment") == "note" for e in active),
        "target_issues": issues,
        "repeated_targets_including_revisions": [
            {"target_seq": seq, "scope": scope, "observations": n}
            for (seq, scope), n in groups.items() if n > 1
        ],
        "not_adjudicated_ground_truth": True,
    }
