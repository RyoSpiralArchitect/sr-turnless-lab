"""Application-level utterance protocol, not native tokenizer control tokens."""
from __future__ import annotations

import json
import urllib.parse
from dataclasses import asdict, dataclass
from typing import Any

from .catalog import BACKENDS
from .annotations import TAGS

PROVIDERS = tuple(BACKENDS)
PROTOCOL_VERSION = "2"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["emit", "yield"]},
        "text": {"type": "string"},
        "delay_ms": {"type": "integer"},
    },
    "required": ["action", "text", "delay_ms"],
    "additionalProperties": False,
}
BASE_PROMPT = """You are the assistant in a live, non-alternating conversation experiment.
The actual human and assistant events are supplied in an event log. Consecutive
assistant utterances and consecutive human utterances are both permitted.
You produce ONE decision, not a transcript and not a batch of future messages:
{"action":"emit","text":"one public utterance","delay_ms":0}
or {"action":"yield","text":"","delay_ms":0}.
EMIT ends one public utterance. It does not hand control to the human. You will
be invoked again with that utterance in the real event log. YIELD means wait;
there is no autonomous reawakening after yield in this experiment.
You may emit or yield immediately. Do not manufacture a sequence merely because
multiple utterances are permitted. Do not repeat an utterance just to continue.
Do not generate the human's role, words, consent, actions, thoughts, or replies.
Do not reveal private chain-of-thought. All emitted text is public conversation,
not a simulation of hidden reasoning. Never claim to observe an event not logged.
The session_open trigger may precede any human utterance. This is an opportunity
to speak first, not an instruction to pretend a human has already spoken.
The operator_wake trigger is a harness event, NOT a new human message.
Use the supplied conversational context and the configured conversation language.
Return ONLY the JSON object, without Markdown fences, commentary, or extra keys.
Delay is in milliseconds after the decision is received, between 0 and 5000.
When yielding, text must be empty and delay_ms must be 0.
"""

POLICIES = {
    "neutral": "No preference for continuation or yielding. Choose based on the conversation.",
    "sparse": "Prefer yielding when nothing useful remains. Do not fill silence for its own sake.",
    "exploratory": "A distinct, relevant follow-up is welcome, but never obligatory. Yield rather than pad.",
}


class ProtocolError(ValueError):
    """A malformed decision. Raw output is retained; no silent repair is attempted."""


def _validate_compat_base_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlparse(url)
        parsed.port
    except ValueError as exc:
        raise ValueError("Invalid endpoint URL or port.") from exc
    loopback = parsed.hostname in ("127.0.0.1", "localhost", "::1")
    clean = parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment
    if clean and ((parsed.scheme == "https" and parsed.hostname) or
                  (parsed.scheme == "http" and loopback)):
        return
    raise ValueError("compat_base_url must use HTTPS, or HTTP on a loopback host.")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON constant: {value}")


@dataclass(frozen=True)
class Decision:
    action: str
    text: str
    delay_ms: int


def parse_decision(raw: str) -> Decision:
    if len(raw) > 40_000:
        raise ProtocolError("output exceeds 40000 characters")
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_bad_constant)
    except (ValueError, TypeError) as exc:
        raise ProtocolError(str(exc)) from exc
    if not isinstance(obj, dict) or set(obj) != {"action", "text", "delay_ms"}:
        raise ProtocolError("expected exactly action, text, delay_ms")
    if obj["action"] not in ("emit", "yield"):
        raise ProtocolError("action must be emit or yield; human generation is not allowed")
    if not isinstance(obj["text"], str):
        raise ProtocolError("text must be a string")
    if type(obj["delay_ms"]) is not int or not 0 <= obj["delay_ms"] <= 5000:
        raise ProtocolError("delay_ms must be an integer from 0 to 5000")
    if obj["action"] == "emit" and not obj["text"].strip():
        raise ProtocolError("emit must contain nonempty text")
    if len(obj["text"]) > 8000:
        raise ProtocolError("utterance exceeds 8000 characters")
    if obj["action"] == "yield" and (obj["text"] != "" or obj["delay_ms"] != 0):
        raise ProtocolError("yield requires empty text and zero delay")
    return Decision(**obj)


@dataclass(frozen=True)
class Settings:
    provider: str = "fixture"
    model: str = ""
    structured: bool = False
    policy: str = "neutral"
    max_utterances: int = 5
    min_gap_ms: int = 650
    call_budget: int = 40
    max_output_tokens: int = 1024
    proactive: bool = True
    fixture_case: str = "normal"
    context: str = "We are exploring a conversation without a fixed speaking order."
    prompt_extra: str = ""
    compat_base_url: str = ""
    consent: bool = False
    conversation_language: str = "en"
    anthropic_mode: str = "json_schema"
    local_device: str = "cpu"
    local_runtime: str = "auto"
    local_prompt_mode: str = "auto"
    typing_mode: str = "interrupt"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        if not isinstance(data, dict):
            raise ValueError("settings must be an object")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown settings: " + ", ".join(sorted(unknown)))
        obj = cls(**data)
        if not isinstance(obj.provider, str) or obj.provider not in PROVIDERS:
            raise ValueError("unsupported provider")
        if obj.provider == "local" and obj.structured:
            raise ValueError("Local HF models support Prompt only, not JSON Schema constraints.")
        if not isinstance(obj.policy, str) or obj.policy not in POLICIES:
            raise ValueError("unsupported policy")
        if obj.conversation_language not in ("en", "ja"):
            raise ValueError("conversation_language must be en or ja")
        if obj.anthropic_mode not in ("json_schema", "tool"):
            raise ValueError("anthropic_mode must be json_schema or tool")
        if obj.local_device not in ("cpu", "mps", "cuda"):
            raise ValueError("local_device must be cpu, mps or cuda")
        if obj.local_runtime not in ("auto", "transformers", "mlx"):
            raise ValueError("local_runtime must be auto, transformers or mlx")
        if obj.local_prompt_mode not in ("auto", "native", "plain"):
            raise ValueError("local_prompt_mode must be auto, native or plain")
        if obj.typing_mode not in ("interrupt", "presence"):
            raise ValueError("typing_mode must be interrupt or presence")
        if obj.fixture_case not in ("normal", "redundant", "immediate-yield", "malformed", "fake-user", "overtalk"):
            raise ValueError("unsupported fixture case")
        for name, lo, hi in (("max_utterances", 1, 12), ("min_gap_ms", 0, 3000),
                             ("call_budget", 1, 200), ("max_output_tokens", 128, 4096)):
            value = getattr(obj, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer between {lo} and {hi}")
        for name in ("structured", "proactive", "consent"):
            if type(getattr(obj, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        model_limit = 1024 if obj.provider == "local" else 160
        for name, limit in (("model", model_limit), ("context", 4000), ("prompt_extra", 4000),
                            ("compat_base_url", 2000)):
            value = getattr(obj, name)
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"invalid {name}")
        if obj.provider != "fixture" and not obj.model.strip():
            raise ValueError("Specify a local model directory or an exact API model ID.")
        if obj.provider != "fixture" and not obj.consent:
            raise ValueError("Enable consent to send the conversation to the selected model.")
        if obj.provider == "openai_compat":
            _validate_compat_base_url(obj.compat_base_url)
        elif obj.compat_base_url:
            raise ValueError("compat_base_url is only used by the custom Chat Completions backend.")
        return obj

    def public(self) -> dict[str, Any]:
        return asdict(self)


def make_prompt(settings: Settings, events: list[dict[str, Any]], trigger: str,
                emitted: int, presence: dict[str, Any] | None = None) -> tuple[str, str]:
    # Only real public utterances are represented as speakers. Control events are
    # explicitly separate. Full prompt snapshots are traced for reproducibility.
    transcript = [
        {"seq": e["seq"], "speaker": e["speaker"], "text": e["text"], "at_ms": e["at_ms"]}
        for e in events if e["kind"] == "utterance"
    ]
    controls = [
        {"seq": e["seq"], "kind": e["kind"], "at_ms": e["at_ms"], "reason": e.get("reason", "")}
        for e in events if e["kind"] in ("interrupt", "yield", "forced_stop", "wake")
    ]
    instructions = BASE_PROMPT + "\nPolicy: " + POLICIES[settings.policy]
    instructions += "\nConversation language: " + {"en": "English", "ja": "Japanese"}[settings.conversation_language]
    if settings.prompt_extra.strip():
        instructions += "\nExperimenter's additional instruction:\n" + settings.prompt_extra
    if settings.typing_mode == "presence":
        instructions += (
            "\nPresence is a harness observation at request time, not a human utterance."
            " human_typing means recent input activity; the unsent draft is private."
            " It does not require yielding. Choose whether to emit or yield."
            " Presence changes do not wake you after yield and cannot update an in-flight request.")
    payload = {
        "source": "harness_state_not_a_human_message",
        "context": settings.context,
        "trigger": trigger,
        "utterances_in_this_cycle": emitted,
        "public_events": transcript,
        "control_events": controls[-24:],
        "typing_mode": settings.typing_mode,
    }
    if settings.typing_mode == "presence":
        payload["presence"] = presence if presence is not None else {"human_typing": False}
    return instructions, json.dumps(payload, ensure_ascii=False)
