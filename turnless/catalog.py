"""Public backend capabilities; credentials never belong in experiment settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass

ADAPTER_VERSION = "2"


@dataclass(frozen=True)
class Backend:
    id: str
    label: str
    transport: str
    schema_mode: str
    endpoint: str = ""
    key_env: str = ""
    model_kind: str = "id"

    def public(self) -> dict:
        return {**asdict(self), "structured_supported": self.schema_mode != "unsupported"}


BACKENDS = {
    b.id: b for b in (
        Backend("fixture", "Scripted demo", "fixture", "fixture_noop"),
        Backend("openai", "OpenAI Responses", "responses", "responses_json_schema",
                "https://api.openai.com/v1/responses", "OPENAI_API_KEY"),
        Backend("anthropic", "Anthropic Claude", "messages", "anthropic_json_schema",
                "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY"),
        Backend("gemini", "Gemini · Chat Completions", "chat_completions", "chat_json_schema",
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                "GEMINI_API_KEY"),
        Backend("mistral", "Mistral · Chat Completions", "chat_completions", "chat_json_schema",
                "https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY"),
        Backend("openai_compat", "Custom · Chat Completions", "chat_completions", "chat_json_schema",
                key_env="OPENAI_COMPAT_API_KEY"),
        Backend("ollama", "Ollama · local", "ollama", "ollama_format"),
        Backend("local", "Hugging Face · offline directory", "transformers", "unsupported",
                model_kind="directory"),
    )
}


def constraint_mode(provider: str, structured: bool, anthropic_mode: str = "json_schema") -> str:
    if not structured:
        return "prompt_only"
    if provider == "anthropic" and anthropic_mode == "tool":
        return "anthropic_forced_tool"
    return BACKENDS[provider].schema_mode
