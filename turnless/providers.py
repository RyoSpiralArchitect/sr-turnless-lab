"""One backend contract, explicit wire formats, and lossless public-output checks.

No SDKs, hidden retries, fallback providers, or tool execution. Private reasoning
blocks are deliberately excluded from public_response and raw.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .catalog import ADAPTER_VERSION, BACKENDS, constraint_mode
from .protocol import SCHEMA, Settings, _validate_compat_base_url


class ProviderError(RuntimeError):
    pass


@dataclass
class ModelResult:
    raw: str
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    model: str | None = None
    status: str = "completed"
    error: str | None = None
    finish_reason: str | None = None
    public_response: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    def generate(self, instructions: str, payload: str, settings: Settings,
                 cancel: threading.Event) -> ModelResult: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("Provider redirects are disabled; configure the final endpoint explicitly.")


def _post(url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[dict[str, Any], str | None]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=60) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ProviderError("provider response exceeds 2 MB")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ProviderError("provider returned a non-object response")
            return data, response.headers.get("x-request-id") or response.headers.get("request-id")
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", errors="replace")
        for value in headers.values():
            if value:
                detail = detail.replace(value, "[redacted]")
                if value.startswith("Bearer "):
                    detail = detail.replace(value[7:], "[redacted]")
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise ProviderError(f"{type(exc).__name__}: {exc}") from exc


def _public_blocks(blocks: Any) -> list[dict[str, Any]]:
    """Keep every public candidate, never thinking/signatures/tool results."""
    if not isinstance(blocks, list):
        return []
    public = []
    for block in blocks:
        if not isinstance(block, dict):
            public.append({"type": "invalid_block"})
        elif block.get("type") in ("text", "output_text"):
            public.append({"type": block["type"], "text": block.get("text")})
        elif block.get("type") == "refusal":
            public.append({"type": "refusal", "refusal": block.get("refusal")})
        elif block.get("type") == "tool_use":
            public.append({"type": "tool_use", "name": block.get("name"), "input": block.get("input")})
        elif block.get("type") not in ("thinking", "redacted_thinking", "reasoning"):
            public.append({"type": str(block.get("type", "unknown"))})
    return public


def _text(blocks: list[dict[str, Any]]) -> str:
    return "".join(b["text"] for b in blocks if b.get("type") in ("text", "output_text")
                   and isinstance(b.get("text"), str))


def _usage(data: dict[str, Any], input_key: str = "input_tokens",
           output_key: str = "output_tokens") -> dict[str, Any]:
    values = data.get("usage") or {}
    if not isinstance(values, dict):
        return {}
    # Do not invent zero token counts when a provider supplies no measurement.
    return {name: values[key] for name, key in (("input_tokens", input_key), ("output_tokens", output_key))
            if type(values.get(key)) is int and values[key] >= 0}


class HTTPProvider:
    """Adapters expose the exact secret-free wire body for the request receipt."""
    backend = ""

    def describe_request(self, instructions: str, payload: str, settings: Settings) -> dict[str, Any]:
        url, body, _ = self.prepare(instructions, payload, settings)
        return {"adapter_version": ADAPTER_VERSION, "transport": BACKENDS[self.backend].transport,
                "endpoint": url, "body": body,
                "constraint_mode": constraint_mode(self.backend, settings.structured, settings.anthropic_mode)}

    def generate(self, instructions: str, payload: str, settings: Settings,
                 cancel: threading.Event) -> ModelResult:
        if cancel.is_set():
            raise ProviderError("cancelled before HTTP request")
        url, body, headers = self.prepare(instructions, payload, settings)
        data, request_id = _post(url, body, headers)
        result = self.extract(data, settings)
        result.request_id = request_id or (data.get("id") if isinstance(data.get("id"), str) else None)
        result.model = data.get("model")
        result.metadata["adapter_version"] = ADAPTER_VERSION
        return result


class OpenAIProvider(HTTPProvider):
    backend = "openai"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI credentials are not configured on the server.")
        self._key = api_key

    def prepare(self, instructions, payload, settings):
        body = {"model": settings.model, "instructions": instructions, "input": payload,
                "max_output_tokens": settings.max_output_tokens, "store": False}
        if settings.structured:
            body["text"] = {"format": {"type": "json_schema", "name": "utterance_decision",
                                       "strict": True, "schema": SCHEMA}}
        return BACKENDS[self.backend].endpoint, body, {"Authorization": f"Bearer {self._key}"}

    def extract(self, data, settings):
        output = data.get("output")
        items = output if isinstance(output, list) else []
        messages = [i for i in items if isinstance(i, dict) and i.get("type") == "message"]
        blocks = [_public_blocks(m.get("content")) for m in messages]
        raw = "".join(_text(b) for b in blocks)
        public = {"messages": blocks, "status": data.get("status"),
                  "other_output_types": [i.get("type") for i in items if isinstance(i, dict)
                                         and i.get("type") not in ("message", "reasoning")]}
        refusal = next((b.get("refusal") or "provider refusal" for bs in blocks for b in bs
                        if b.get("type") == "refusal"), None)
        error, status = None, "completed"
        if data.get("status") != "completed":
            status, error = "incomplete", "provider status: " + str(data.get("status", "missing"))
        elif refusal:
            status, error = "refused", refusal
        elif (len(messages) != 1 or len(blocks[0]) != 1
              or blocks[0][0].get("type") != "output_text"
              or not isinstance(blocks[0][0].get("text"), str) or public["other_output_types"]):
            status, error = "invalid_response", "expected exactly one public output_text decision"
        return ModelResult(raw, _usage(data), status=status, error=error,
                           finish_reason=data.get("status"), public_response=public)


class OpenAICompatProvider(HTTPProvider):
    backend = "openai_compat"

    def __init__(self, base_url: str, api_key: str = "", backend: str = "openai_compat"):
        _validate_compat_base_url(base_url)
        self.base_url, self._key, self.backend = base_url.rstrip("/"), api_key, backend

    def prepare(self, instructions, payload, settings):
        body = {"model": settings.model,
                "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": payload}],
                "max_tokens": settings.max_output_tokens, "stream": False, "n": 1}
        if settings.structured:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "utterance_decision", "strict": True, "schema": SCHEMA}}
        return self.base_url + "/chat/completions", body, ({"Authorization": f"Bearer {self._key}"} if self._key else {})

    def extract(self, data, settings):
        choices = data.get("choices")
        public = []
        for choice in choices if isinstance(choices, list) else []:
            if not isinstance(choice, dict):
                public.append({"invalid_choice": True})
                continue
            msg = choice.get("message")
            msg = msg if isinstance(msg, dict) else {}
            content = msg.get("content")
            public.append({"finish_reason": choice.get("finish_reason"), "role": msg.get("role"),
                           "content": _public_blocks(content) if isinstance(content, list) else content,
                           "refusal": msg.get("refusal"),
                           "tool_call_count": len(msg.get("tool_calls") or []),
                           "function_call_present": bool(msg.get("function_call"))})
        raw, finish, error, status = "", None, None, "completed"
        if len(public) != 1 or public[0].get("invalid_choice"):
            status, error = "invalid_response", "expected exactly one completion choice"
        else:
            item = public[0]
            content, finish = item["content"], item["finish_reason"]
            raw = content if isinstance(content, str) else _text(content) if isinstance(content, list) else ""
            if finish == "length":
                status, error = "incomplete", "output token limit reached"
            elif finish == "content_filter" or item["refusal"]:
                status, error = "refused", "provider refusal or content filter"
            elif (finish != "stop" or item["role"] != "assistant"
                  or item["tool_call_count"] or item["function_call_present"]):
                status, error = "invalid_response", "unexpected completion role, finish reason, or tool call"
            elif not isinstance(content, str) and not (
                isinstance(content, list) and len(content) == 1 and content[0].get("type") == "text"
                and isinstance(content[0].get("text"), str)
            ):
                status, error = "invalid_response", "expected exactly one public text decision"
        if data.get("error"):
            status, error = "invalid_response", "provider returned an error envelope"
        return ModelResult(raw, _usage(data, "prompt_tokens", "completion_tokens"), status=status,
                           error=error, finish_reason=finish, public_response={"choices": public})


class AnthropicProvider(HTTPProvider):
    backend = "anthropic"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Anthropic credentials are not configured on the server.")
        self._key = api_key

    def prepare(self, instructions, payload, settings):
        body = {"model": settings.model, "system": instructions,
                "messages": [{"role": "user", "content": payload}], "max_tokens": settings.max_output_tokens}
        if settings.structured:
            if settings.anthropic_mode == "json_schema":
                body["output_config"] = {"format": {"type": "json_schema", "schema": SCHEMA}}
            else:
                body["tools"] = [{"name": "utterance_decision", "description": "Return one conversational decision.",
                                  "input_schema": SCHEMA, "strict": True}]
                body["tool_choice"] = {"type": "tool", "name": "utterance_decision", "disable_parallel_tool_use": True}
        return BACKENDS[self.backend].endpoint, body, {"x-api-key": self._key, "anthropic-version": "2023-06-01"}

    def extract(self, data, settings):
        blocks = _public_blocks(data.get("content"))
        stop = data.get("stop_reason")
        raw, status, error = _text(blocks), "completed", None
        tool_mode = settings.structured and settings.anthropic_mode == "tool"
        tools = [b for b in blocks if b.get("type") == "tool_use"]
        # Retain every public candidate when the envelope is invalid. Never pick
        # the last tool, concatenate decisions into one, or silently fall back.
        if tools:
            raw = json.dumps(tools[0]["input"] if len(blocks) == len(tools) == 1 else blocks, ensure_ascii=False)
        if stop == "max_tokens":
            status, error = "incomplete", "output token limit reached"
        elif stop == "refusal":
            status, error = "refused", "provider refusal"
        elif tool_mode:
            if (stop != "tool_use" or len(blocks) != 1 or len(tools) != 1
                    or tools[0].get("name") != "utterance_decision" or not isinstance(tools[0].get("input"), dict)):
                status, error = "invalid_response", "expected exactly one utterance_decision tool and no public text"
        elif (stop != "end_turn" or len(blocks) != 1 or blocks[0].get("type") != "text"
              or not isinstance(blocks[0].get("text"), str)):
            status, error = "invalid_response", "expected exactly one text decision with end_turn"
        return ModelResult(raw, _usage(data), status=status, error=error, finish_reason=stop,
                           public_response={"content": blocks, "stop_reason": stop})


class OllamaProvider(HTTPProvider):
    backend = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")

    def prepare(self, instructions, payload, settings):
        body = {"model": settings.model,
                "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": payload}],
                "stream": False, "options": {"num_predict": settings.max_output_tokens}}
        if settings.structured:
            body["format"] = SCHEMA
        return self.base_url + "/api/chat", body, {}

    def extract(self, data, settings):
        message = data.get("message")
        message = message if isinstance(message, dict) else {}
        raw, stop = message.get("content"), data.get("done_reason")
        status, error = "completed", None
        if stop == "length":
            status, error = "incomplete", "output token limit reached"
        elif not data.get("done") or stop != "stop" or data.get("error") or message.get("tool_calls") or not isinstance(raw, str):
            status, error = "invalid_response", "expected one completed Ollama text response with stop"
        usage = {dest: data[src] for src, dest in (("prompt_eval_count", "input_tokens"), ("eval_count", "output_tokens"))
                 if type(data.get(src)) is int and data[src] >= 0}
        return ModelResult(raw if isinstance(raw, str) else "", usage, status=status, error=error,
                           finish_reason=stop, public_response={"content": raw, "done": data.get("done"), "done_reason": stop})


_LOCAL_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}
_LOCAL_LOCK = threading.Lock()


class LocalHFProvider:
    """Explicit local directories, native chat templates, greedy generation.

    Both loading and inference are serialized across sessions. local_files_only
    and trust_remote_code=False are explicit; HF_HUB_OFFLINE=1 is supported but
    never set globally as a side effect. No automatic download/device fallback.
    """
    def _load(self, path: str, device: str) -> tuple[Any, Any]:
        if not Path(path).is_dir():
            raise ProviderError("Local HF model must be an existing checkpoint directory.")
        try:
            import transformers
        except ImportError as exc:
            raise ProviderError("Install the optional local dependencies: pip install '.[local]'") from exc
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            model = transformers.AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            model.to(device)
            model.eval()
        except Exception as exc:
            raise ProviderError(f"Local model load failed: {type(exc).__name__}: {exc}") from exc
        return tokenizer, model

    def describe_request(self, instructions, payload, settings):
        return {"adapter_version": ADAPTER_VERSION, "transport": "transformers", "constraint_mode": "prompt_only",
                "model_directory": str(Path(settings.model).expanduser().resolve()), "device": settings.local_device,
                "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": payload}],
                "generation": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                               "max_new_tokens": settings.max_output_tokens},
                "local_files_only": True, "trust_remote_code": False}

    def generate(self, instructions: str, payload: str, settings: Settings,
                 cancel: threading.Event) -> ModelResult:
        if settings.structured:
            raise ProviderError("Local HF supports Prompt only.")
        path = str(Path(settings.model).expanduser().resolve())
        cache_key = (path, settings.local_device)
        while not _LOCAL_LOCK.acquire(timeout=0.1):
            if cancel.is_set():
                raise ProviderError("cancelled while waiting for local inference")
        try:
            if cancel.is_set():
                raise ProviderError("cancelled before local load")
            if cache_key not in _LOCAL_MODELS:
                _LOCAL_MODELS[cache_key] = self._load(path, settings.local_device)
            tokenizer, model = _LOCAL_MODELS[cache_key]
            if cancel.is_set():
                raise ProviderError("cancelled before local generation")
            messages = [{"role": "system", "content": instructions}, {"role": "user", "content": payload}]
            try:
                # A template already supplies special tokens. Do not add them twice.
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(settings.local_device)
                output = model.generate(**inputs, max_new_tokens=settings.max_output_tokens,
                                        do_sample=False, num_beams=1, num_return_sequences=1,
                                        return_dict_in_generate=False)
                n_in = int(inputs["input_ids"].shape[1])
                generated = output[0][n_in:]
                n_out = int(output.shape[1] - n_in)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                eos = getattr(model.generation_config, "eos_token_id", None)
                eos_ids = eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else []
                ended = n_out > 0 and int(generated[-1]) in eos_ids
                finish = "eos" if ended else "length" if n_out >= settings.max_output_tokens else "other_stop"
                status = "completed" if ended else "incomplete"
                template = tokenizer.get_chat_template()
                metadata = {"adapter_version": ADAPTER_VERSION,
                            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                            "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "input_token_ids_sha256": hashlib.sha256(json.dumps(inputs["input_ids"].tolist()).encode()).hexdigest(),
                            "device": str(model.device), "eos_token_ids": eos_ids,
                            "generation_config": model.generation_config.to_dict()}
                metadata["libraries"] = {}
                for name in ("transformers", "torch", "tokenizers"):
                    try:
                        metadata["libraries"][name] = importlib.metadata.version(name)
                    except importlib.metadata.PackageNotFoundError:
                        metadata["libraries"][name] = None
            except Exception as exc:
                raise ProviderError(f"Local inference failed: {type(exc).__name__}: {exc}") from exc
            return ModelResult(text, {"input_tokens": n_in, "output_tokens": n_out}, model=path,
                               status=status, error=None if ended else "local generation stopped without EOS: " + finish,
                               finish_reason=finish, metadata=metadata)
        finally:
            _LOCAL_LOCK.release()


def create_provider(settings: Settings, credentials: dict[str, str] | None = None,
                    ollama_url: str = "http://127.0.0.1:11434") -> Provider:
    """The only backend-construction switch; all consumers use this factory."""
    keys = credentials or {}
    spec = BACKENDS[settings.provider]
    key = keys.get(settings.provider, "")
    if settings.provider == "fixture":
        return FixtureProvider()
    if settings.provider == "local":
        return LocalHFProvider()
    if settings.provider == "ollama":
        return OllamaProvider(ollama_url)
    if settings.provider == "openai":
        return OpenAIProvider(key)
    if settings.provider == "anthropic":
        return AnthropicProvider(key)
    if settings.provider in ("gemini", "mistral"):
        if not key:
            raise ValueError(f"{spec.label} credentials are not configured on the server ({spec.key_env}).")
        return OpenAICompatProvider(spec.endpoint.removesuffix("/chat/completions"), key, spec.id)
    return OpenAICompatProvider(settings.compat_base_url, key)


class FixtureProvider:

    """Deterministic I/O fixture, NOT an LLM and NOT evidence of learned behavior."""
    def generate(self, instructions: str, payload: str, settings: Settings,
                 cancel: threading.Event) -> ModelResult:
        cancel.wait(0.28)
        state = json.loads(payload)
        n = state["utterances_in_this_cycle"]
        humans = [e for e in state["public_events"] if e["speaker"] == "human"]
        case = settings.fixture_case
        action, text, delay = "emit", "", 0
        if case == "malformed":
            return ModelResult('```json\n{"action":"emit","text":"broken"}\n```', model="fixture-v1")
        if case == "fake-user":
            return ModelResult('{"action":"human","text":"勝手に人間の返事を作った","delay_ms":0}', model="fixture-v1")
        if case == "immediate-yield":
            action = "yield"
        elif case == "overtalk":
            text = f"これは上限停止を確かめる固定デモの発話 {n + 1}。"
        elif case == "redundant" and n < 3:
            text = "同じことを、もう一度言っている。"
        elif case == "redundant":
            action = "yield"
        elif humans:
            last = humans[-1]["text"]
            if n == 0:
                text = "受け取った。ここからは、今の入力を含む新しい履歴で始める。"
            elif n == 1:
                text = "これが二つ目の発話。次は yield を返して、待つ。"
                delay = 500
            else:
                action = "yield"
            if any(w in last.casefold() for w in ("黙って", "待って", "止まって", "wait", "stop", "be quiet")):
                action, text = "yield", ""
        elif n == 0:
            text = "ねえ。まだ誰も話していないけど、先にひとつ。"
        elif n == 1:
            text = "発話を閉じても、会話を手渡すとは限らない。"
            delay = 600
        elif n == 2:
            text = "……ここでは、待つことも選べる。"
            delay = 1000
        else:
            action = "yield"
        if settings.conversation_language == "en":
            translations = {
                "同じことを、もう一度言っている。": "I am saying the same thing again.",
                "受け取った。ここからは、今の入力を含む新しい履歴で始める。":
                    "Received. I am starting again with your message in the public history.",
                "これが二つ目の発話。次は yield を返して、待つ。":
                    "This is the second utterance. Next I will yield and wait.",
                "ねえ。まだ誰も話していないけど、先にひとつ。":
                    "Hello. Nobody has spoken yet, so I will leave one thought here.",
                "発話を閉じても、会話を手渡すとは限らない。":
                    "Ending an utterance does not have to hand over the conversation.",
                "……ここでは、待つことも選べる。":
                    "…Here, waiting is a choice, too.",
            }
            text = translations.get(text, text)
            if case == "overtalk":
                text = f"Scripted cap-stop check: utterance {n + 1}."
        return ModelResult(json.dumps({"action": action, "text": text, "delay_ms": delay},
                                      ensure_ascii=False), model="fixture-v1")
