"""Offline checkpoint inspection and explicit local prompt/output conventions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

PLAIN_PROMPT_VERSION = "1"


def checkpoint_config(path: str) -> dict:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Local model must be an existing checkpoint directory; Hub IDs are not downloaded.")
    try:
        config = json.loads((directory / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("The checkpoint needs a readable config.json.") from exc
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object.")
    return config


def resolve_runtime(config: dict, requested: str) -> str:
    quant = config.get("quantization_config") or config.get("quantization") or {}
    mlx_format = (isinstance(quant, dict) and "quant_method" not in quant
                  and "bits" in quant and "group_size" in quant)
    if requested == "transformers" and mlx_format:
        raise ValueError("This is an MLX quantized checkpoint. Select Auto or MLX, not Transformers/MPS.")
    return ("mlx" if mlx_format else "transformers") if requested == "auto" else requested


def context_window(config: dict) -> int | None:
    limits = [config.get(key) for key in ("max_position_embeddings", "n_positions", "n_ctx")]
    return min((v for v in limits if type(v) is int and 0 < v < 10**9), default=None)


def inspect_checkpoint(path: str, runtime: str = "auto", prompt_mode: str = "auto") -> dict:
    if not isinstance(path, str) or not path.strip() or len(path) > 1024:
        raise ValueError("Specify a local checkpoint directory.")
    if runtime not in ("auto", "transformers", "mlx") or prompt_mode not in ("auto", "native", "plain"):
        raise ValueError("Invalid local runtime or prompt mode.")
    config = checkpoint_config(path)
    directory = Path(path).expanduser().resolve()
    tokenizer_file = directory / "tokenizer_config.json"
    try:
        tokenizer_config = json.loads(tokenizer_file.read_text()) if tokenizer_file.is_file() else {}
        native = bool(tokenizer_config.get("chat_template")) or (directory / "chat_template.jinja").is_file()
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError("Cannot read tokenizer_config.json.") from exc
    if prompt_mode == "native" and not native:
        raise ValueError("No native chat template. Select Auto or Plain completion for a base model.")
    resolved = resolve_runtime(config, runtime)
    library = "mlx-lm" if resolved == "mlx" else "transformers"
    try:
        version = importlib.metadata.version(library)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"runtime": resolved, "prompt_mode": ("native" if native else "plain") if prompt_mode == "auto" else prompt_mode,
            "native_chat_template": native, "model_type": config.get("model_type"),
            "context_window": context_window(config), "library": library, "library_version": version,
            "weights_loaded": False}


def render_prompt(tokenizer: Any, instructions: str, payload: str, mode: str) -> tuple[str, dict]:
    template = None
    if mode != "plain":
        try:
            template = tokenizer.get_chat_template()
        except ValueError:
            if getattr(tokenizer, "chat_template", None):
                raise ValueError("The native chat template is ambiguous or unsupported. Select Plain explicitly, "
                                 "or use a checkpoint with a supported default template.")
    if mode == "native" and not template:
        raise ValueError("No native chat template. Select Auto or Plain completion for a base model.")
    resolved = ("native" if template else "plain") if mode == "auto" else mode
    if resolved == "native":
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": instructions}, {"role": "user", "content": payload}],
            add_generation_prompt=True, tokenize=False)
    else:
        prompt = instructions + "\n\nHarness state:\n" + payload + "\n\nAssistant decision:\n"
    return prompt, {"prompt_mode": resolved,
                    "plain_prompt_version": PLAIN_PROMPT_VERSION if resolved == "plain" else None,
                    "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest() if resolved == "native" else None,
                    "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def output_allowance(config: dict, n_input: int, requested: int) -> tuple[int, dict]:
    limit = context_window(config)
    available = requested if limit is None else min(requested, limit - n_input)
    if available <= 0:
        raise ValueError(f"Prompt has {n_input} tokens but the context window is {limit}. "
                         "Start a shorter experiment or reduce the context; no input was truncated.")
    return available, {"context_window": limit, "requested_max_new_tokens": requested,
                       "effective_max_new_tokens": available, "context_policy": "cap_output_only"}


def public_output(text: str, model_type: str | None) -> tuple[str, dict]:
    """Extract public final content before logging; never persist private channels.

    gpt-oss checkpoints use either the Harmony names or their token aliases.
    A missing/ambiguous final channel fails closed, without trying to repair JSON.
    """
    if model_type == "gpt_oss":
        finals = list(re.finditer(
            r"(?:\A|<\|(?:start|im_start)\|>assistant)<\|(?:channel|meta_sep)\|>final<\|(?:message|im_sep)\|>", text))
        metadata = {"output_extraction": "harmony_final_v1", "private_reasoning_omitted": True}
        if len(finals) != 1:
            raise ValueError("Expected exactly one public final channel from gpt-oss; private channels were omitted.")
        tail = text[finals[0].end():]
        # No second role, tool call, or channel may be smuggled into the decision.
        match = re.search(r"<\|[^>]*\|>", tail)
        if match:
            if match.group() not in ("<|return|>", "<|fim_suffix|>", "<|end|>", "<|im_end|>"):
                raise ValueError("Unexpected control token in public final channel; output omitted.")
            if tail[match.end():].strip():
                raise ValueError("Multiple messages after the public final channel; output omitted.")
            tail = tail[:match.start()]
        return tail, metadata
    if "<think>" in text or "</think>" in text:
        # Conventional leading reasoning block, not a general-purpose markup repair.
        match = re.fullmatch(r"\s*<think>.*?</think>\s*(.*)", text, flags=re.DOTALL)
        if not match or "<think>" in match[1] or "</think>" in match[1]:
            raise ValueError("Unclosed or ambiguous private reasoning block; output omitted.")
        return match[1], {"output_extraction": "public_after_think_v1", "private_reasoning_omitted": True}
    return text, {"output_extraction": "text"}
