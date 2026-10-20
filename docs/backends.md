# Backend contracts

`catalog.BACKENDS` is the shared capability catalog. `create_provider()` is the single construction boundary used by the server. Gemini and Mistral use the Chat Completions adapter with distinct fixed endpoints and credential slots. Custom compatible servers use the same adapter with an explicitly selected base URL. Local HF implements the same inference contract without HTTP.

## Result acceptance

Every provider returns `ModelResult`: public `raw`, normalized usage, model/request identifiers, normalized status, native `finish_reason`, sanitized `public_response`, and adapter-specific metadata. Only `status="completed"` with no error reaches the common `parse_decision()` validator. Schema constraints do not bypass that validator.

| Transport | Accepted completed envelope |
| --- | --- |
| Responses | Completed response, exactly one message and one `output_text` block; no other public actions |
| Chat Completions | Exactly one assistant choice, `finish_reason="stop"`, one text candidate; no refusal or tool/function call |
| Claude native/prompt | `end_turn`, exactly one public text block; no tool candidates |
| Claude forced tool | `tool_use`, exactly one `utterance_decision` with object input and no accompanying public text |
| Ollama | `done=true`, `done_reason="stop"`, string content, no tool calls |
| Local HF | One greedy sequence ending in the configured EOS; no schema constraints |

`incomplete`, `refused`, and `invalid_response` retain public evidence and stop the cycle. There are no implicit repairs, tool execution, endpoint fallback, or retries. Multiple candidates are retained in the public envelope rather than silently selecting one. Thinking/reasoning and signatures are excluded.

`raw` means the public candidate, not a raw HTTP capture. Forced-tool input is already parsed by the HTTP JSON decoder and serialized for the common parser; it is identified by `constraint_mode`. This distinction matters for studies of lexical JSON behavior.

## Reproducibility

The session receipt freezes source hashes, protocol/adapter versions and settings. HTTP request events contain the exact body, endpoint and constraint method, but no headers or credentials. Local events describe the role messages and explicit generation overrides; results include native template/rendered-prompt/input-token hashes and the model's generation configuration.

Hosted sampling uses provider defaults unless the request body says otherwise. Missing usage stays missing in the result; the UI totals only reported tokens. Provider defaults, model aliases, native templates, library versions, reasoning token allocation and latency can all affect results. Do not pool transports or constraint methods as identical conditions.

Local models use explicit CPU/MPS/CUDA selection, greedy decoding, one beam, one return sequence, and offline directory loading. The cache is per directory/device and lasts for the process. These receipts do not fingerprint the complete model weight files; use externally pinned immutable checkpoints for research comparisons.

## v0.1 to v0.2

- Legacy traces are readable and retain their original bytes. Missing fields are marked `legacy_unspecified` in summaries.
- New sessions default to English conversation; UI locale is independent.
- Claude schema mode now defaults to native `output_config.format`. Select `anthropic_mode="tool"` to study the legacy transport explicitly. The new tool path also adds strict schema requests and disables parallel tool use, so it is not an identical replication of v0.1.
- Unknown or ambiguous termination now stops at the provider boundary.
- `local` remains the provider ID for existing local-directory integrations.

## Official specifications consulted

Checked for this implementation on 2026-10-20. Supported models and provider behavior may change.

- [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai): Chat Completions base URL and structured output support.
- [Mistral custom structured outputs](https://docs.mistral.ai/studio/conversations/structured-output/custom): schema-guided chat output.
- [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs): native `output_config.format` and strict tools.
- [Hugging Face chat templates](https://huggingface.co/docs/transformers/chat_templating): avoid adding special tokens a second time after rendering a template.
- [Hugging Face offline mode](https://huggingface.co/docs/transformers/installation#offline-mode): `HF_HUB_OFFLINE=1` and local loading.
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) and [Ollama chat](https://docs.ollama.com/api/chat): original adapter references; hosted execution remains separately unverified.
