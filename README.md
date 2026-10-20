# SR Turnless Lab ∴

**An utterance can end before the conversation is handed over.**

[日本語](README.ja.md) · [Experiment guide](docs/experiments.en.md) · [Backend contracts](docs/backends.md)

A small local laboratory for conversational boundaries. A model produces one public utterance (`emit`) or chooses to wait (`yield`). After each emitted utterance, it sees the updated public history and makes a fresh decision. Humans can interrupt or send consecutive messages, too.

This is an **application-level protocol**, not a modification of model weights, tokenizers, native chat templates, or EOS semantics. `emit` and `yield` are JSON values, not special tokens. There is no tool execution, background thinking, timer-driven reawakening, or fine-tuning pipeline.

## Start in three minutes

Python 3.10+. The scripted demo and HTTP backends require only the standard library. No build, npm, or API key is needed for the demo.

```sh
git clone https://github.com/RyoSpiralArchitect/sr-turnless-lab.git
cd sr-turnless-lab
python3 -m turnless
```

Open **http://127.0.0.1:8765**. Windows: `py -3 -m turnless` or `run.bat`. macOS/Linux: `./run.sh`.

1. Leave **Scripted demo** selected and start an experiment. It produces three consecutive utterances, then yields.
2. Press **Resume**, then start typing to invalidate an unpublished response. Text is sent to the model only when you submit it.
3. Disable **Wake model after sending** to record consecutive human messages.
4. Select an utterance or trace event, add an observation, and export JSONL.

**The demo is a deterministic fixture, not evidence of LLM behavior.** Its repetition, malformed JSON, fabricated-role, and cap-stop cases are deliberately injected failures.

English is the default UI. Use the header selector for Japanese; the choice persists in that browser. **Conversation language is a separate experiment setting**, under Context, prompt & budget. Switching UI language does not change prompts, stored utterances, or the active experiment.

```sh
python3 -m turnless --port 8766 --log-dir ./runs
python3 -m unittest discover -s tests -v
python3 -m turnless.analyze "examples/*.jsonl"
```

Optional package installation: `python3 -m pip install .`. Both `sr-turnless-lab` and `turnless-lab` launch the server.

## One interface, explicit backend differences

All backends use the same settings, request receipts, result contract, protocol parser, interruption engine, and observation UI. Transport and format-constraint methods remain visible experimental conditions.

| Backend | Connection | Structured mode | Server credential |
| --- | --- | --- | --- |
| OpenAI | Responses | `text.format` | `OPENAI_API_KEY` |
| Anthropic Claude | Messages | Native `output_config.format`; explicit forced-tool alternative | `ANTHROPIC_API_KEY` |
| Gemini | Fixed Chat Completions compatibility endpoint | `response_format` | `GEMINI_API_KEY` |
| Mistral | Fixed Chat Completions endpoint | `response_format` | `MISTRAL_API_KEY` |
| Custom compatible server | User-selected Chat Completions base URL | `response_format`, if supported | `OPENAI_COMPAT_API_KEY`, optional for local servers |
| Ollama | Loopback `/api/chat` | `format` | None |
| Local Hugging Face | Checkpoint directory + native chat template | Prompt only | None |

Enter an **exact model ID you can access**, or a local checkpoint directory. Schema support depends on the selected model and server. Unsupported requests fail visibly; there are no silent fallbacks or retries. The Gemini and Mistral presets have fixed endpoints and separate credentials. Custom endpoints accept HTTPS or loopback HTTP.

Credentials remain server-side and are never part of settings, browser responses, or request receipts. Use your secure credential-management workflow. In Codex, OpenAI credentials can be provisioned through the secure OpenAI Platform flow. Existing server-managed credentials use the environment names above. Credential files outside this repository can also be selected with `--key-file`, `--anthropic-key-file`, `--gemini-key-file`, `--mistral-key-file`, or `--compat-key-file`. Restart the server after changing its credential configuration.

The custom endpoint default can be selected with `--compat-base-url`. Ollama defaults to `http://127.0.0.1:11434`; `OLLAMA_HOST` may select another loopback HTTP endpoint. Redirects are rejected. A single custom credential is used for the custom endpoint, so select it for the intended destination.

### Offline Hugging Face checkpoint

Install the optional dependencies only on the machine doing local inference:

```sh
python3 -m pip install '.[local]'
HF_HUB_OFFLINE=1 python3 -m turnless
```

Choose **Hugging Face · offline directory**, enter an existing checkpoint directory, and select CPU, MPS, or CUDA. On PowerShell, set `$env:HF_HUB_OFFLINE = "1"` before starting Python.

The loader uses `local_files_only=True` and `trust_remote_code=False`. It does not download Hub model IDs or install templates. A checkpoint needs a compatible native chat template; unsupported roles/templates fail explicitly. Template rendering is followed by tokenization with `add_special_tokens=False`. Generation is greedy, with one beam and one sequence. No device fallback occurs.

Local loading and generation are serialized across sessions. Models are cached by resolved directory and device until the process exits; restart after changing a checkpoint in place or to release cached models. Cancellation is checked before loading/generation and while waiting for the local lock. Running generation is not interrupted; its stale output is discarded by the engine. EOS, token-limit stops, and other stops are recorded separately. Greedy decoding does not guarantee bitwise reproducibility across devices or library versions.

### Output constraints

**Prompt only** requests JSON in the prompt. Malformed output is retained and stops the cycle. Fences are not removed and outputs are not repaired automatically.

**JSON Schema** adds the backend's explicit constraint mechanism. Claude uses native JSON outputs by default. Its forced-tool alternative is a separately recorded condition; exactly one correctly named tool decision and no accompanying public text are accepted. Tools are never executed. Multiple choices, multiple public decision blocks, unexpected tools, refusals, unknown stop reasons, and incomplete outputs cannot become public utterances or model yields.

## Protocol and interruption

```json
{"action":"emit","text":"One thought to leave here.","delay_ms":0}
```

```json
{"action":"yield","text":"","delay_ms":0}
```

One inference produces one decision. `emit` publishes an utterance and requests the next decision using the public history. `yield` waits for human input or an explicit resume. Starting an experiment is an external trigger, not evidence of autonomous reawakening.

`delay_ms` is a wait **after inference returns**, from 0 to 5000 ms. After the first utterance, the applied wait is the larger of the model's proposal and the configured minimum gap. Both values are recorded.

Typing, submitting a message, and stopping invalidate the current epoch. Late results are retained as discarded output and never published. Already published utterances remain in history. One inference worker runs per session; pending wake requests are coalesced to the latest history.

Upstream inference cancellation and billing cessation are **not guaranteed**. A synchronous HTTP call may delay the next inference until it returns or times out. HTTP I/O timeout is 60 seconds; the 120-second cycle budget is checked between calls, not a hard deadline for running inference. Default limits: five utterances per cycle and 40 calls per session. Yield itself consumes a call.

## Traces, evidence, and privacy

- `session_open`: frozen settings, explicit constraint method, source manifest/hash, package/protocol/adapter versions, Python version, and the local Git revision/dirty state when available.
- `request`: instructions, public-history input, prompt hash, and the secret-free wire body or local-generation description.
- `raw_output`: public candidate text, sanitized public response blocks, normalized status, native finish reason, model/request identifiers, supplied token counts, and latency. Local receipts include template, rendered-prompt, and input-token hashes plus generation configuration.
- `utterance`, `yield`, `forced_stop`, and `discard`: distinct events. A cap stop is not a model decision to wait.
- `flag`: a heuristic candidate, not ground truth. `annotation`: observer labels/notes excluded from model input.

Private thinking/reasoning blocks are not requested for observation or stored. A structured tool input is a parsed public payload, not verbatim wire text; the receipt identifies its transport. Source hashes identify the harness, not hosted model weights. Local template/config receipts do not constitute a full weight-file fingerprint.

Logs contain conversation and prompt text. They are append-only local files; `runs/` and `reviews/` are excluded from Git and package distributions. Export may be a partial snapshot: check `export_metadata.upstream_pending`. Closing a browser tab does not stop the server's active cycle. Use Stop or terminate the server.

All public utterances are retained in model context. Only the latest 24 control events are included. Instructions plus input exceeding 80,000 characters cause an explicit stop, not automatic summarization. Characters are not tokens.

The server binds only to loopback, checks Host and API tokens, serves a CSP, and renders utterances as text. It is a single-user research prototype. Do not expose it as a public web service without a separate deployment/security design.

## Validation and research status

See [validation](docs/validation.md) for tested versions, checks, and limitations. Offline unit/HTTP tests verify protocol behavior and adapters with synthetic responses. An opt-in CPU integration test creates a tiny synthetic checkpoint locally to exercise real Transformers loading, native templating, and EOS detection, without downloads:

```sh
TURNLESS_TEST_LOCAL=1 HF_HUB_OFFLINE=1 python3 -m unittest discover -s tests -p test_offline_checkpoint.py -v
```

These checks do not establish model quality or hosted-provider reliability. The legacy `examples/` traces remain v0.1 fixtures. Private pre-release exploratory logs are not included in the public repository and are not a benchmark. The initial research task is to distinguish useful continuation, good silence, and uncertain cases using explicit observer judgments and controlled scenarios.

## Layout

```text
turnless/catalog.py       backend capabilities and constraint identities
turnless/providers.py     factory, transports, strict extraction, offline HF
turnless/protocol.py      decisions, settings, prompt construction
turnless/engine.py        serial inference, epochs, publication, JSONL
turnless/provenance.py    source/version receipts
turnless/server.py        local HTTP API
turnless/web/             English/Japanese UI, no build step
turnless/analyze.py       descriptive trace summaries
examples/                legacy scripted fixtures, not LLM results
tests/                   offline regressions and optional HF integration
```

## License

[Apache License 2.0](LICENSE). Copyright 2026 Ryo ∴ SpiralArchitect. Model weights, datasets, and provider services retain their own licenses and terms.
