# SR Turnless Lab ∴

**An utterance can end before the conversation is handed over.**

[日本語](README.ja.md) · [Experiment guide](docs/experiments.en.md) · [Backend contracts](docs/backends.md)

A small local laboratory for conversational boundaries. A model produces one public utterance (`emit`) or chooses to wait (`yield`). After each emitted utterance, it sees the updated public history and makes a fresh decision. Humans can interrupt or send consecutive messages, too.

This is an **application-level protocol**, not a modification of model weights, tokenizers, native chat templates, or EOS semantics. `emit` and `yield` are JSON values, not special tokens. There is no tool execution, background thinking, timer-driven reawakening, or fine-tuning pipeline.

## An observed exchange

Sentence excerpts from one exploratory conversation, with intervening text omitted:

> **Human · #211:** it is often not easy to decide when you speak, stop talking, or keep silent for human too.
>
> **Assistant · #220, utterance 2 of 5:** Neither speaker creates the whole conversational rhythm alone; it arises between contributions.
>
> **Assistant · #232, utterance 5 of 5:** A pause can mean thoughtfulness, discomfort, respect, or nothing at all depending on the setting.

The assistant produced five utterances without an intervening human message, then yielded. Repeated agreement accompanied shifts in framing; whether that adds something or becomes redundant is a question for observation, not something message count decides.

Recorded 2026-10-21 JST with requested/reported model `gpt-5.6-luna`, v0.4.0 / protocol 2, Prompt only, Exploratory, English, typing presence, an 11-utterance cap and 350 ms minimum gap. Private trace `ed97a591…`, events #211–235; only the selected sentences are published here. Across that session: 26 calls, 19 published utterances, six model yields and one format-error stop. Earlier, the model repeatedly emitted once then yielded; the longer sequences followed human encouragement within the same evolving conversation. All 26 request snapshots had `human_typing: false`, so this run does not demonstrate a typing-presence effect. It is an example, not a controlled comparison or evidence of the model's inner experience.

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
4. Click **Observe sequence** beside a yield/stop, or select an individual utterance. Choose an observation and export JSONL.

For a contrasting condition, select **When the human types → Share typing presence** before starting a new experiment. The assistant keeps generating, while both participants receive activity information. The composer shows **You are typing… / Assistant is composing…**; the model receives only a typing boolean at the next inference, never the unsent draft.

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
| Local checkpoint | Transformers or MLX; native chat template or plain completion | Prompt only | None |

Enter an **exact model ID you can access**, or a local checkpoint directory. Schema support depends on the selected model and server. Unsupported requests fail visibly; there are no silent fallbacks or retries. The Gemini and Mistral presets have fixed endpoints and separate credentials. Custom endpoints accept HTTPS or loopback HTTP.

Credentials remain server-side and are never part of settings, browser responses, or request receipts. Use your secure credential-management workflow. In Codex, OpenAI credentials can be provisioned through the secure OpenAI Platform flow. Existing server-managed credentials use the environment names above. Credential files outside this repository can also be selected with `--key-file`, `--anthropic-key-file`, `--gemini-key-file`, `--mistral-key-file`, or `--compat-key-file`. Restart the server after changing its credential configuration.

The custom endpoint default can be selected with `--compat-base-url`. Ollama defaults to `http://127.0.0.1:11434`; `OLLAMA_HOST` may select another loopback HTTP endpoint. Redirects are rejected. A single custom credential is used for the custom endpoint, so select it for the intended destination.

### Offline local checkpoint

Install the optional dependencies for the checkpoint's runtime on the inference machine:

```sh
python3 -m pip install '.[local]'
# For MLX checkpoints on Apple silicon instead:
python3 -m pip install '.[mlx]'
HF_HUB_OFFLINE=1 python3 -m turnless
```

Choose **Local checkpoint · Transformers / MLX**, enter an existing directory, and press **Check checkpoint**. This checks metadata without loading weights. On PowerShell, set `$env:HF_HUB_OFFLINE = "1"` before starting Python.

- **Runtime Auto** selects MLX for its quantization configuration (`bits` / `group_size`, without a Transformers `quant_method`); otherwise it selects Transformers. Choose MLX explicitly for an unquantized MLX checkpoint. MLX uses Metal; CPU / MPS / CUDA applies to Transformers. MLX quantization cannot be loaded through Transformers simply by selecting MPS.
- **Prompt Auto** uses the native chat template when present; otherwise it uses a recorded, versioned plain completion prompt. This lets base models such as GPT-2 start without inventing a native template. **Native** requires a template; **Plain** explicitly uses the completion convention. A base model may load and generate successfully while failing the JSON protocol.
- Inputs are never silently truncated. If a known context window cannot hold the requested output, the effective output allowance is capped to the remaining space and recorded. An input that already fills the window fails with an actionable error.

Loaders require a local directory, use offline tokenizer loading and `trust_remote_code=False`, and never download a Hub ID. Native templates are tokenized without adding special tokens twice. Generation is greedy. Local loading and generation are serialized; only the most recently selected checkpoint is cached. Restart after modifying weights in place or to release the cache. Cancellation is checked before loading, while waiting, and between generated tokens; loading and a running compute step may still need to finish. EOS, output limits and cancellation remain separate outcomes. gpt-oss uses only its public final channel; private reasoning is not persisted. Details: [backend contracts](docs/backends.md).

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

Since v0.5 / protocol 3, a continuation request uses `trigger: "after_emit"`; `cycle_trigger` retains the original wake reason. This distinguishes a fresh decision after the assistant's own utterance from a new human message. Earlier versions repeated the original trigger throughout the cycle. Treat this prompt change as a new experimental condition; its effect on model behavior has not yet been measured.

`delay_ms` is a wait **after inference returns**, from 0 to 5000 ms. After the first utterance, the applied wait is the larger of the model's proposal and the configured minimum gap. Both values are recorded.

**Interrupt generation** is the default typing mode: typing invalidates the current epoch. **Share typing presence** leaves generation running and adds recent human activity to the next request's harness state. Both modes show real assistant activity (queued, generating, or waiting to publish), not hidden thought. Mode selection is fixed for a session and saved in JSONL.

Typing activity is throttled to at most one update per second. The UI clears it after two idle seconds, blur, or sending; the server's three-second lease handles lost tabs/connections. Only transitions are logged, with fresh presence snapshots in requests. Presence is not an utterance or a wake trigger. An in-flight request cannot see later activity; ending typing never wakes a model that yielded.

Submitting a message or pressing Stop still invalidates the current epoch in both modes. Late public results are retained as discarded output and never published. Already published utterances remain in history. One inference worker runs per session; pending wake requests are coalesced to the latest history.

Upstream inference cancellation and billing cessation are **not guaranteed**. A synchronous HTTP call may delay the next inference until it returns or times out. HTTP I/O timeout is 60 seconds; the 120-second cycle budget is checked between calls, not a hard deadline for running inference. Default limits: five utterances per cycle and 40 calls per session. Yield itself consumes a call.

## Observe without forcing a verdict

The observation panel shows the exact event IDs, public text, and preceding human message. **Whole sequence** covers assistant utterances in one completed cycle plus its yield or harness stop. **This event** targets only the selected event. Raw outputs and wake events remain inspectable but cannot receive behavioral labels.

There is no default label. Every available label has an English/Japanese definition. **Unnecessary splitting** (`unnecessary-fragmentation`) describes boundaries that add no useful pause, focus, or response opportunity. It differs from **Redundant continuation** (repeating content) and **Yielding too early** (choosing to wait before finishing something needed). Multiple utterances alone do not establish a failure.

Choose **Not sure yet** or **Note only** when a label would overstate the observation. Notes and uncertainty are recorded separately from labels. Saved observations remain visible; **Save revision** appends a replacement linked by `supersedes`, preserving the original. **Next unreviewed** helps navigate completed sequences. Switching targets keeps unfinished drafts for that target during the page session; it never carries a label to a different target.

Saving clears the selected target, so a later observation cannot silently revise that same sequence. Select it again to inspect or revise its saved history. The save button names the target ID; **Latest sequence** and an older-target notice help keep a live review aligned with the conversation. Incoming events never move an unfinished review to a different target.

See the [annotation guide](docs/annotations.md) for target rules, event fields, and legacy-log handling. These are exploratory operator judgments, not independently adjudicated ground truth. The model prompt and speaking policy are unchanged in v0.3.

## Traces, evidence, and privacy

- `session_open`: frozen settings, explicit constraint method, source manifest/hash, package/protocol/adapter versions, Python version, and the local Git revision/dirty state when available.
- `request`: instructions, public-history input, prompt hash, and the secret-free wire body or local-generation description.
- `raw_output`: public candidate text, sanitized public response blocks, normalized status, native finish reason, model/request identifiers, supplied token counts, and latency. Local receipts include template, rendered-prompt, and input-token hashes plus generation configuration.
- `utterance`, `yield`, `forced_stop`, and `discard`: distinct events. A cap stop is not a model decision to wait.
- `flag`: a heuristic candidate, not ground truth. `annotation`: observer labels/notes excluded from model input.

`python3 -m turnless.analyze "runs/*.jsonl"` reports each cycle's published event IDs and ending (`model_yield`, `harness_stop`, interruption or an unfinished snapshot). `presence_observations_at_request` counts the booleans recorded in request payloads; UI typing events alone do not establish that an active-typing signal was included. Missing legacy snapshots remain unknown, and a failed request does not prove delivery. Annotation summaries also identify reviews saved on an older sequence, without treating retrospective review as an error or changing its label.

Private thinking/reasoning blocks are not requested for observation or stored. A structured tool input is a parsed public payload, not verbatim wire text; the receipt identifies its transport. Source hashes identify the harness, not hosted model weights. Local template/config receipts do not constitute a full weight-file fingerprint.

Logs contain conversation and prompt text. They are append-only local files; `runs/` and `reviews/` are excluded from Git and package distributions. Export may be a partial snapshot: check `export_metadata.upstream_pending`. Closing a browser tab does not stop the server's active cycle. Use Stop or terminate the server.

All public utterances are retained in model context. Only the latest 24 control events are included. Instructions plus input exceeding 80,000 characters cause an explicit stop, not automatic summarization. Characters are not tokens.

The server binds only to loopback, checks Host and API tokens, serves a CSP, and renders utterances as text. It is a single-user research prototype. Do not expose it as a public web service without a separate deployment/security design.

## Validation and research status

See [v0.4 validation](docs/validation-v0.4.md) for tested versions, checks, and limitations. Offline unit/HTTP tests use synthetic responses. Opt-in integrations create tiny checkpoints locally to exercise Transformers native/plain prompts and MLX quantized loading with EOS detection, without downloads:

```sh
TURNLESS_TEST_LOCAL=1 HF_HUB_OFFLINE=1 python3 -m unittest discover -s tests -p test_offline_checkpoint.py -v
# Apple silicon + MLX:
TURNLESS_TEST_MLX=1 HF_HUB_OFFLINE=1 python3 -m unittest discover -s tests -p test_offline_checkpoint.py -v
```

These checks do not establish model quality or hosted-provider reliability. The legacy `examples/` traces remain v0.1 fixtures. Private pre-release exploratory logs are not included in the public repository and are not a benchmark. The initial research task is to distinguish useful continuation, good silence, and uncertain cases using explicit observer judgments and controlled scenarios.

## Layout

```text
turnless/catalog.py       backend capabilities and constraint identities
turnless/providers.py     factory, transports, strict extraction, offline HF/MLX
turnless/local_models.py  checkpoint metadata, local prompt/output conventions
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
