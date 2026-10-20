# Validation record · v0.2.0

Local validation date: 2026-10-20. This document records engineering checks, not conversational quality or provider rankings.

## Automated checks

- Python 3.12.6 on macOS: 64 discovered unit/HTTP tests; 63 passed and the optional local integration test was skipped in the dependency-free suite.
- Full suite with the optional integration enabled: all 64 tests passed.
- Opt-in local integration: passed with Transformers 4.57.6 and PyTorch 2.12.1 on CPU, `HF_HUB_OFFLINE=1`. A tiny synthetic checkpoint was created and loaded locally; native chat-template rendering and EOS detection ran with network connections disabled. No model was downloaded and this was not a language-quality evaluation.
- Regression cases include multiple/wrong/missing Claude tools, extra public text, native Claude JSON requests, multiple Chat Completions choices, unknown stop reasons, refusals, local token-limit classification, endpoint/credential separation, cancelled requests, source drift, prompt language, stale-generation discard and secret-free receipts.
- English/Japanese catalog parity, all HTML translation keys, and JavaScript syntax passed.
- Wheel and source distribution built with setuptools 80.8.0. The built wheel matched the package source byte-for-byte. A separate virtual environment successfully installed it and ran the CLI; web assets and Apache-2.0 metadata were present. Neither archive included private runs or reviews.
- GitHub Actions runs the offline suite and packaging on Python 3.10, 3.12 and 3.13. Its live result is visible in the repository's Actions tab; a local pass alone is not a CI pass.

## Browser checks

Chrome through the connected computer-use browser interface, direct navigation to `http://127.0.0.1:8766/`. No HTTP bridge or injected page was used. The test server had no provider credentials and used `/tmp/sr-turnless-ui-traces`, separate from research runs.

| Check | Result |
| --- | --- |
| Page URL/title and meaningful initial screen | Passed |
| English default / Japanese switch | Passed |
| UI switch preserves conversation text and conversation-language setting | Passed |
| Scripted A → A → A → yield | Passed |
| Consecutive human inputs with automatic reply disabled | Passed; call count unchanged |
| Typing interruption | Passed; pending publication invalidated |
| Observer annotation | Passed; trace count increased |
| JSONL export to the OS Downloads directory | Passed; file parsed, annotation and export metadata present |
| Local HF selection disables unsupported schema mode | Passed |
| Gemini selection shows its own credential slot | Passed |
| Desktop layout and 390 × 844 viewport | Passed; no horizontal overflow, Stop visible |
| Browser console warnings/errors | None observed during these flows |

The browser's download-event wait timed out, but the actual OS download succeeded and its JSONL contents were verified from disk. This distinction is retained here instead of treating the automation event as the file outcome.

## Not established

- Live OpenAI, Claude, Gemini, Mistral, custom-server or Ollama reliability after this refactor; HTTP adapters were validated with synthetic responses.
- Local language-model quality, large-model memory use, GPU/MPS runtime behavior, cross-device determinism, or throughput.
- Safari, Firefox, Windows UI behavior, long-duration operation or multi-user deployment.
- Semantic label accuracy. Public examples are legacy scripted fixtures. Private exploratory traces are preserved separately and are not published as a benchmark.

No new billable hosted inference was used for these checks. Source and raw research traces were kept separate from UI QA fixtures.
