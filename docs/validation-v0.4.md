# Validation record · v0.4.0

Local validation date: 2026-10-21. These are engineering checks, not model-quality results. [Earlier v0.2 checks](validation.md) remain available separately.

## Automated checks

- macOS / Python 3.12.6: 98 tests discovered; 96 passed with two optional integrations skipped in the dependency-free suite.
- With both integrations enabled: all 98 tests passed. Dependencies: Transformers 4.57.6, PyTorch 2.12.1, MLX 0.31.2, MLX-LM 0.31.3.
- Tiny synthetic checkpoints exercised Transformers native-template and template-free completion, and MLX quantized native-template loading. Network connections were disabled; no weights were downloaded.
- Regression coverage includes runtime detection, incompatible quantization rejection, ambiguous/missing templates, context allowance, EOS versus token limits, private-only incomplete output, public final extraction, cancellation, presence leases, in-flight interruption, request-time presence and draft rejection.
- English/Japanese catalogs, HTML translation keys and JavaScript syntax passed.

## Bounded real-checkpoint checks

These used existing local directories, a synthetic session-open prompt, greedy decoding, offline mode and disabled network connections. Weights were not fingerprinted or published.

| Checkpoint | Loading/generation | Observed result |
| --- | --- | --- |
| GPT-2, Transformers CPU, plain prompt | Passed; 467 input / 128 output tokens | Reached the 128-token limit; not valid decision JSON |
| gpt-oss 20B MLX quantization, native template | Passed; 487 input / 512 output tokens in the final smoke | Reached the 512-token limit without a public final channel; `incomplete`, public raw empty, usage retained |

An earlier 128-token MLX smoke also loaded and generated without reaching a final channel. It exposed missing termination/usage receipts on extraction failures; the adapter now retains those fields. No private reasoning text was persisted. These checks establish that the two reported loader failures are resolved, not that either checkpoint follows the decision protocol at these budgets.

## Rendered UI

Target flow: select typing mode → start a scripted experiment → type while the assistant is active → observe continued generation with presence, or invalidation in interrupt mode → observe activity clearing.

Chrome Guest at `http://127.0.0.1:8766/`, desktop 1200 × 924 CSS content viewport and responsive 390 × 844. The browser extension was unavailable (`Browser is not available: 1`); native Chrome computer-use controls provided the rendered validation. The test server had no hosted credentials and wrote separate temporary fixture logs. The existing research server was left running.

| Check | Result |
| --- | --- |
| Correct URL/title, meaningful first screen, no error overlay | Passed |
| English default and Japanese switch | Passed; existing conversation text unchanged |
| Presence mode: both activity labels, no typing interruption | Passed |
| Interrupt mode: unpublished result discarded, assistant indicator clears | Passed |
| Idle activity clears; drafts stay out of JSONL | Passed |
| Request-time typing signal reaches next model invocation | Passed in fixture receipts and gated engine tests |
| Mode selection affects new sessions; active mode remains visible | Passed |
| GPT-2 metadata inspection reports Transformers/plain/1024 without loading weights | Passed |
| Desktop and mobile composer, presence, Stop, observation panel | Passed; screenshot evidence captured outside the repository |
| Repeated new-session transitions and console | Passed after fixing a poll/close race; zero console messages in the final transition checks |

The initial UI pass found a request for an already-pruned old session during new-session creation. The UI now drains the old poll before closing, pauses new polls while creating, and clears typing timers. A server restart also invalidates stored session IDs; the expected failed restore was cleared before the final console checks.

## Limits

- Actual local runs did not produce an accepted decision in the bounded smoke calls. Conversational usefulness, JSON adherence over longer runs and typing-sensitive model behavior remain unmeasured.
- Hosted APIs, Ollama, Transformers MPS/CUDA, non-Apple MLX, Safari/Firefox and long-duration operation were not exercised here.
- Local checkpoints must remain immutable for reproducible comparisons. Automatic runtime/prompt resolution is recorded, but native and plain inputs are different conditions.
- The scripted demo verifies the apparatus; it does not measure a learned response to typing.
- [GitHub Actions](https://github.com/RyoSpiralArchitect/sr-turnless-lab/actions) runs the offline suite and package checks on Python 3.10, 3.12 and 3.13. It does not run the optional model integrations.
