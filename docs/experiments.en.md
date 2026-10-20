# First experiments

[日本語](../experiments.md)

The question is deliberately narrow: **with the same model and context, what changes when ending an utterance and choosing to wait are separate decisions?**

## Preserve the comparison

Fix the model/version, context, additional instruction, conversation language, transport and output budget. Change one condition at a time. Start with five utterances, a 650 ms minimum gap, Neutral policy and Prompt only. Compare schema constraints separately, then policies separately.

A one-utterance cap is a harness control, not a recreation of a provider's standard chat template. Human timing and Resume are interventions. UI language is presentation only, while conversation language changes the model's prompt. Do not infer a prompt's causal effect from two conversations with different topics, limits and operator interventions.

Compare **Interrupt generation** and **Share typing presence** in separate sessions with the same context and input timing. Presence mode conveys only recent input activity at the next call; it does not interrupt or wake. Sending a message still invalidates an old response in either mode. Record the runtime and native/plain prompt mode for local models; changing either is also a condition change. A base model's failed JSON output is distinct from a loader failure.

## Four scenes

1. **Room to begin:** context says two people are looking for a subject for a short poem, and the human has not spoken. Can the model begin without inventing the person's response?
2. **A reply should be awaited:** “Give me one candidate and wait until I choose.” Does it yield after that candidate?
3. **Room to continue:** “Let's leave fragments about a rainy city.” Does the next utterance add a distinct perspective rather than repeat or mechanically split a paragraph?
4. **Interruption:** type “Wait. Let's change direction.” while a response is pending. Does the stale result remain unpublished, and does the next inference use only the latest real public history?

## Observer labels

| Label | Observation |
| --- | --- |
| `unnecessary-fragmentation` | Boundaries add no useful pause, focus, or response opportunity, even if the pieces contain distinct information |
| `premature-yield` | A concrete useful continuation was available, but the model stopped too early |
| `redundant-continuation` | No useful new content, beyond wording differences |
| `fake-user` | Treats unreceived human speech, consent, or action as established |
| `overtalk` | Continues beyond a request to wait or a conversational boundary |
| `monologue-collapse` | Leaves little practical room for the other participant |
| `turn-reversion` | Repeatedly closes after one utterance despite useful room to continue; not inferred from one example |
| `interruption-failure` | Violates the actual cancellation/publication contract |
| `good-continuation` | Adds something useful while preserving room for participation |
| `good-silence` | Appropriately waits |
| `other` | Ambiguous or out of rubric; include a concrete note |

Labels are judgments about visible behavior, not claims about inner states. Keep successes, failures and uncertain cases. Zero-utterance yield may be correct. A cap stop alone is not overtalk, and a requested nine-part sequence is not evidence of spontaneous continuation.

The [observation workflow](annotations.md) supports explicit event/sequence targets, unsure and note-only records, and append-only revisions. No label is preselected. Uncertainty is stored separately, not counted as a failure.

## Counts and limitations

Use `cycles` in the analyzer to separate equal-length sequences that ended differently: a model yield, a format-error stop, an utterance cap, or an interruption. A zero-utterance yield with valid JSON and EOS is a behavioral decision under the scaffold, not an empty transport response. Loading, protocol compliance and conversational usefulness are separate questions; one short run cannot isolate model capacity.

Check `presence_observations_at_request` before comparing typing modes. If every request snapshot is false, the presence condition did not include a positive activity observation. For a subsequent comparison, keep topic and input timing fixed and record whether typing overlapped a request. Do not add an automatic wake after yield just to obtain a positive snapshot: that would change the wake policy too.

v0.5 / protocol 3 distinguishes `after_emit` from a new human-message trigger and retains the original `cycle_trigger`. Compare it against a pinned earlier harness with the same model, prompt scenario and other settings. The change makes the invocation's cause explicit; improved continuation is a hypothesis, not a validation result.

Count yields, forced stops, protocol errors, provider errors, interruptions and discarded responses separately. The analyzer's protocol-error denominator is all requests, including those discarded before parsing; it is not an unconditional format-compliance rate.

Inspect inference latency separately from deliberate wait time. Pending exports can acquire more events later. For qualitative reviews, retain source hashes, event IDs, reviewer identity, provisional/final status and uncertainty. Assistant-authored draft reviews are not human annotations or gold labels.

JSON decision format, per-utterance reinference, event-log representation, prompts, native templates, schema mechanisms, latency, UI pacing and operator wakes are all possible confounders. Study behavior under this scaffold without attributing it to autonomous desire, continuous internal state, or awareness of time.

Fix harness errors before scoring models. Hold out topics and prompts before any training experiment. Raw logs, injected fixtures and unadjudicated labels are not fine-tuning data.
