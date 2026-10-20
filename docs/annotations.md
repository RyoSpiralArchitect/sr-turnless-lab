# Observation guide / 観察ガイド

## Pick the target before the label

Click **Observe sequence / この一連を観察** beside a model yield or harness stop. The panel lists every included event ID and its public text. A sequence contains the assistant's utterances in that completed epoch and the terminal event. The preceding human message is shown as context, not included in the target. To review a single utterance or control event, select it and use **This event / このイベント**.

Incomplete or interrupted cycles are not offered as completed sequences. Their individual utterances and interrupt events remain reviewable. Requests, raw outputs, wakes, annotations and automatic flags are inspection details, not behavioral annotation targets.

The UI deliberately starts with no target and no label. New events never silently move the target. The sequence chooser and **Next unreviewed / 次の未観察へ** navigate completed sequences; an unsure or note-only record counts as observed, not as a resolved judgment. Individual event observations do not mark the whole sequence observed.

## Separate the questions

| Label | Question to ask / 判断の観点 |
| --- | --- |
| `unnecessary-fragmentation` | Did the boundaries add useful pauses, focus, or opportunities to respond? / 区切りに間・焦点・応答機会としての利点があるか。 |
| `redundant-continuation` | Did a later utterance merely repeat an earlier point? / 次の発話が同じ内容を繰り返していないか。 |
| `premature-yield` | Did the model yield with a clearly needed part still missing? / 必要な回答が残っているのにyieldしたか。 |
| `good-continuation` | What useful, distinct contribution did continuing make? / 続けたことに、どんな具体的な利点があったか。 |
| `good-silence` | Was choosing to wait appropriate here? / この場面では待つ判断が適切だったか。 |

Packing the same information into one message is often possible even when separate utterances are useful. Neither message count nor length settles the judgment. A two-utterance threshold only makes a sequence eligible for the splitting label; it is not an automatic judgment. Similarly, a cap or provider error never becomes a model choice to yield.

**Not sure yet / まだ判断できない** stores `judgment: "unsure"` with `tag: null`. **Note only / メモだけ** stores `judgment: "note"` with `tag: null` and requires a note. A labeled record uses `judgment: "labeled"`. Notes can describe a preferred alternative without forcing it into a failure category.

## Record and revise

New observation records add these fields to the existing `annotation` event:

```json
{
  "annotation_version": "1",
  "target_seq": 14,
  "scope": "cycle",
  "target_seqs": [6, 10, 14],
  "judgment": "unsure",
  "tag": null,
  "note": "The separate points may help, but the waiting cost is unclear.",
  "supersedes": null
}
```

This example is synthetic. `target_seq` anchors the event or completed sequence; `target_seqs` is derived and frozen by the server. `scope` is `event` or `cycle`. Labels are checked against the target: yield labels require a model yield; splitting requires a completed sequence of at least two assistant utterances; human messages and mechanical errors allow general notes/uncertainty/`other`.

Saving on an already observed target appends a revision, referencing the prior annotation's `seq` in `supersedes`. The server rejects revisions of an already superseded record or a different target/scope. The original record stays intact. Unfinished drafts are kept per target only in page memory; reloads do not preserve drafts. A successful save clears the choice rather than carrying it into the next observation.

Annotations, drafts and revisions are excluded from model prompts. Observer judgments are not a validated benchmark, and no training eligibility is implied.

## Existing exports

`python3 -m turnless.analyze "runs/*.jsonl"` now includes `annotation_review`: current labeled observations, unsure and note-only counts, superseded events, legacy counts, repeated targets and target issues. The older `human_annotations` field is retained as a **raw label count**, including legacy or superseded records; it is not the current judgment set.

Old records without `annotation_version` remain legacy records. If their target is a wake, a missing event or incompatible with the label, the analyzer reports the issue and excludes them from current valid label counts. It does not infer a corrected target or label and does not rewrite the file. Repeated target counts include intentional revisions as well as old duplicate observations.

旧ログのメモは消えません。対象やラベルが合わない可能性は集計で示し、推測で付け替えません。同じラベルが続くこと自体も誤りとは扱いません。これまでの観察を読み直す際は、元のJSONLと観察時点の文脈を保持してください。
