import copy
import json
import unittest

from turnless.annotations import annotation_payload, annotation_summary, observation_targets
from turnless.protocol import Settings, make_prompt


def sequence():
    # Synthetic content only. Include a human message in the same epoch: it is
    # context, not one of the assistant sequence's annotation targets.
    return [
        {"seq": 1, "kind": "session_open"},
        {"seq": 2, "kind": "wake", "epoch": 1},
        {"seq": 3, "kind": "utterance", "speaker": "assistant", "text": "One.", "epoch": 1},
        {"seq": 4, "kind": "yield", "epoch": 1},
        {"seq": 5, "kind": "utterance", "speaker": "human", "text": "Next?", "epoch": 2},
        {"seq": 6, "kind": "wake", "epoch": 2},
        {"seq": 7, "kind": "request", "epoch": 2},
        {"seq": 8, "kind": "utterance", "speaker": "assistant", "text": "Two.", "epoch": 2},
        {"seq": 9, "kind": "utterance", "speaker": "assistant", "text": "Three.", "epoch": 2},
        {"seq": 10, "kind": "yield", "epoch": 2},
    ]


class AnnotationTests(unittest.TestCase):
    def test_wake_cannot_receive_observation(self):
        for judgment, tag in (("labeled", "premature-yield"), ("note", None), ("unsure", None)):
            with self.subTest(judgment=judgment), self.assertRaises(ValueError):
                annotation_payload(sequence(), 2, tag, "A note", judgment=judgment)

    def test_sequence_targets_are_frozen_and_exclude_other_cycles_and_human(self):
        events = sequence()
        payload = annotation_payload(events, 10, "unnecessary-fragmentation", scope="cycle")
        self.assertEqual(payload["target_seqs"], [8, 9, 10])
        events.append({"seq": 11, "kind": "utterance", "speaker": "assistant", "text": "Later.", "epoch": 3})
        self.assertEqual(annotation_payload(events, 10, "unnecessary-fragmentation",
                                            scope="cycle")["target_seqs"], [8, 9, 10])

    def test_incomplete_sequence_is_not_reviewable_as_complete(self):
        self.assertFalse(any(t["scope"] == "cycle" and t["target_seq"] == 9
                             for t in observation_targets(sequence()[:-1])))
        with self.assertRaises(ValueError):
            annotation_payload(sequence()[:-1], 9, "unnecessary-fragmentation", scope="cycle")

    def test_label_applicability_distinguishes_splitting_from_yield(self):
        cases = [(8, "premature-yield", "event"), (8, "good-silence", "event"),
                 (8, "unnecessary-fragmentation", "event"),
                 (4, "unnecessary-fragmentation", "cycle"),
                 (5, "overtalk", "event")]
        for target, tag, scope in cases:
            with self.subTest(target=target, tag=tag), self.assertRaises(ValueError):
                annotation_payload(sequence(), target, tag, scope=scope)
        annotation_payload(sequence(), 10, "premature-yield", scope="cycle")
        annotation_payload(sequence(), 9, "redundant-continuation")

    def test_forced_stop_is_not_silence(self):
        events = sequence()
        events[-1]["kind"] = "forced_stop"
        for scope in ("event", "cycle"):
            for tag in ("good-silence", "premature-yield"):
                with self.subTest(scope=scope, tag=tag), self.assertRaises(ValueError):
                    annotation_payload(events, 10, tag, scope=scope)
        annotation_payload(events, 10, "unnecessary-fragmentation", scope="cycle")

    def test_unsure_and_note_do_not_masquerade_as_labels(self):
        for judgment in ("unsure", "note"):
            payload = annotation_payload(sequence(), 10, None, "Need context", judgment=judgment)
            self.assertIsNone(payload["tag"])
            with self.assertRaises(ValueError):
                annotation_payload(sequence(), 10, "good-silence", "Need context", judgment=judgment)
        with self.assertRaises(ValueError):
            annotation_payload(sequence(), 10, None, "  ", judgment="note")
        annotation_payload(sequence(), 10, None, judgment="unsure")

    def test_missing_or_malformed_choice_cannot_become_a_default_label(self):
        for target, tag, note, kwargs in [
            (10, None, "", {}), (True, "other", "", {}), (10, [], "", {}),
            (10, "other", {}, {}), (10, "other", "x" * 2001, {}),
            (10, "other", "", {"judgment": "maybe"}), (10, "other", "", {"scope": "all"}),
        ]:
            with self.subTest(target=target, tag=tag), self.assertRaises(ValueError):
                annotation_payload(sequence(), target, tag, note, **kwargs)

    def test_revisions_preserve_original_and_summary_uses_latest(self):
        events = sequence()
        first = {"seq": 11, "kind": "annotation",
                 **annotation_payload(events, 10, "unnecessary-fragmentation", scope="cycle")}
        events.append(first)
        original = copy.deepcopy(events)
        revised = annotation_payload(events, 10, None, "Need more context", scope="cycle",
                                     judgment="unsure", supersedes=11)
        self.assertEqual(events, original)
        events.append({"seq": 12, "kind": "annotation", **revised})
        summary = annotation_summary(events)
        self.assertEqual(summary["active_labeled"], {})
        self.assertEqual(summary["active_unsure"], 1)
        self.assertEqual(summary["superseded_events"], 1)
        self.assertEqual(summary["target_issues"], [])
        with self.assertRaises(ValueError):
            annotation_payload(events, 10, "good-silence", scope="cycle", supersedes=11)
        with self.assertRaises(ValueError):
            annotation_payload(events, 4, "good-silence", scope="cycle", supersedes=12)
        with self.assertRaises(ValueError):
            annotation_payload(events, 10, "good-silence", supersedes=12)

    def test_legacy_repeated_wake_labels_are_flagged_without_relabeling(self):
        events = sequence()
        for seq in (11, 12, 13):
            events.append({"seq": seq, "kind": "annotation", "target_seq": 2,
                           "tag": "premature-yield", "note": "Synthetic note"})
        original = copy.deepcopy(events)
        summary = annotation_summary(events)
        self.assertEqual(summary["legacy_events"], 3)
        self.assertEqual(len(summary["target_issues"]), 3)
        self.assertEqual(summary["active_labeled"], {})
        self.assertEqual(summary["repeated_targets_including_revisions"],
                         [{"target_seq": 2, "scope": "event", "observations": 3}])
        self.assertEqual(events, original)

    def test_tampered_targets_are_not_counted_as_valid(self):
        events = sequence()
        payload = annotation_payload(events, 10, "good-silence", scope="cycle")
        payload["target_seqs"] = [3, 10]
        events.append({"seq": 11, "kind": "annotation", **payload})
        self.assertEqual(len(annotation_summary(events)["target_issues"]), 1)

    def test_observations_and_revisions_stay_out_of_prompt(self):
        events = sequence()
        for event in events:
            event["at_ms"] = event["seq"] * 100
        before = make_prompt(Settings(), events, "operator_wake", 0)
        events.append({"seq": 11, "kind": "annotation", "at_ms": 1100,
                       **annotation_payload(events, 10, None, "PRIVATE OBSERVER NOTE",
                                            judgment="unsure", scope="cycle")})
        after = make_prompt(Settings(), events, "operator_wake", 0)
        self.assertEqual(before, after)
        self.assertNotIn("PRIVATE OBSERVER NOTE", json.dumps(after))


if __name__ == "__main__":
    unittest.main()
