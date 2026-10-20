import json
import tempfile
import unittest
from pathlib import Path

from turnless.analyze import cycle_summaries, presence_observations, summarize
from turnless.protocol import Settings


class TraceAnalysisTests(unittest.TestCase):
    def test_same_utterance_count_different_stops_are_not_conflated(self):
        events = []
        for epoch, terminal, reason in [(1, "forced_stop", "protocol_error"),
                                        (2, "yield", None), (3, "forced_stop", "utterance_cap")]:
            events.extend([
                {"kind": "wake", "epoch": epoch, "reason": "human_message"},
                {"kind": "request", "epoch": epoch},
                {"kind": "utterance", "epoch": epoch, "speaker": "assistant"},
                {"kind": "request", "epoch": epoch},
                {"kind": terminal, "epoch": epoch, "reason": reason},
            ])
        for seq, event in enumerate(events, 1):
            event["seq"] = seq
        result = cycle_summaries(events)
        self.assertEqual([c["assistant_utterances"] for c in result], [1, 1, 1])
        self.assertEqual([c["outcome"] for c in result], ["harness_stop", "model_yield", "harness_stop"])
        self.assertEqual([c["reason"] for c in result], ["protocol_error", None, "utterance_cap"])

    def test_interrupt_uses_previous_epoch_and_late_output_is_not_speech(self):
        events = [
            {"seq": 1, "kind": "wake", "epoch": 1, "reason": "session_open"},
            {"seq": 2, "kind": "request", "epoch": 1},
            {"seq": 3, "kind": "interrupt", "epoch": 2, "reason": "human_message"},
            {"seq": 4, "kind": "utterance", "epoch": 2, "speaker": "human", "text": "Hi"},
            {"seq": 5, "kind": "wake", "epoch": 2, "reason": "human_message"},
            {"seq": 6, "kind": "raw_output", "epoch": 1, "discarded": True},
            {"seq": 7, "kind": "request", "epoch": 2},
            {"seq": 8, "kind": "yield", "epoch": 2},
            {"seq": 9, "kind": "session_close"},
        ]
        first, second = cycle_summaries(events)
        self.assertEqual((first["outcome"], first["end_seq"], first["assistant_utterances"]),
                         ("interrupted", 3, 0))
        self.assertEqual((second["outcome"], second["end_seq"], second["preceding_human_seq"]),
                         ("model_yield", 8, 4))
        self.assertEqual([first["requests"], second["requests"]], [1, 1])

    def test_partial_snapshot_is_not_a_yield_or_stop(self):
        result = cycle_summaries([{"seq": 1, "kind": "wake", "epoch": 1},
                                  {"seq": 2, "kind": "request", "epoch": 1}])
        self.assertEqual(result[0]["outcome"], "open_at_snapshot")
        self.assertIsNone(result[0]["end_seq"])

    def test_presence_requires_boolean_in_request_not_typing_event(self):
        events = [{"kind": "typing", "active": True}]
        for value in [True, False, 1, "false", None]:
            events.append({"kind": "request", "input": json.dumps({"presence": {"human_typing": value}})})
        events.extend([{"kind": "request", "input": "{"}, {"kind": "request", "input": "{}"},
                       {"kind": "request", "input": "null"}, {"kind": "request"}])
        self.assertEqual(presence_observations(events),
                         {"human_typing_true": 1, "human_typing_false": 1, "missing_or_invalid": 7})

    def test_summary_retains_legacy_unknown_presence_and_protocol(self):
        events = [{"seq": 1, "session_id": "synthetic", "kind": "session_open", "settings": Settings().public()},
                  {"seq": 2, "kind": "wake", "epoch": 1, "reason": "session_open"},
                  {"seq": 3, "kind": "request", "epoch": 1, "input": "{}"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            result = summarize(path)
        self.assertEqual(result["protocol_version"], "legacy_unspecified")
        self.assertEqual(result["presence_observations_at_request"]["missing_or_invalid"], 1)
        self.assertEqual(result["cycles"][0]["outcome"], "open_at_snapshot")
        self.assertTrue(result["trace_has_unresolved_requests"])


if __name__ == "__main__":
    unittest.main()
