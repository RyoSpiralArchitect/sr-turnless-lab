import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from turnless.engine import Session
from turnless.protocol import Settings
from turnless.providers import ModelResult


def emit(text, delay=0):
    return json.dumps({"action": "emit", "text": text, "delay_ms": delay})

YIELD = '{"action":"yield","text":"","delay_ms":0}'


class Script:
    def __init__(self, *items):
        self.items = list(items)
        self.payloads = []

    def generate(self, instructions, payload, settings, cancel):
        self.payloads.append(json.loads(payload))
        item = self.items.pop(0) if self.items else YIELD
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, ModelResult) else ModelResult(item, model="test-fixture")


def wait_for(predicate, timeout=3):
    start = time.monotonic()
    while not predicate():
        if time.monotonic() - start > timeout:
            raise AssertionError("timed out waiting for test condition")
        time.sleep(0.005)


class EngineTests(unittest.TestCase):
    def session(self, provider, **kwargs):
        settings = Settings(proactive=False, min_gap_ms=0, **kwargs)
        session = Session(settings, provider)
        self.addCleanup(session.close)
        return session

    def settle(self, s):
        wait_for(lambda: s.snapshot()["status"] == "waiting")

    def test_assistant_first_and_consecutive_emits(self):
        p = Script(emit("A1"), emit("A2"), YIELD)
        s = self.session(p)
        s.wake("session_open")
        self.settle(s)
        snapshot = s.snapshot()
        self.assertEqual([e["speaker"] for e in snapshot["events"] if e["kind"] == "utterance"], ["assistant", "assistant"])
        self.assertEqual(snapshot["metrics"]["yields"], 1)
        self.assertEqual(snapshot["metrics"]["calls"], 3)
        self.assertEqual(p.payloads[0]["public_events"], [])
        self.assertEqual(p.payloads[1]["public_events"][0]["text"], "A1")

    def test_yield_does_not_automatically_wake(self):
        s = self.session(Script(YIELD, emit("should not happen")))
        s.wake()
        self.settle(s)
        time.sleep(0.07)
        self.assertEqual(s.calls, 1)
        self.assertEqual(s.snapshot()["metrics"]["assistant"], 0)

    def test_consecutive_humans_without_generation(self):
        s = self.session(Script(YIELD))
        s.human("one", False)
        s.human("two", False)
        self.assertEqual(s.calls, 0)
        self.assertEqual([e["text"] for e in s.events if e["kind"] == "utterance"], ["one", "two"])
        s.wake()
        self.settle(s)
        self.assertEqual(s.calls, 1)

    def test_forced_cap_is_not_yield(self):
        s = self.session(Script(emit("1"), emit("2"), emit("3")), max_utterances=2)
        s.wake()
        self.settle(s)
        m = s.snapshot()["metrics"]
        self.assertEqual((m["assistant"], m["yields"], m["forced_stops"]), (2, 0, 1))

    def test_call_budget_is_session_wide(self):
        s = self.session(Script(emit("one"), YIELD, emit("never")), call_budget=2)
        s.wake()
        self.settle(s)
        s.human("more")
        self.settle(s)
        self.assertEqual(s.calls, 2)
        self.assertTrue(any(e.get("reason") == "session_call_budget" for e in s.events))

    def test_raw_protocol_failure_is_not_repaired(self):
        raw = '```json\n{"bad":true}\n```'
        s = self.session(Script(raw, emit("never")))
        s.wake()
        self.settle(s)
        self.assertEqual(s.calls, 1)
        self.assertEqual(next(e["raw"] for e in s.events if e["kind"] == "raw_output"), raw)
        self.assertEqual(s.snapshot()["metrics"]["assistant"], 0)
        self.assertTrue(any(e["kind"] == "protocol_error" for e in s.events))

    def test_provider_error_is_not_silence(self):
        s = self.session(Script(RuntimeError("offline")))
        s.wake()
        self.settle(s)
        self.assertEqual(s.snapshot()["metrics"]["yields"], 0)
        self.assertTrue(any(e["kind"] == "provider_error" for e in s.events))

    def test_incomplete_result_is_retained_but_not_emitted(self):
        s = self.session(Script(ModelResult(emit("no"), status="incomplete", error="token limit")))
        s.wake()
        self.settle(s)
        self.assertEqual(s.snapshot()["metrics"]["assistant"], 0)
        self.assertTrue(any(e["kind"] == "raw_output" for e in s.events))

    def test_interrupt_during_delay_prevents_emission(self):
        s = self.session(Script(emit("old", 700), emit("new"), YIELD))
        s.wake()
        wait_for(lambda: s.snapshot()["status"] == "delaying")
        s.human("interrupt")
        self.settle(s)
        self.assertEqual([e["text"] for e in s.events if e["kind"] == "utterance"], ["interrupt", "new"])
        self.assertTrue(any(e.get("reason") == "interrupted_before_emit" for e in s.events))

    def test_late_network_result_is_discarded_serially(self):
        entered, release = threading.Event(), threading.Event()
        class SlowProvider:
            calls = 0
            payloads = []
            def generate(inner, instructions, payload, settings, cancel):
                inner.calls += 1
                inner.payloads.append(json.loads(payload))
                if inner.calls == 1:
                    entered.set()
                    release.wait(2)  # deliberately ignore cancellation like a slow HTTP request
                    return ModelResult(emit("STALE"), {"input_tokens": 7, "output_tokens": 3})
                return ModelResult(YIELD)
        p = SlowProvider()
        s = self.session(p)
        s.wake()
        self.assertTrue(entered.wait(1))
        s.human("new one")
        s.human("new two")
        self.assertEqual(p.calls, 1, "must not create parallel billable requests")
        release.set()
        self.settle(s)
        public = [e["text"] for e in s.events if e["kind"] == "utterance"]
        self.assertEqual(public, ["new one", "new two"])
        self.assertEqual(p.calls, 2, "pending wake should be coalesced to the newest history")
        stale = next(e for e in s.events if e["kind"] == "raw_output")
        self.assertTrue(stale["discarded"])
        self.assertEqual(s.snapshot()["metrics"]["input_tokens"], 7)
        self.assertEqual([e["text"] for e in p.payloads[-1]["public_events"]], public)

    def test_close_during_network_work_never_publishes(self):
        entered, release = threading.Event(), threading.Event()
        class Slow:
            def generate(inner, instructions, payload, settings, cancel):
                entered.set(); release.wait(2)
                return ModelResult(emit("old"))
        s = self.session(Slow())
        s.wake(); self.assertTrue(entered.wait(1)); s.close(); release.set()
        s.worker.join(2)
        self.assertFalse(s.worker.is_alive())
        self.assertEqual(s.snapshot()["metrics"]["assistant"], 0)
        self.assertEqual(s.status, "closed")

    def test_exact_repeat_is_only_a_heuristic(self):
        s = self.session(Script(emit("one"), emit("one"), YIELD))
        s.wake(); self.settle(s)
        flag = next(e for e in s.events if e["kind"] == "flag")
        self.assertEqual(flag["source"], "heuristic")
        self.assertEqual(s.snapshot()["metrics"]["assistant"], 2)
        s.annotate(flag["target_seq"], "redundant-continuation", "reviewed")
        self.assertEqual(s.snapshot()["metrics"]["annotations"], 1)

    def test_disk_log_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(Settings(proactive=False, min_gap_ms=0), Script(emit("saved"), YIELD), Path(tmp))
            try:
                s.wake(); self.settle(s)
                disk = [json.loads(line) for line in s.log_path.read_text().splitlines()]
                self.assertEqual(disk, s.events)
                exported = [json.loads(line) for line in s.export_jsonl().splitlines()]
                self.assertEqual(exported[-1]["kind"], "export_metadata")
                self.assertFalse(exported[-1]["upstream_pending"])
            finally:
                s.close()

    def test_seq_monotonic_and_snapshot_cursor(self):
        s = self.session(Script(emit("a"), YIELD))
        s.wake(); self.settle(s)
        self.assertEqual([e["seq"] for e in s.events], list(range(1, len(s.events) + 1)))
        last = s.snapshot()["cursor"]
        self.assertEqual(s.snapshot(last)["events"], [])


if __name__ == "__main__":
    unittest.main()
