import json
import threading
import unittest
from unittest.mock import patch

from turnless.engine import Session
from turnless.protocol import Settings, make_prompt
from turnless.providers import ModelResult
from tests.test_engine import Script, YIELD, emit, wait_for


class TypingTests(unittest.TestCase):
    def session(self, provider, mode="presence"):
        session = Session(Settings(proactive=False, min_gap_ms=0, typing_mode=mode), provider)
        self.addCleanup(session.close)
        return session

    def test_presence_is_control_state_not_a_message_or_a_wake(self):
        provider = Script(YIELD)
        session = self.session(provider)
        session.typing(True)
        self.assertEqual(session.calls, 0)
        self.assertEqual(session.status, "waiting")
        session.wake()
        wait_for(lambda: session.status == "waiting")
        self.assertTrue(provider.payloads[0]["presence"]["human_typing"])
        self.assertEqual(provider.payloads[0]["public_events"], [])
        session.typing(False)
        self.assertEqual(session.calls, 1)
        self.assertEqual([e["active"] for e in session.events if e["kind"] == "typing"], [True, False])

    def test_presence_during_generation_does_not_cancel_and_next_call_sees_it(self):
        started, release = threading.Event(), threading.Event()
        class Gate(Script):
            def generate(self, instructions, payload, settings, cancel):
                if not self.payloads:
                    started.set()
                    release.wait(2)
                    self.was_cancelled = cancel.is_set()
                return super().generate(instructions, payload, settings, cancel)
        provider = Gate(emit("visible"), YIELD)
        session = self.session(provider)
        self.addCleanup(release.set)
        session.wake()
        self.assertTrue(started.wait(1))
        session.typing(True)
        self.assertTrue(session.snapshot()["presence"]["assistant_composing"])
        release.set()
        wait_for(lambda: session.status == "waiting")
        self.assertFalse(provider.was_cancelled)
        self.assertFalse(provider.payloads[0]["presence"]["human_typing"])
        self.assertTrue(provider.payloads[1]["presence"]["human_typing"])
        self.assertEqual(session.snapshot()["metrics"]["assistant"], 1)
        self.assertEqual(session.snapshot()["metrics"]["interrupts"], 0)

    def test_interrupt_mode_discards_inflight_result(self):
        started, release = threading.Event(), threading.Event()
        class Gate:
            def generate(self, *args):
                started.set()
                release.wait(2)
                return ModelResult(emit("stale"))
        session = self.session(Gate(), "interrupt")
        self.addCleanup(release.set)
        session.wake()
        self.assertTrue(started.wait(1))
        session.typing(True)
        self.assertEqual(session.status, "waiting")
        self.assertFalse(session.snapshot()["presence"]["assistant_composing"])
        self.assertTrue(session.inflight)  # Upstream work may still be returning.
        release.set()
        wait_for(lambda: not session.inflight)
        self.assertEqual(session.snapshot()["metrics"]["assistant"], 0)
        self.assertEqual(session.snapshot()["metrics"]["interrupts"], 1)
        self.assertTrue(next(e for e in session.events if e["kind"] == "raw_output")["discarded"])

    def test_heartbeat_and_disconnect_expiry(self):
        session = self.session(Script())
        with patch("turnless.engine.time.monotonic", return_value=100):
            session.typing(True)
        with patch("turnless.engine.time.monotonic", return_value=102):
            session.typing(True)
        with patch("turnless.engine.time.monotonic", return_value=104):
            self.assertTrue(session.snapshot()["presence"]["human_typing"])
        with patch("turnless.engine.time.monotonic", return_value=105.1):
            self.assertFalse(session.snapshot()["presence"]["human_typing"])
        typing = [e for e in session.events if e["kind"] == "typing"]
        self.assertEqual([e["active"] for e in typing], [True, False])
        self.assertEqual(typing[-1]["reason"], "idle_timeout")
        self.assertEqual(session.calls, 0)

    def test_send_and_close_clear_presence(self):
        session = self.session(Script())
        session.typing(True)
        session.human("sent", False)
        self.assertFalse(session.snapshot()["presence"]["human_typing"])
        session.typing(True)
        session.close()
        self.assertFalse(session.snapshot()["presence"]["human_typing"])
        with self.assertRaises(ValueError):
            session.typing(True)

    def test_legacy_interrupt_route_obeys_selected_mode(self):
        session = self.session(Script(YIELD))
        session.interrupt("human_typing")
        self.assertTrue(session.snapshot()["presence"]["human_typing"])
        self.assertEqual(session.epoch, 0)
        self.assertFalse(any(e["kind"] == "pause" for e in session.events))

    def test_invalid_state_and_modes(self):
        session = self.session(Script())
        for active in ("true", 1, None, {}):
            with self.assertRaises(ValueError):
                session.typing(active)
        for field in ("typing_mode", "local_runtime", "local_prompt_mode"):
            with self.assertRaises(ValueError):
                Settings.from_dict({field: "unknown"})

    def test_interrupt_prompt_has_no_presence_signal(self):
        instructions, payload = make_prompt(Settings(), [], "session_open", 0, {"human_typing": True})
        self.assertNotIn("presence", json.loads(payload))
        self.assertNotIn("human_typing means", instructions)
