"""Offline contract regressions. These fixtures are not live-model evidence."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from turnless.catalog import BACKENDS
from turnless.engine import Session
from turnless.protocol import Settings
from turnless.providers import (AnthropicProvider, OpenAICompatProvider, OpenAIProvider,
                                OllamaProvider, LocalHFProvider, create_provider)
from turnless.server import App

YIELD = '{"action":"yield","text":"","delay_ms":0}'


class AdapterContracts(unittest.TestCase):
    def extract(self, provider, response, **kwargs):
        settings = Settings(model="mock", **kwargs)
        with patch("turnless.providers._post", return_value=(response, "request-test")):
            return provider.generate("instructions", "{}", settings, threading.Event())

    def test_native_anthropic_schema(self):
        provider = AnthropicProvider("mock-key")
        settings = Settings(provider="anthropic", model="mock", structured=True)
        wire = provider.describe_request("i", "{}", settings)
        self.assertEqual(wire["constraint_mode"], "anthropic_json_schema")
        self.assertEqual(wire["body"]["output_config"]["format"]["type"], "json_schema")
        self.assertNotIn("tools", wire["body"])
        self.assertNotIn("mock-key", json.dumps(wire))
        result = self.extract(provider, {"stop_reason": "end_turn", "content": [
            {"type": "text", "text": YIELD}]}, structured=True)
        self.assertEqual(result.status, "completed")

    def test_anthropic_multiple_wrong_missing_mixed_tools_fail_closed(self):
        good = {"type": "tool_use", "name": "utterance_decision", "input": json.loads(YIELD)}
        bad_cases = [
            [good, good],
            [{**good, "name": "different_tool"}],
            [{"type": "text", "text": YIELD}],
            [good, {"type": "text", "text": "another public utterance"}],
            [{**good, "input": "not an object"}],
        ]
        for blocks in bad_cases:
            with self.subTest(blocks=blocks):
                result = self.extract(AnthropicProvider("mock"), {"stop_reason": "tool_use", "content": blocks},
                                      structured=True, anthropic_mode="tool")
                self.assertEqual(result.status, "invalid_response")
                self.assertEqual(len(result.public_response["content"]), len(blocks))

    def test_anthropic_public_text_multiplicity_and_unknown_stop(self):
        for reason, blocks in (
            ("end_turn", [{"type": "text", "text": YIELD}] * 2),
            (None, [{"type": "text", "text": YIELD}]),
            ("pause_turn", [{"type": "text", "text": YIELD}]),
        ):
            result = self.extract(AnthropicProvider("mock"), {"stop_reason": reason, "content": blocks})
            self.assertEqual(result.status, "invalid_response")

    def test_private_blocks_never_enter_receipt(self):
        result = self.extract(AnthropicProvider("mock"), {"stop_reason": "end_turn", "content": [
            {"type": "thinking", "thinking": "PRIVATE_MARKER"},
            {"type": "redacted_thinking", "data": "PRIVATE_MARKER"},
            {"type": "text", "text": YIELD}]})
        self.assertEqual(result.status, "completed")
        self.assertNotIn("PRIVATE_MARKER", json.dumps(result.__dict__))

    def test_chat_completion_cardinality_and_stop_are_required(self):
        provider = OpenAICompatProvider("http://localhost:9999/v1")
        choice = {"finish_reason": "stop", "message": {"role": "assistant", "content": YIELD}}
        for choices in ([], [choice, choice], [{**choice, "finish_reason": None}],
                        [{**choice, "finish_reason": "tool_calls"}]):
            result = self.extract(provider, {"choices": choices})
            self.assertEqual(result.status, "invalid_response")

    def test_chat_text_blocks_and_refusal(self):
        provider = OpenAICompatProvider("http://localhost:9999/v1")
        for content, expected in (([{"type": "text", "text": YIELD}], "completed"),
                                  ([{"type": "text", "text": YIELD}] * 2, "invalid_response"),
                                  (None, "invalid_response")):
            result = self.extract(provider, {"choices": [{"finish_reason": "stop",
                "message": {"role": "assistant", "content": content, "reasoning_content": "PRIVATE"}}]})
            self.assertEqual(result.status, expected)
            self.assertNotIn("PRIVATE", json.dumps(result.__dict__))
        result = self.extract(provider, {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": YIELD, "refusal": "declined"}}]})
        self.assertEqual(result.status, "refused")

    def test_responses_multiple_messages_are_not_one_decision(self):
        msg = {"type": "message", "content": [{"type": "output_text", "text": YIELD}]}
        result = self.extract(OpenAIProvider("mock"), {"status": "completed", "output": [msg, msg]})
        self.assertEqual(result.status, "invalid_response")
        self.assertEqual(len(result.public_response["messages"]), 2)

    def test_ollama_missing_stop_is_not_success(self):
        result = self.extract(OllamaProvider(), {"done": True, "message": {"content": YIELD}})
        self.assertEqual(result.status, "invalid_response")

    def test_fixed_presets_keep_credentials_and_endpoints_separate(self):
        for backend in ("gemini", "mistral"):
            settings = Settings.from_dict({"provider": backend, "model": "mock", "consent": True,
                                           "structured": True})
            provider = create_provider(settings, {backend: backend + "-only", "openai_compat": "wrong"})
            with patch("turnless.providers._post", return_value=({"choices": [{
                "finish_reason": "stop", "message": {"role": "assistant", "content": YIELD}}]}, None)) as post:
                result = provider.generate("i", "{}", settings, threading.Event())
            url, body, headers = post.call_args.args
            self.assertEqual(url, BACKENDS[backend].endpoint)
            self.assertEqual(headers["Authorization"], "Bearer " + backend + "-only")
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertEqual(result.status, "completed")
            with self.assertRaises(ValueError):
                create_provider(settings, {"openai_compat": "not-a-preset-key"})

    def test_fixed_preset_does_not_accept_endpoint_override(self):
        with self.assertRaises(ValueError):
            Settings.from_dict({"provider": "gemini", "model": "mock", "consent": True,
                                "compat_base_url": "https://example.com/v1"})

    def test_factory_covers_catalog_and_server(self):
        keys = {p: "mock-" + p for p in BACKENDS}
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                settings = Settings(provider=backend, model="/tmp/mock" if backend == "local" else "mock",
                                    compat_base_url="http://127.0.0.1:9999/v1" if backend == "openai_compat" else "")
                self.assertTrue(callable(create_provider(settings, keys).generate))
        with tempfile.TemporaryDirectory() as tmp:
            app = App(Path(tmp), gemini_key="mock", mistral_key="mock")
            Path(tmp, "config.json").write_text("{}")
            for backend in ("gemini", "mistral", "local"):
                s = app.create({"provider": backend, "model": tmp if backend == "local" else "mock",
                                "consent": True, "proactive": False})
                s.close()
                s.worker.join(1)

    def test_cancelled_http_does_not_call_transport(self):
        cancel = threading.Event()
        cancel.set()
        with patch("turnless.providers._post") as post:
            with self.assertRaises(RuntimeError):
                OpenAICompatProvider("http://localhost:9999/v1").generate("i", "{}", Settings(), cancel)
        post.assert_not_called()

    def test_source_and_request_receipts_are_bound_without_credentials(self):
        provider = OpenAICompatProvider("http://localhost:9999/v1", "SECRET_TEST_MARKER")
        settings = Settings(provider="openai_compat", model="mock", proactive=False,
                            compat_base_url="http://localhost:9999/v1")
        s = Session(settings, provider)
        try:
            with patch("turnless.providers._post", return_value=({"choices": [{
                "finish_reason": "stop", "message": {"role": "assistant", "content": YIELD}}]}, None)):
                s.wake()
                with s.condition:
                    s.condition.wait_for(lambda: s.calls > 0, timeout=0.1)
                import time
                deadline = time.monotonic() + 2
                while s.snapshot()["status"] != "waiting" and time.monotonic() < deadline:
                    time.sleep(0.005)
            start = s.events[0]
            self.assertEqual(len(start["provenance"]["source_sha256"]), 64)
            self.assertEqual(start["provenance"]["trace_version"], "2")
            request = next(e for e in s.events if e["kind"] == "request")
            self.assertEqual(request["wire_request"]["body"]["messages"][1]["content"], request["input"])
            self.assertNotIn("SECRET_TEST_MARKER", s.export_jsonl())
        finally:
            s.close()
            s.worker.join(1)
