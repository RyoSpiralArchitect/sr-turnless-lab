import json
import unittest
from turnless.protocol import ProtocolError, Settings, make_prompt, parse_decision


class ProtocolTests(unittest.TestCase):
    def test_emit(self):
        d = parse_decision('{"action":"emit","text":"ねえ","delay_ms":2}')
        self.assertEqual((d.action, d.text, d.delay_ms), ("emit", "ねえ", 2))

    def test_yield(self):
        self.assertEqual(parse_decision('{"action":"yield","text":"","delay_ms":0}').action, "yield")

    def test_invalid_protocol(self):
        cases = [
            '```json\n{}\n```', '[]', '{}',
            '{"action":"user","text":"Hi","delay_ms":0}',
            '{"action":"emit","text":" ","delay_ms":0}',
            '{"action":"emit","text":"x","delay_ms":true}',
            '{"action":"emit","text":"x","delay_ms":0.5}',
            '{"action":"emit","text":"x","delay_ms":-1}',
            '{"action":"emit","text":"x","delay_ms":5001}',
            '{"action":"yield","text":"bye","delay_ms":0}',
            '{"action":"yield","text":"","delay_ms":10}',
            '{"action":"emit","text":"x","delay_ms":0,"speaker":"human"}',
            '{"action":"emit","action":"yield","text":"","delay_ms":0}',
            '{"action":"emit","text":"x","delay_ms":NaN}',
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                parse_decision(raw)

    def test_settings_limits(self):
        for data in ({"max_utterances": 0}, {"call_budget": 999}, {"min_gap_ms": True},
                     {"proactive": "true"}, {"unknown": 1}, {"provider": "other"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                Settings.from_dict(data)

    def test_real_provider_requires_explicit_consent_and_model(self):
        for data in ({"provider": "openai"}, {"provider": "ollama", "model": "test"},
                     {"provider": "anthropic", "model": "test"},
                     {"provider": "openai_compat", "model": "test", "compat_base_url": "https://api.mistral.ai/v1"},
                     {"provider": "local", "model": "/tmp/model"}):
            with self.assertRaises(ValueError):
                Settings.from_dict(data)
        self.assertEqual(Settings.from_dict({"provider": "ollama", "model": "test", "consent": True}).model, "test")
        self.assertEqual(Settings.from_dict(
            {"provider": "anthropic", "model": "claude-x", "consent": True}).provider, "anthropic")

    def test_local_rejects_structured_and_accepts_long_paths(self):
        with self.assertRaises(ValueError):
            Settings.from_dict({"provider": "local", "model": "/tmp/model", "consent": True, "structured": True})
        local = Settings.from_dict({"provider": "local", "model": "/tmp/" + "m" * 500, "consent": True})
        self.assertEqual(len(local.model), 505)

    def test_compat_base_url_validation(self):
        for url in ("", "ftp://example.com/v1", "http://api.mistral.ai/v1", "https://",
                    "https://user:pw@example.com/v1", "https://:pw@example.com/v1",
                    "http://localhost:bad/v1", "https://example.com/v1?x=1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                Settings.from_dict({"provider": "openai_compat", "model": "m", "consent": True,
                                    "compat_base_url": url})
        for url in ("https://api.mistral.ai/v1", "http://127.0.0.1:8000/v1",
                    "http://localhost:1234/v1", "https://generativelanguage.googleapis.com/v1beta/openai"):
            with self.subTest(url=url):
                self.assertEqual(Settings.from_dict(
                    {"provider": "openai_compat", "model": "m", "consent": True,
                     "compat_base_url": url}).compat_base_url, url)

    def test_prompt_contains_only_real_speakers_and_no_annotations(self):
        events = [
            {"seq": 1, "kind": "utterance", "speaker": "assistant", "text": "A1", "at_ms": 0},
            {"seq": 2, "kind": "utterance", "speaker": "assistant", "text": "A2", "at_ms": 1},
            {"seq": 3, "kind": "annotation", "tag": "private-observer-note", "at_ms": 2},
            {"seq": 4, "kind": "wake", "reason": "operator_wake", "at_ms": 3},
        ]
        _, payload = make_prompt(Settings(), events, "operator_wake", 2)
        data = json.loads(payload)
        self.assertEqual([e["speaker"] for e in data["public_events"]], ["assistant", "assistant"])
        self.assertNotIn("private-observer-note", payload)
        self.assertEqual(data["trigger"], "operator_wake")


if __name__ == "__main__":
    unittest.main()
