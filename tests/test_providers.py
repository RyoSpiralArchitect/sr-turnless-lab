import json
import sys
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from turnless.protocol import Settings
from turnless import providers
from turnless.providers import (AnthropicProvider, LocalHFProvider, OllamaProvider,
                                 OpenAICompatProvider, OpenAIProvider, ProviderError)


class ProviderTests(unittest.TestCase):
    @patch("turnless.providers._post")
    def test_responses_adapter_wire_format(self, post):
        post.return_value = ({"status": "completed", "model": "wire-test", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": '{"action":"yield","text":"","delay_ms":0}'}]},
        ], "usage": {"input_tokens": 9, "output_tokens": 4}}, "request-id")
        provider = OpenAIProvider("unit-test-token-not-a-real-key")
        result = provider.generate("instructions", "{}", Settings(model="wire-test", structured=True), threading.Event())
        url, body, headers = post.call_args.args
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertFalse(body["store"])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertEqual(result.request_id, "request-id")
        self.assertNotIn("reasoning", result.raw)
        self.assertEqual(result.usage["input_tokens"], 9)

    @patch("turnless.providers._post")
    def test_prompt_only_does_not_apply_schema(self, post):
        post.return_value = ({"status": "completed", "output": []}, None)
        OpenAIProvider("test").generate("i", "p", Settings(model="test"), threading.Event())
        self.assertNotIn("text", post.call_args.args[1])

    @patch("turnless.providers._post")
    def test_refusal_is_error_not_emission(self, post):
        post.return_value = ({"status": "completed", "output": [{"type": "message", "content": [
            {"type": "refusal", "refusal": "refusal text"}]}]}, None)
        result = OpenAIProvider("test").generate("i", "p", Settings(model="test"), threading.Event())
        self.assertEqual(result.error, "refusal text")
        self.assertEqual(result.raw, "")

    @patch("turnless.providers._post")
    def test_ollama_wire_format(self, post):
        post.return_value = ({"done": True, "done_reason": "stop", "message": {"content": "{}", "thinking": "not public"},
                              "prompt_eval_count": 3, "eval_count": 2, "model": "test"}, None)
        result = OllamaProvider().generate("i", "p", Settings(model="test", structured=True), threading.Event())
        url, body, _ = post.call_args.args
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertFalse(body["stream"])
        self.assertEqual(body["format"]["type"], "object")
        self.assertEqual(result.raw, "{}")
        self.assertEqual(result.usage["output_tokens"], 2)
        self.assertEqual(result.status, "completed")

    @patch("turnless.providers._post")
    def test_anthropic_wire_format(self, post):
        post.return_value = ({"model": "claude-wire", "stop_reason": "end_turn", "content": [
            {"type": "thinking", "thinking": "not public"},
            {"type": "text", "text": '{"action":"yield","text":"","delay_ms":0}'},
        ], "usage": {"input_tokens": 5, "output_tokens": 2}}, "req-1")
        result = AnthropicProvider("unit-test").generate("i", "p", Settings(model="claude-wire"), threading.Event())
        url, body, headers = post.call_args.args
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], "unit-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(body["system"], "i")
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertNotIn("tools", body)
        self.assertNotIn("not public", result.raw)
        self.assertEqual(result.usage["input_tokens"], 5)

    @patch("turnless.providers._post")
    def test_anthropic_structured_uses_forced_tool(self, post):
        post.return_value = ({"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "name": "utterance_decision",
             "input": {"action": "emit", "text": "ひとつ", "delay_ms": 100}},
        ], "usage": {}}, None)
        result = AnthropicProvider("k").generate("i", "p", Settings(model="c", structured=True, anthropic_mode="tool"), threading.Event())
        _, body, _ = post.call_args.args
        self.assertEqual(body["tool_choice"], {"type": "tool", "name": "utterance_decision",
                                              "disable_parallel_tool_use": True})
        self.assertEqual(body["tools"][0]["input_schema"]["type"], "object")
        decision = json.loads(result.raw)
        self.assertEqual(decision["action"], "emit")
        self.assertEqual(result.status, "completed")

    @patch("turnless.providers._post")
    def test_anthropic_max_tokens_is_incomplete(self, post):
        post.return_value = ({"stop_reason": "max_tokens",
                               "content": [{"type": "text", "text": "x"}]}, None)
        result = AnthropicProvider("k").generate("i", "p", Settings(model="c"), threading.Event())
        self.assertEqual(result.status, "incomplete")
        self.assertIn("token limit", result.error)

    @patch("turnless.providers._post")
    def test_compat_wire_format(self, post):
        post.return_value = ({"model": "compat", "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}, None)
        result = OpenAICompatProvider("https://api.mistral.ai/v1/", "k").generate(
            "i", "p", Settings(model="m", structured=True), threading.Event())
        url, body, headers = post.call_args.args
        self.assertEqual(url, "https://api.mistral.ai/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertFalse(body["stream"])
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"]["type"], "object")
        self.assertEqual(result.raw, "{}")
        self.assertEqual(result.usage, {"input_tokens": 7, "output_tokens": 3})

    @patch("turnless.providers._post")
    def test_compat_without_key_and_length_stop(self, post):
        post.return_value = ({"choices": [
            {"finish_reason": "length", "message": {"content": "x", "reasoning_content": "not public"}}]}, None)
        result = OpenAICompatProvider("http://127.0.0.1:8000/v1").generate(
            "i", "p", Settings(model="m"), threading.Event())
        url, _, headers = post.call_args.args
        self.assertEqual(url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.raw, "x")

    def test_local_hf_requires_transformers(self):
        provider = LocalHFProvider()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaises(ProviderError):
                provider.generate("i", "p", Settings(provider="local", model=tmp, consent=True),
                                  threading.Event())

    def test_local_hf_cancel_before_generation(self):
        provider = LocalHFProvider()
        cancel = threading.Event()
        cancel.set()
        path = "/tmp/turnless-cancel-test"
        with patch.object(LocalHFProvider, "_load", return_value=(object(), object())) as load:
            with self.assertRaises(ProviderError):
                provider.generate("i", "p", Settings(provider="local", model=path, consent=True), cancel)
        load.assert_not_called()

    def test_local_hf_generation_uses_chat_template_and_greedy(self):
        class FakeTensor:
            def __init__(self, seq): self.seq = seq
            @property
            def shape(self): return (1, len(self.seq))
            def tolist(self): return [self.seq]
        class FakeBatch(dict):
            def to(self, device): return self
        class FakeOutput:
            def __init__(self, seq): self.seq = seq
            @property
            def shape(self): return (1, len(self.seq))
            def __getitem__(self, index): return self.seq
        class FakeTokenizer:
            def __init__(self): self.template_args = None
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                self.template_args = (messages, add_generation_prompt, tokenize)
                return "PROMPT"
            def __call__(self, prompt, return_tensors, add_special_tokens):
                self.add_special_tokens = add_special_tokens
                return FakeBatch(input_ids=FakeTensor([1, 2, 3]))
            def decode(self, seq, skip_special_tokens): return "生成結果"
            def get_chat_template(self): return "TEST_TEMPLATE"
        class FakeConfig:
            eos_token_id = 9
            def to_dict(self): return {"eos_token_id": 9}
        class FakeModel:
            generation_config = FakeConfig()
            config = FakeConfig()
            device = "cpu"
            def __init__(self): self.kwargs = None
            def generate(self, input_ids, **kwargs):
                self.kwargs = kwargs
                return FakeOutput([1, 2, 3, 9, 9])
        tokenizer, model = FakeTokenizer(), FakeModel()
        path = "/tmp/turnless-local-fake"
        resolved = str(Path(path).resolve())
        self.addCleanup(providers._LOCAL_MODELS.pop, (resolved, "cpu"), None)
        provider = LocalHFProvider()
        with patch.object(LocalHFProvider, "_load", return_value=(tokenizer, model)):
            result = provider.generate("i", "p", Settings(provider="local", model=path, consent=True),
                                       threading.Event())
        self.assertEqual(result.raw, "生成結果")
        self.assertEqual(result.model, resolved)
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(result.status, "completed")
        self.assertFalse(tokenizer.add_special_tokens)
        self.assertEqual(result.usage, {"input_tokens": 3, "output_tokens": 2})
        messages, add_generation_prompt, tokenize = tokenizer.template_args
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertTrue(add_generation_prompt)
        self.assertFalse(tokenize)
        self.assertEqual(model.kwargs["max_new_tokens"], 1024)
        self.assertFalse(model.kwargs["do_sample"])


if __name__ == "__main__":
    unittest.main()
