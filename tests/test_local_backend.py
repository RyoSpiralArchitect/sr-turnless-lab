import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

from turnless.protocol import Settings
from turnless.providers import LocalHFProvider, ProviderError, _LOCAL_MODELS


class Tensor:
    def __init__(self, tokens):
        self.tokens = tokens
        self.shape = (1, len(tokens))

    def __getitem__(self, index):
        return self.tokens

    def tolist(self):
        return [self.tokens]


class Batch(dict):
    def to(self, device):
        return self


class LocalBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name).resolve())
        self.addCleanup(_LOCAL_MODELS.pop, (self.path, "cpu"), None)
        self.settings = Settings(provider="local", model=self.path, max_output_tokens=128)
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.return_value = "native template prompt"
        self.tokenizer.return_value = Batch(input_ids=Tensor([1, 2, 3]))
        self.tokenizer.decode.return_value = '{"action":"yield","text":"","delay_ms":0}'
        self.tokenizer.get_chat_template.return_value = "test-template"
        self.model = Mock()
        self.model.device = "cpu"
        self.model.config = SimpleNamespace(to_dict=lambda: {})
        self.model.generation_config = SimpleNamespace(eos_token_id=[9, 10], to_dict=lambda: {"eos_token_id": [9, 10]})

    def run_model(self, tokens):
        self.model.generate.return_value = Tensor([1, 2, 3] + tokens)
        with patch.object(LocalHFProvider, "_load", return_value=(self.tokenizer, self.model)):
            return LocalHFProvider().generate("i", "p", self.settings, threading.Event())

    def test_length_limit_never_becomes_model_yield(self):
        result = self.run_model([8] * 128)
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.status, "incomplete")
        self.assertIn("without EOS", result.error)
        self.assertEqual(json.loads(result.raw)["action"], "yield")

    def test_eos_at_token_limit_is_completed(self):
        result = self.run_model([8] * 127 + [10])
        self.assertEqual((result.status, result.finish_reason), ("completed", "eos"))
        self.assertEqual(len(result.metadata["input_token_ids_sha256"]), 64)
        self.tokenizer.assert_called_once_with("native template prompt", return_tensors="pt", add_special_tokens=False)

    def test_unknown_local_stop_is_incomplete(self):
        result = self.run_model([8, 8])
        self.assertEqual((result.status, result.finish_reason), ("incomplete", "other_stop"))

    def test_short_window_caps_generation_without_truncating_input(self):
        self.model.config = SimpleNamespace(to_dict=lambda: {"n_positions": 5})
        result = self.run_model([8, 9])
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 2)
        self.assertEqual(result.usage["input_tokens"], 3)
        self.assertEqual(result.metadata["effective_max_new_tokens"], 2)
        self.assertEqual(result.status, "completed")

    def test_private_only_capped_output_keeps_usage_and_stop_reason(self):
        self.model.config = SimpleNamespace(to_dict=lambda: {"model_type": "gpt_oss"})
        self.tokenizer.decode.return_value = "<|channel|>analysis<|message|>private"
        result = self.run_model([8] * 128)
        self.assertEqual((result.status, result.finish_reason, result.raw), ("incomplete", "length", ""))
        self.assertEqual(result.usage["output_tokens"], 128)
        self.assertTrue(result.metadata["private_reasoning_omitted"])
        self.assertIn("final channel", result.metadata["public_extraction_error"])

    def test_eos_without_public_final_is_invalid_response(self):
        self.model.config = SimpleNamespace(to_dict=lambda: {"model_type": "gpt_oss"})
        self.tokenizer.decode.return_value = "<|channel|>analysis<|message|>private<|return|>"
        result = self.run_model([8, 9])
        self.assertEqual((result.status, result.finish_reason, result.raw), ("invalid_response", "eos", ""))
        self.assertIn("final channel", result.error)

    def test_offline_load_does_not_download_or_trust_remote_code(self):
        tf = SimpleNamespace(AutoTokenizer=Mock(), AutoModelForCausalLM=Mock())
        tf.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        tf.AutoModelForCausalLM.from_pretrained.return_value = self.model
        with patch.dict("sys.modules", {"transformers": tf}):
            LocalHFProvider()._load(self.path, "cpu")
        for loader in (tf.AutoTokenizer, tf.AutoModelForCausalLM):
            loader.from_pretrained.assert_called_once_with(self.path, local_files_only=True, trust_remote_code=False)
        self.model.to.assert_called_once_with("cpu")
        self.model.eval.assert_called_once()

    def test_hub_id_is_not_implicitly_downloaded(self):
        with self.assertRaisesRegex(ProviderError, "existing checkpoint directory"):
            LocalHFProvider()._load("not-an-existing-org/not-a-model", "cpu")

    def test_cancelled_request_does_not_load_checkpoint(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(LocalHFProvider, "_load") as load:
            with self.assertRaises(ProviderError):
                LocalHFProvider().generate("i", "p", self.settings, cancel)
        load.assert_not_called()
