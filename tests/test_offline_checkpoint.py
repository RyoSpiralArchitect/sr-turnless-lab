"""Opt-in CPU integration test; a tiny synthetic checkpoint, never model-quality evidence."""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("TURNLESS_TEST_LOCAL") == "1", "optional local dependencies")
class OfflineCheckpointTests(unittest.TestCase):
    def test_native_template_directory_load_and_eos_without_network(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            import torch
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
            from turnless.providers import LocalHFProvider, _LOCAL_MODELS
            from turnless.protocol import Settings

            native = Tokenizer(WordLevel({"<eos>": 0, "<bos>": 1, "<unk>": 2, "system": 3,
                                         "user": 4, "assistant": 5, "hello": 6}, unk_token="<unk>"))
            native.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=native, eos_token="<eos>",
                                                bos_token="<bos>", unk_token="<unk>", pad_token="<eos>")
            tokenizer.chat_template = (
                "{{ bos_token }}{% for message in messages %}{{ message['role'] }} "
                "{{ message['content'] }} {{ eos_token }}{% endfor %}"
                "{% if add_generation_prompt %}assistant {% endif %}")
            model = GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=256, n_embd=8,
                                              n_layer=1, n_head=1, eos_token_id=0, bos_token_id=1,
                                              pad_token_id=0))
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp).resolve())
                tokenizer.save_pretrained(path)
                model.save_pretrained(path)
                try:
                    with patch("socket.create_connection", side_effect=AssertionError("network disabled")):
                        result = LocalHFProvider().generate(
                            "hello", "hello", Settings(provider="local", model=path, max_output_tokens=128),
                            threading.Event())
                    self.assertEqual((result.status, result.finish_reason), ("completed", "eos"))
                    self.assertEqual(result.usage["output_tokens"], 1)
                    self.assertEqual(result.metadata["device"], "cpu")
                    self.assertEqual(len(result.metadata["chat_template_sha256"]), 64)
                finally:
                    _LOCAL_MODELS.pop((path, "cpu"), None)
