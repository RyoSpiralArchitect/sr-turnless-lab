import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from turnless.local_models import inspect_checkpoint, resolve_runtime, render_prompt, output_allowance, public_output
from turnless.protocol import Settings
from turnless.providers import LocalMLXProvider, LocalProvider, LocalHFProvider, _LOCAL_MODELS


class LocalFormatTests(unittest.TestCase):
    def test_mlx_quantization_routes_to_mlx_without_guessing_from_directory_name(self):
        config = {"quantization_config": {"bits": 4, "group_size": 32, "mode": "mxfp4"}}
        self.assertEqual(resolve_runtime(config, "auto"), "mlx")
        self.assertEqual(resolve_runtime({"quantization_config": {"quant_method": "mxfp4"}}, "auto"), "transformers")
        with self.assertRaisesRegex(ValueError, "MLX quantized"):
            resolve_runtime(config, "transformers")
        self.assertEqual(resolve_runtime({}, "mlx"), "mlx")  # explicit unquantized MLX

    def test_inspect_reads_only_metadata_and_reports_plain_base_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text('{"model_type":"gpt2","n_positions":1024}')
            info = inspect_checkpoint(tmp)
            self.assertEqual((info["runtime"], info["prompt_mode"], info["context_window"]),
                             ("transformers", "plain", 1024))
            self.assertFalse(info["weights_loaded"])
            with self.assertRaisesRegex(ValueError, "No native chat template"):
                inspect_checkpoint(tmp, prompt_mode="native")
            Path(tmp, "chat_template.jinja").write_text("template")
            self.assertEqual(inspect_checkpoint(tmp)["prompt_mode"], "native")
            self.assertIsInstance(LocalProvider()._backend(Settings(model=tmp)), LocalHFProvider)

    def test_no_template_uses_versioned_plain_prompt_without_mutating_tokenizer(self):
        tokenizer = Mock()
        tokenizer.chat_template = None
        tokenizer.get_chat_template.side_effect = ValueError("missing")
        text, metadata = render_prompt(tokenizer, "instructions", '{"context":"x"}', "auto")
        self.assertIn("Harness state:", text)
        self.assertTrue(text.endswith("Assistant decision:\n"))
        self.assertEqual(metadata["prompt_mode"], "plain")
        self.assertEqual(metadata["plain_prompt_version"], "1")
        self.assertIsNone(metadata["chat_template_sha256"])
        tokenizer.apply_chat_template.assert_not_called()
        with self.assertRaises(ValueError):
            render_prompt(tokenizer, "i", "p", "native")

    def test_short_context_caps_output_only_and_rejects_full_input(self):
        cap, receipt = output_allowance({"n_positions": 1024}, 700, 1024)
        self.assertEqual(cap, 324)
        self.assertEqual(receipt["requested_max_new_tokens"], 1024)
        self.assertEqual(receipt["effective_max_new_tokens"], 324)
        with self.assertRaisesRegex(ValueError, "no input was truncated"):
            output_allowance({"n_positions": 1024}, 1024, 128)

    def test_ambiguous_native_template_is_not_a_missing_template(self):
        tokenizer = Mock(chat_template={"a": "one", "b": "two"})
        tokenizer.get_chat_template.side_effect = ValueError("choose a template")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            render_prompt(tokenizer, "i", "p", "auto")
        _, metadata = render_prompt(tokenizer, "i", "p", "plain")
        self.assertEqual(metadata["prompt_mode"], "plain")

    def test_harmony_public_final_extraction_and_private_omission(self):
        for start, channel, message, end in (
            ("<|start|>", "<|channel|>", "<|message|>", "<|return|>"),
            ("<|im_start|>", "<|meta_sep|>", "<|im_sep|>", "<|fim_suffix|>"),
        ):
            value = channel + "analysis" + message + "private" + start + "assistant" + channel + "final" + message + '{"action":"yield"}' + end
            raw, metadata = public_output(value, "gpt_oss")
            self.assertEqual(raw, '{"action":"yield"}')
            self.assertTrue(metadata["private_reasoning_omitted"])
        for invalid in ("private", '<|channel|>analysis<|message|>{"action":"yield"}',
                        "<|channel|>final<|message|>a<|start|>user",
                        "<|channel|>final<|message|>a<|return|>extra",
                        "<|channel|>final<|message|>a<|channel|>final<|message|>b"):
            with self.assertRaises(ValueError):
                public_output(invalid, "gpt_oss")

    def test_incomplete_private_block_is_not_returned_as_raw(self):
        with self.assertRaisesRegex(ValueError, "private reasoning"):
            public_output("<think>private", "other")
        raw, _ = public_output("<think>private</think>{}", "other")
        self.assertEqual(raw, "{}")

    def test_mlx_load_keeps_tokenizer_offline(self):
        load = Mock(return_value=("model", "tokenizer", {}))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text("{}")
            with patch.dict("sys.modules", {"mlx_lm": SimpleNamespace(load=load)}):
                self.assertEqual(LocalMLXProvider()._load(tmp), ("tokenizer", "model", {}))
            load.assert_called_once_with(tmp, tokenizer_config={
                "local_files_only": True, "trust_remote_code": False}, return_config=True)

    def test_mlx_tokens_eos_and_cancellation(self):
        for cancel_after_first in (False, True):
            with self.subTest(cancel=cancel_after_first), tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp).resolve())
                tokenizer = Mock(eos_token_ids={9})
                tokenizer.get_chat_template.return_value = "native"
                tokenizer.apply_chat_template.return_value = "prompt"
                tokenizer.encode.return_value = [1, 2]
                tokenizer.decode.return_value = "{}"
                cancel, closed = threading.Event(), threading.Event()
                def generate(*args, **kwargs):
                    try:
                        if cancel_after_first:
                            cancel.set()
                        yield SimpleNamespace(token=8, finish_reason=None)
                        yield SimpleNamespace(token=9, finish_reason="stop")
                    finally:
                        closed.set()
                stream = Mock(side_effect=generate)
                sampler = Mock(return_value="greedy")
                with patch.object(LocalMLXProvider, "_load", return_value=(tokenizer, object(), {"n_positions": 16})), \
                     patch.dict("sys.modules", {"mlx_lm": SimpleNamespace(stream_generate=stream),
                                               "mlx_lm.sample_utils": SimpleNamespace(make_sampler=sampler)}), \
                     patch("importlib.metadata.version", return_value="test"):
                    result = LocalMLXProvider().generate("i", "p", Settings(model=path, max_output_tokens=128), cancel)
                self.assertEqual(result.finish_reason, "cancelled" if cancel_after_first else "eos")
                self.assertTrue(closed.is_set())
                self.assertEqual(result.metadata["effective_max_new_tokens"], 14)
                self.assertEqual(stream.call_args.args[2], [1, 2])
                sampler.assert_called_once_with(temp=0)
                tokenizer.encode.assert_called_once_with("prompt", add_special_tokens=False)
                _LOCAL_MODELS.pop((path, "mlx"), None)
