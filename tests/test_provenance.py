import unittest
from unittest.mock import patch

from turnless.provenance import source_identity
from turnless.protocol import Settings, make_prompt


class ProvenanceTests(unittest.TestCase):
    def test_source_change_requires_restart(self):
        with patch("turnless.provenance._source_manifest", return_value={"engine.py": "changed"}):
            with self.assertRaisesRegex(ValueError, "Restart"):
                source_identity()

    def test_identity_has_versions_and_hashes(self):
        identity = source_identity()
        self.assertEqual(identity["version"], "0.2.0")
        self.assertEqual(identity["trace_version"], "2")
        self.assertEqual(len(identity["schema_sha256"]), 64)
        self.assertIn("web/i18n.js", identity["source_files"])

    def test_conversation_language_is_explicit(self):
        en, _ = make_prompt(Settings(), [], "session_open", 0)
        ja, _ = make_prompt(Settings(conversation_language="ja"), [], "session_open", 0)
        self.assertIn("Conversation language: English", en)
        self.assertIn("Conversation language: Japanese", ja)
        self.assertNotEqual(en, ja)
        self.assertNotIn("ui_language", Settings().public())
