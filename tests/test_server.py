import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from turnless.server import App, make_handler


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = App(Path(cls.tmp.name))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.app))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        for session in cls.app.sessions.values():
            session.close()
        cls.server.shutdown(); cls.server.server_close(); cls.tmp.cleanup()

    def call(self, path, data=None, token=True, host=None):
        headers = {"Content-Type": "application/json"}
        if token: headers["X-Turnless-Token"] = self.app.token
        if host: headers["Host"] = host
        req = urllib.request.Request(self.base + path, data=None if data is None else json.dumps(data).encode(), headers=headers)
        return urllib.request.urlopen(req, timeout=3)

    def test_bootstrap(self):
        with self.call("/api/bootstrap", token=False) as response:
            data = json.load(response)
        self.assertEqual(data["token"], self.app.token)
        self.assertFalse(data["openai_ready"])
        self.assertTrue(data["providers"]["fixture"])
        self.assertTrue(data["providers"]["local"])
        self.assertFalse(data["providers"]["anthropic"])
        self.assertFalse(data["providers"]["openai"])

    def test_anthropic_without_key_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.call("/api/session", {"provider": "anthropic", "model": "claude-x", "consent": True})
        self.assertEqual(ctx.exception.code, 400)

    def test_compat_session_accepts_loopback_base_url(self):
        with self.call("/api/session", {"provider": "openai_compat", "model": "m", "consent": True,
                                        "compat_base_url": "http://127.0.0.1:9/v1", "proactive": False}) as response:
            sid = json.load(response)["session_id"]
        with self.call("/api/close", {"session": sid}) as response:
            self.assertEqual(json.load(response)["status"], "closed")

    def test_missing_token_blocked(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.call("/api/session", {}, token=False)
        self.assertEqual(ctx.exception.code, 403)

    def test_rebinding_host_blocked(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.call("/api/bootstrap", host="attacker.example")
        self.assertEqual(ctx.exception.code, 403)

    def test_static_no_path_traversal(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.call("/../server.py")
        self.assertEqual(ctx.exception.code, 404)

    def test_session_human_annotate_export(self):
        with self.call("/api/session", {"proactive": False}) as response:
            sid = json.load(response)["session_id"]
        with self.call("/api/human", {"session": sid, "text": "hello", "auto_reply": False}) as response:
            snapshot = json.load(response)
        utterance = next(e for e in snapshot["events"] if e["kind"] == "utterance")
        with self.call("/api/annotate", {"session": sid, "target_seq": utterance["seq"], "tag": "other", "note": "test"}) as response:
            self.assertEqual(json.load(response)["metrics"]["annotations"], 1)
        with self.call(f"/api/export?session={sid}") as response:
            rows = [json.loads(line) for line in response.read().decode().splitlines()]
        self.assertEqual(rows[-1]["kind"], "export_metadata")
        self.assertEqual(snapshot["metrics"]["calls"], 0)
        annotation = next(e for e in rows if e["kind"] == "annotation")
        self.assertEqual(annotation["target_seqs"], [utterance["seq"]])
        with self.call("/api/annotate", {"session": sid, "target_seq": utterance["seq"],
                                       "judgment": "unsure", "supersedes": annotation["seq"]}) as response:
            revised = json.load(response)
        self.assertEqual(revised["events"][-1]["judgment"], "unsure")
        self.assertIsNone(revised["events"][-1]["tag"])
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.call("/api/annotate", {"session": sid, "target_seq": 1, "tag": "other"})
        self.assertEqual(ctx.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.call("/api/annotate", {"session": sid, "target_seq": utterance["seq"], "tag": "premature-yield"})
        self.assertEqual(ctx.exception.code, 400)
        with self.call("/api/close", {"session": sid}) as response:
            self.assertEqual(json.load(response)["status"], "closed")


if __name__ == "__main__":
    unittest.main()
