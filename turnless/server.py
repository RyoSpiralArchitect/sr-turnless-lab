"""Loopback-only local HTTP server. Never deploy this research server publicly."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .engine import Session
from .protocol import TAGS, Settings, _validate_compat_base_url
from .providers import create_provider
from .catalog import BACKENDS
from . import __version__

WEB = Path(__file__).parent / "web"


class App:
    def __init__(self, log_dir: Path, key: str = "", ollama_url: str = "http://127.0.0.1:11434",
                 anthropic_key: str = "", compat_key: str = "", compat_base_url: str = "",
                 gemini_key: str = "", mistral_key: str = ""):
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.log_dir, self.key, self.ollama_url = log_dir, key, ollama_url
        self.anthropic_key, self.compat_key = anthropic_key, compat_key
        self.compat_base_url = compat_base_url
        self.credentials = {"openai": key, "anthropic": anthropic_key, "openai_compat": compat_key,
                            "gemini": gemini_key, "mistral": mistral_key}

    def create(self, data: dict[str, Any]) -> Session:
        settings = Settings.from_dict(data)
        with self.lock:
            # Retire abandoned tabs. Disk traces remain available locally.
            expired = [sid for sid, s in self.sessions.items()
                       if s.closed and not s.inflight or time.monotonic() - s.last_access > 1800]
            for sid in expired:
                self.sessions.pop(sid).close()
            if len(self.sessions) >= 12:
                raise ValueError("At most 12 concurrent experiments. Close unused experiments first.")
            provider = create_provider(settings, self.credentials, self.ollama_url)
            session = Session(settings, provider, self.log_dir)
            self.sessions[session.id] = session
            return session

    def get(self, sid: str) -> Session:
        with self.lock:
            session = self.sessions.get(sid)
            if session is None:
                raise ValueError("Experiment not found. Start a new experiment.")
            session.last_access = time.monotonic()
            return session


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # Avoid logging private transcript text or request payloads to console.
            pass

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "")
            port = self.server.server_address[1]
            return host in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _send(self, status: int, data: bytes, content_type: str = "application/json; charset=utf-8",
                  attachment: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _authorized(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-Turnless-Token", ""), app.token)

        def do_GET(self) -> None:
            if not self._host_ok():
                self._json(403, {"error": "Loopback host required"})
                return
            route = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(route.query)
            if route.path == "/api/bootstrap":
                self._json(200, {"token": app.token, "defaults": Settings().public(), "tags": TAGS,
                                 "openai_ready": bool(app.key), "version": __version__,
                                 "providers": {key: bool(app.credentials.get(key)) or not spec.key_env
                                               or key == "openai_compat" for key, spec in BACKENDS.items()},
                                 "backends": [spec.public() for spec in BACKENDS.values()],
                                 "compat_base_url": app.compat_base_url})
                return
            if route.path.startswith("/api/"):
                if not self._authorized():
                    self._json(403, {"error": "Missing session token"})
                    return
                try:
                    session = app.get(query.get("session", [""])[0])
                    if route.path == "/api/events":
                        after = int(query.get("after", ["0"])[0])
                        self._json(200, session.snapshot(max(0, after)))
                    elif route.path == "/api/export":
                        self._send(200, session.export_jsonl().encode("utf-8"),
                                   "application/x-ndjson; charset=utf-8", f"turnless-{session.id[:8]}.jsonl")
                    else:
                        self._json(404, {"error": "Unknown endpoint"})
                except (ValueError, TypeError) as exc:
                    self._json(400, {"error": str(exc)})
                return
            filename = {"/": "index.html", "/app.js": "app.js", "/i18n.js": "i18n.js",
                        "/style.css": "style.css"}.get(route.path)
            if filename is None:
                self._json(404, {"error": "Not found"})
                return
            data = (WEB / filename).read_bytes()
            mime = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[Path(filename).suffix]
            self._send(200, data, mime + "; charset=utf-8")

        def do_POST(self) -> None:
            if not self._host_ok() or not self._authorized():
                self._json(403, {"error": "Local authenticated request required"})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self._json(415, {"error": "application/json required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 40_000:
                    raise ValueError("invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("JSON object required")
                route = urllib.parse.urlparse(self.path).path
                if route == "/api/session":
                    session = app.create(data)
                else:
                    session = app.get(data.get("session", ""))
                    if route == "/api/human":
                        session.human(data.get("text", ""), data.get("auto_reply", True))
                    elif route == "/api/interrupt":
                        session.interrupt(data.get("reason", "operator_stop"))
                    elif route == "/api/wake":
                        session.wake()
                    elif route == "/api/close":
                        session.close()
                    elif route == "/api/annotate":
                        session.annotate(data.get("target_seq"), data.get("tag"), data.get("note", ""))
                    else:
                        self._json(404, {"error": "Unknown endpoint"})
                        return
                self._json(200, session.snapshot())
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": str(exc)})
            except OSError:
                self._json(500, {"error": "Cannot write the trace or local file. Check the output directory."})

    return Handler


def read_key(path: Path | None, env_name: str = "OPENAI_API_KEY",
              names: tuple[str, ...] = ("openai", "api_key", "OPENAI_API_KEY", "key")) -> str:
    if path is None:
        return os.environ.get(env_name, "").strip()
    text = path.expanduser().read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        obj = json.loads(text)
        text = next((obj[name] for name in names if obj.get(name)), "")
    if not isinstance(text, str) or not text or any(c.isspace() for c in text):
        raise ValueError("Invalid credential file format. Keep credentials out of chat and source control.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Turnless Lab: local inference-time experiment")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--key-file", type=Path, help="key file from the secure setup flow; never put keys in chat")
    parser.add_argument("--anthropic-key-file", type=Path, help="Anthropic key file or ANTHROPIC_API_KEY")
    parser.add_argument("--compat-key-file", type=Path, help="OpenAI-compatible key file or OPENAI_COMPAT_API_KEY")
    parser.add_argument("--gemini-key-file", type=Path, help="Gemini credential file or GEMINI_API_KEY")
    parser.add_argument("--mistral-key-file", type=Path, help="Mistral credential file or MISTRAL_API_KEY")
    parser.add_argument("--compat-base-url", default=os.environ.get("OPENAI_COMPAT_BASE_URL", ""),
                        help="default base URL for the OpenAI-compatible provider (https, or loopback http)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    parsed = urllib.parse.urlparse(ollama_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        parser.error("OLLAMA_HOST must be a loopback HTTP URL")
    if args.compat_base_url:
        try:
            _validate_compat_base_url(args.compat_base_url)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        key = read_key(args.key_file)
        anthropic_key = read_key(args.anthropic_key_file, "ANTHROPIC_API_KEY",
                                 ("anthropic", "ANTHROPIC_API_KEY", "api_key", "key"))
        compat_key = read_key(args.compat_key_file, "OPENAI_COMPAT_API_KEY",
                              ("compat", "openai_compat", "OPENAI_COMPAT_API_KEY", "api_key", "key"))
        app = App(args.log_dir.resolve(), key, ollama_url, anthropic_key, compat_key,
                  args.compat_base_url.strip(),
                  read_key(args.gemini_key_file, "GEMINI_API_KEY", ("gemini", "GEMINI_API_KEY", "api_key", "key")),
                  read_key(args.mistral_key_file, "MISTRAL_API_KEY", ("mistral", "MISTRAL_API_KEY", "api_key", "key")))
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"\n  SR Turnless Lab  ∴  http://127.0.0.1:{args.port}\n")
    print(f"  Trace directory: {app.log_dir}")
    print("  Default: deterministic fixture, not a real model. Ctrl+C to stop.\n")
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        with app.lock:
            for session in app.sessions.values():
                session.close()
        server.server_close()


if __name__ == "__main__":
    main()
