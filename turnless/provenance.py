"""Version receipts from public source files only, without inspecting secrets."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from . import __version__
from .catalog import ADAPTER_VERSION
from .annotations import ANNOTATION_VERSION
from .protocol import PROTOCOL_VERSION, SCHEMA

TRACE_VERSION = "2"

def _source_manifest() -> dict:
    package = Path(__file__).resolve().parent
    files = sorted([*package.glob("*.py"), *package.glob("web/*.js"),
                    *package.glob("web/*.html"), *package.glob("web/*.css")])
    return {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


_STARTUP_MANIFEST = _source_manifest()


def source_identity() -> dict:
    manifest = _source_manifest()
    if manifest != _STARTUP_MANIFEST:
        raise ValueError("Source files changed since server startup. Restart before creating an experiment.")
    result = {
        "version": __version__, "trace_version": TRACE_VERSION,
        "protocol_version": PROTOCOL_VERSION, "adapter_version": ADAPTER_VERSION,
        "annotation_version": ANNOTATION_VERSION,
        "python": platform.python_version(),
        "source_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "source_files": manifest,
        "schema_sha256": hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest(),
        "git_commit": None, "git_dirty": None,
    }
    root = Path(__file__).resolve().parent.parent
    # An installed wheel or an untracked folder must not borrow a parent's Git ID.
    if (root / ".git").exists():
        try:
            result["git_commit"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                text=True, timeout=3).strip()
            result["git_dirty"] = bool(subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                stderr=subprocess.DEVNULL, text=True, timeout=3).strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return result
