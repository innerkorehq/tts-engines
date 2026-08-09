"""
Shared stdin/stdout JSON protocol for per-engine `synth.py` CLIs.

Each engine's synth.py is invoked as:

    <engine>/.venv/bin/python <engine>/synth.py

It reads ONE JSON request object from stdin, does its work, and writes ONE
JSON response object to stdout. All logging/diagnostics go to stderr so they
never corrupt the stdout protocol stream.

This module has zero third-party dependencies so it can be copied verbatim
into every engine's directory (each engine venv is isolated — no shared
import path across tts-engines/* projects).
"""
import contextlib
import json
import sys


def read_request() -> dict:
    raw = sys.stdin.read()
    return json.loads(raw) if raw.strip() else {}


def write_response(ok: bool, **fields) -> None:
    payload = {"ok": ok, **fields}
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def fail(error: str) -> None:
    write_response(False, error=error)
    sys.exit(1)


def succeed(**fields) -> None:
    write_response(True, **fields)


@contextlib.contextmanager
def quiet_stdout():
    """Redirect Python-level stdout to stderr for the duration of the block.

    Some libraries (f5_tts, pydload, ai4bharat-transliteration) print
    progress/status messages directly to stdout via `print()`, which would
    corrupt the single-line JSON response this CLI contract requires.
    """
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real_stdout
