"""Tests for the bounded Codex session scan.

Regression guard for the "HUD takes 20s+ to show anything" bug: the scan must
read only the newest sessions, from their tail, so it stays fast no matter how
many tens of GB of Codex history have accumulated.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import usage_hud


def _event(limit_id: str, ts: str, used: float) -> dict:
    return {
        "type": "event_msg",
        "timestamp": ts,
        "payload": {
            "type": "token_count",
            "rate_limits": {
                "limit_id": limit_id,
                "primary": {"used_percent": used, "window_minutes": 300, "resets_at": 0},
            },
        },
    }


def _write_session(directory: Path, name: str, events: list, mtime: float) -> Path:
    path = directory / name
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    os.utime(path, (mtime, mtime))
    return path


class CodexScanTests(unittest.TestCase):
    def test_latest_timestamp_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_session(d, "a.jsonl", [_event("codex", "2026-01-01T00:00:00Z", 10)], 1000)
            _write_session(d, "b.jsonl", [_event("codex", "2026-02-01T00:00:00Z", 90)], 2000)
            latest = usage_hud.load_latest_codex_snapshots(d)
        self.assertIn("codex", latest)
        self.assertEqual(latest["codex"].primary.used_percent, 90.0)

    def test_caps_to_newest_sessions(self) -> None:
        # Files older than the cap must be skipped entirely. Put a unique bucket
        # only in the oldest files and assert it never surfaces.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            total = usage_hud.CODEX_MAX_SESSIONS_SCANNED + 3
            for i in range(total):
                bucket = "codex" if i >= 3 else "stale-bucket"
                _write_session(
                    d, f"s{i:02d}.jsonl",
                    [_event(bucket, f"2026-03-{i+1:02d}T00:00:00Z", 50)],
                    mtime=1000 + i,  # higher index = newer
                )
            latest = usage_hud.load_latest_codex_snapshots(d)
        self.assertIn("codex", latest)
        self.assertNotIn("stale-bucket", latest)  # oldest 3 beyond the cap

    def test_reads_from_tail_only(self) -> None:
        # With a tiny tail budget and a large junk prefix, the valid event at
        # the end must still be found without reading the whole file.
        original = usage_hud.CODEX_SESSION_TAIL_BYTES
        usage_hud.CODEX_SESSION_TAIL_BYTES = 300
        try:
            with tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                path = d / "big.jsonl"
                junk = "x" * 5000  # far larger than the 300-byte tail window
                event_line = json.dumps(_event("codex", "2026-04-01T00:00:00Z", 42))
                path.write_text(junk + "\n" + event_line + "\n")
                os.utime(path, (5000, 5000))
                latest = usage_hud.load_latest_codex_snapshots(d)
        finally:
            usage_hud.CODEX_SESSION_TAIL_BYTES = original
        self.assertIn("codex", latest)
        self.assertEqual(latest["codex"].primary.used_percent, 42.0)

    def test_missing_dir_is_empty(self) -> None:
        self.assertEqual(
            usage_hud.load_latest_codex_snapshots(Path("/no/such/dir")), {}
        )


if __name__ == "__main__":
    unittest.main()
