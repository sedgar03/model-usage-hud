"""Tests for the Kimi (Moonshot) usage normalizer.

The parser turns the real ``/coding/v1/usages`` response into the Claude-style
{five_hour, weekly} shape. Field values arrive as strings and utilization is
derived as (limit - remaining) / limit.
"""

from __future__ import annotations

import unittest

import usage_hud

# Trimmed real-shape payload (values chosen so utilization is non-trivial).
PAYLOAD = {
    "user": {"membership": {"level": "LEVEL_BASIC"}},
    "usage": {"limit": "100", "remaining": "80", "resetTime": "2026-07-25T16:35:08.6Z"},
    "limits": [
        {
            "window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"},
            "detail": {"limit": "100", "remaining": "50", "resetTime": "2026-07-19T02:35:08.6Z"},
        }
    ],
}


class KimiNormalizeTests(unittest.TestCase):
    def test_maps_windows_and_utilization(self) -> None:
        n = usage_hud.normalize_kimi_payload(PAYLOAD)
        self.assertEqual(n["weekly"]["utilization"], 20)  # (100-80)/100
        self.assertEqual(n["five_hour"]["utilization"], 50)  # (100-50)/100
        self.assertEqual(n["five_hour"]["window_minutes"], 300)
        self.assertEqual(n["membership"], "LEVEL_BASIC")
        self.assertEqual(n["weekly"]["resets_at"], "2026-07-25T16:35:08.6Z")

    def test_picks_shortest_window_as_five_hour(self) -> None:
        payload = {
            "usage": {"limit": "100", "remaining": "100", "resetTime": "z"},
            "limits": [
                {"window": {"duration": 7, "timeUnit": "TIME_UNIT_DAY"},
                 "detail": {"limit": "100", "remaining": "90", "resetTime": "w"}},
                {"window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"},
                 "detail": {"limit": "100", "remaining": "70", "resetTime": "s"}},
            ],
        }
        n = usage_hud.normalize_kimi_payload(payload)
        self.assertEqual(n["five_hour"]["window_minutes"], 300)
        self.assertEqual(n["five_hour"]["utilization"], 30)

    def test_missing_and_garbage_are_safe(self) -> None:
        n = usage_hud.normalize_kimi_payload({})
        self.assertIsNone(n["weekly"])
        self.assertIsNone(n["five_hour"])
        self.assertIsNone(n["membership"])
        # A limit of 0 must not divide-by-zero.
        n2 = usage_hud.normalize_kimi_payload(
            {"usage": {"limit": "0", "remaining": "0", "resetTime": "z"}}
        )
        self.assertIsNone(n2["weekly"]["utilization"])

    def test_missing_credentials_returns_none(self) -> None:
        original = usage_hud.KIMI_CREDENTIALS_PATH
        usage_hud.KIMI_CREDENTIALS_PATH = original.with_name("does-not-exist.json")
        try:
            self.assertIsNone(usage_hud._kimi_read_token())
        finally:
            usage_hud.KIMI_CREDENTIALS_PATH = original


if __name__ == "__main__":
    unittest.main()
