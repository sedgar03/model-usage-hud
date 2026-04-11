from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_usage_hud.core.models import UiState
from model_usage_hud.core.state import load_ui_state, save_ui_state


class UiStateTests(unittest.TestCase):
    def test_missing_state_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = load_ui_state(Path(tmp) / "missing.json")

        self.assertEqual(state.collapsed, {"claude": False, "codex": False, "gemini": False})
        self.assertEqual(state.window_position, (40, 40))

    def test_corrupt_state_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui-state.json"
            path.write_text("{not json")
            state = load_ui_state(path)

        self.assertEqual(state.collapsed, {"claude": False, "codex": False, "gemini": False})
        self.assertEqual(state.window_position, (40, 40))

    def test_loads_known_fields_and_ignores_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui-state.json"
            path.write_text(
                json.dumps(
                    {
                        "collapsed": {
                            "claude": True,
                            "codex": "bad",
                            "gemini": False,
                            "extra": True,
                        },
                        "window_position": ["120", 240],
                    }
                )
            )
            state = load_ui_state(path)

        self.assertEqual(state.collapsed, {"claude": True, "codex": False, "gemini": False})
        self.assertEqual(state.window_position, (120, 240))

    def test_saves_round_trippable_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "ui-state.json"
            saved = save_ui_state(
                UiState(
                    collapsed={"claude": False, "codex": True, "gemini": True},
                    window_position=(12, 34),
                ),
                path,
            )

            self.assertTrue(saved)
            loaded = load_ui_state(path)

        self.assertEqual(loaded.collapsed["codex"], True)
        self.assertEqual(loaded.window_position, (12, 34))


if __name__ == "__main__":
    unittest.main()
