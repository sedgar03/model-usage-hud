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

        self.assertEqual(state.collapsed, {"claude": False, "codex": False, "gemini": False, "system": False})
        self.assertEqual(state.window_position, (40, 40))

    def test_corrupt_state_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui-state.json"
            path.write_text("{not json")
            state = load_ui_state(path)

        self.assertEqual(state.collapsed, {"claude": False, "codex": False, "gemini": False, "system": False})
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

        self.assertEqual(
            state.collapsed,
            {"claude": True, "codex": False, "gemini": False, "system": False},
        )
        self.assertEqual(state.window_position, (120, 240))

    def test_saves_round_trippable_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "ui-state.json"
            saved = save_ui_state(
                UiState(
                    collapsed={"claude": False, "codex": True, "gemini": True},
                    window_position=(12, 34),
                    selected_providers={"claude", "codex"},
                ),
                path,
            )

            self.assertTrue(saved)
            loaded = load_ui_state(path)

        self.assertEqual(loaded.collapsed["codex"], True)
        self.assertEqual(loaded.window_position, (12, 34))
        self.assertEqual(loaded.selected_providers, {"claude", "codex"})

    def test_font_size_round_trip_and_coercion(self) -> None:
        # ``font_size`` is an optional zoom override — None means "use
        # the CLI default", a positive float is a persisted zoom level,
        # anything else must coerce back to None so a hand-edited state
        # file can't strand the user at 0pt.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui-state.json"

            save_ui_state(UiState(font_size=14.0), path)
            self.assertEqual(load_ui_state(path).font_size, 14.0)
            # And the key is present in the serialized JSON.
            self.assertIn("font_size", json.loads(path.read_text()))

            save_ui_state(UiState(font_size=None), path)
            self.assertIsNone(load_ui_state(path).font_size)
            # ``None`` should be omitted so future field removals don't
            # leave dead keys in the file.
            self.assertNotIn("font_size", json.loads(path.read_text()))

            for bad in ("huge", -1, 0, [], {}):
                path.write_text(json.dumps({"font_size": bad}))
                self.assertIsNone(
                    load_ui_state(path).font_size,
                    msg=f"bad font_size {bad!r} should coerce to None",
                )

    def test_selected_providers_round_trip_and_coercion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui-state.json"

            save_ui_state(UiState(selected_providers={"gemini", "codex"}), path)
            self.assertEqual(load_ui_state(path).selected_providers, {"codex", "gemini"})
            self.assertEqual(
                json.loads(path.read_text())["selected_providers"],
                ["codex", "gemini"],
            )

            save_ui_state(UiState(selected_providers=None), path)
            self.assertIsNone(load_ui_state(path).selected_providers)
            self.assertNotIn("selected_providers", json.loads(path.read_text()))

            for bad in ("codex", [], ["bad"], {"claude": True}, None):
                path.write_text(json.dumps({"selected_providers": bad}))
                self.assertIsNone(
                    load_ui_state(path).selected_providers,
                    msg=f"bad selected_providers {bad!r} should coerce to None",
                )

            path.write_text(json.dumps({"selected_providers": ["bad", "claude"]}))
            self.assertEqual(load_ui_state(path).selected_providers, {"claude"})


if __name__ == "__main__":
    unittest.main()
