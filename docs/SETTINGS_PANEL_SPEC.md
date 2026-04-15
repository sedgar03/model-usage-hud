# Settings Panel — Right-Click Expandable Bottom Drawer

**Date**: 2026-04-13
**Status**: Ready for implementation
**Target**: PySide6 desktop app (`model_usage_hud/app/`)

## Summary

Add a collapsible settings panel to the bottom of the HUD window, toggled by
right-click (two-finger click on trackpad). The panel slides into view below
the existing metric grid. First control: a voice picker dropdown for selecting
the Kokoro TTS voice.

## Behavior

### Toggle

- **Trigger**: `contextMenuEvent` on the `HudWindow` (right-click / two-finger
  click anywhere on the window).
- **Action**: Toggle visibility of the settings panel. If currently hidden,
  show it and call `adjustSize()` so the window grows. If currently visible,
  hide it and `adjustSize()` so the window shrinks.
- **No native context menu**: Override `contextMenuEvent` to consume the event
  and toggle the panel. Do not show Qt's default context menu.
- **Keyboard shortcut**: Bind `,` (comma) as a secondary toggle — matches the
  macOS convention of `Cmd+,` for preferences, but since this is frameless we
  just use bare `,`.

### Panel chrome

- The panel is a `QFrame` with objectName `"settingsPanel"`.
- Background: `#1c1f23` — slightly lighter than the root `#111315` but darker
  than the card `#171a1d`, so the drawer reads as a distinct layer without
  clashing. Add this as `COLORS["drawer"]` in `styles.py`.
- Top border: 1px solid `#2a2d30` (existing `border` color) to separate from
  the grid above.
- Border radius: 0 top, 8px bottom (matches the root frame's bottom corners).
- Internal padding: 6px horizontal, 4px vertical.
- The panel lives inside the existing `_root_frame`'s inner `QVBoxLayout`,
  appended after the `error_label`. It participates in the same layout flow
  so `adjustSize` works naturally.

### Animation (optional, skip for v1)

For v1, use plain `show()`/`hide()` with `adjustSize()`. A future version can
add a `QPropertyAnimation` on `maximumHeight` for a slide effect.

## Voice Picker Control

### Layout

Single row inside the settings panel:

```
[ person-talking icon ]  [ voice dropdown          v ]
```

- The icon is a Phosphor `user-sound` (bold weight) SVG, 14px, tinted
  `COLORS["fg"]`. Vendor the SVG into `assets/icons/user-sound.svg` from
  Phosphor Icons (MIT licensed, same as existing icons).
- If `user-sound` is not in the Phosphor Bold set, fall back to
  `user-circle` or `person` and post-fix the icon name in code.
- The icon sits in a `QLabel` with fixed size `UI_ICON_PX + 10` to match
  the top-bar control buttons' footprint.

### Dropdown

- Widget: `QComboBox` with objectName `"voiceCombo"`.
- Populate with the full Kokoro v1.0 voice list, grouped by section:

  ```
  ── American Female ──
  af_heart
  af_alloy
  af_aoede
  af_bella
  af_jessica
  af_kore
  af_nicole
  af_nova
  af_river
  af_sarah
  af_shimmer
  af_sky
  ── American Male ──
  am_adam
  am_echo
  am_eric
  am_fenrir
  am_liam
  am_michael
  am_onyx
  am_puck
  am_santa
  ── British Female ──
  bf_alice
  bf_emma
  bf_isabella
  bf_lily
  ── British Male ──
  bm_daniel
  bm_fable
  bm_george
  bm_lewis
  ```

  Section headers are non-selectable separator items (`QComboBox.insertSeparator`
  or a disabled item with the header text).

- **Current selection**: On init, read `~/.claude/voice-config.json`, parse
  `kokoro_voice`, and set the combo to that value. If the file doesn't exist
  or the key is missing, default to `af_heart`. If the stored voice is a
  blend spec (contains `:` or `,`), show it as-is in a non-editable text
  label above/below the combo (blends can't be picked from the list — just
  displayed). Set the combo to the first voice in the blend.

- **On change**: When the user selects a new voice:
  1. Read `~/.claude/voice-config.json` (create `{}` if missing).
  2. Set `kokoro_voice` to the selected value.
  3. Write the file back atomically (write to `.tmp`, rename).
  4. Do NOT restart the TTS daemon — the daemon re-reads config on each
     request, so the next spoken output will use the new voice.

### Styling

Add to `build_stylesheet()` in `styles.py`:

```css
QFrame#settingsPanel {
    background: #1c1f23;
    border-top: 1px solid #2a2d30;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}

QComboBox#voiceCombo {
    background: #111315;
    color: #f0f6fc;
    border: 1px solid #2a2d30;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: Menlo;
    min-width: 120px;
}

QComboBox#voiceCombo::drop-down {
    border: none;
    width: 16px;
}

QComboBox#voiceCombo QAbstractItemView {
    background: #1c1f23;
    color: #f0f6fc;
    border: 1px solid #2a2d30;
    selection-background-color: #2a2d30;
    selection-color: #f0f6fc;
}
```

The combo should look like a native dark-mode dropdown but with the HUD's
color palette — not the default Qt blue/white.

## Icon Asset

Download Phosphor Bold `user-sound.svg` from:
https://raw.githubusercontent.com/phosphor-icons/core/main/assets/bold/user-sound-bold.svg

Place at: `model_usage_hud/app/assets/icons/user-sound.svg`

Ensure the SVG uses `fill="currentColor"` so the existing `_tint_svg()` in
`icons.py` can recolor it. If the downloaded SVG uses a different fill
attribute, normalize it.

Add attribution to the existing `LICENSE-phosphor.txt` if needed (same
license, just a new icon from the same set).

## State Persistence

For v1, do **not** persist the panel's open/closed state. It starts hidden on
every launch. The voice selection is persisted via `voice-config.json` (the
handsfree config file), not `ui-state.json`.

## Files to modify

| File | Change |
|---|---|
| `model_usage_hud/app/styles.py` | Add `"drawer": "#1c1f23"` to `COLORS`, add `QFrame#settingsPanel` and `QComboBox#voiceCombo` rules to `build_stylesheet()` |
| `model_usage_hud/app/window.py` | Add settings panel construction in `_build_layout()`, add `contextMenuEvent` override, add voice-config read/write helpers, add `,` keyboard shortcut |
| `model_usage_hud/app/icons.py` | No changes needed — existing `ui_icon()` handles any new SVG in the icons dir |
| `model_usage_hud/app/assets/icons/user-sound.svg` | New file — Phosphor Bold icon |

## Testing

- Right-click toggles panel visibility.
- Window resizes correctly when panel appears/disappears.
- Dropdown shows all Kokoro voices grouped by category.
- Selecting a voice writes to `~/.claude/voice-config.json`.
- If `voice-config.json` doesn't exist, it's created.
- If `voice-config.json` has other keys, they're preserved.
- Blend specs in the config are displayed but don't break the combo.
- Panel looks correct in frameless + always-on-top mode.
- Drag-to-move still works when panel is open.

## Out of scope (future controls)

These are not part of this spec but are likely future additions to the
settings panel:

- Speed slider (`kokoro_speed`)
- Voice preset editor (blend specs)
- Summary backend picker
- Silence timeout slider
- Wake word configuration
