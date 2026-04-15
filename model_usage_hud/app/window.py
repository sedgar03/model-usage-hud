"""Minimal PySide6 desktop window for the usage HUD.

The window is a dense ``QGridLayout`` where column 0 is a provider logo
spanning that provider's rows, and the remaining columns are label, value,
pace bar, and detail text. Drops the old card-per-provider chrome and the
header title / timestamp row in favor of a tooltip on the whole frame.
"""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import time

import usage_hud
from model_usage_hud.app.icons import provider_icon, provider_pixmap, ui_icon, ui_pixmap
from model_usage_hud.app.styles import COLORS, build_stylesheet, color_for_style
from model_usage_hud.app.widgets.pace_bar import PaceBarWidget
from model_usage_hud.core.models import (
    MetricRow,
    NoteLine,
    ProviderName,
    ProviderSection,
    SnapshotBundle,
)
from model_usage_hud.core.state import load_ui_state, save_ui_state
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QFontMetrics, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


GEOMETRY_RE = re.compile(r"^(?P<w>\d+)x(?P<h>\d+)\+(?P<x>-?\d+)\+(?P<y>-?\d+)$")
MUTE_PATHS: dict[ProviderName, Path] = {
    "claude": Path.home() / ".claude" / "mute",
    "codex": Path.home() / ".codex" / "mute",
}
SPEECH_TOGGLE_PATH = Path.home() / ".handsfree" / "speech-enabled"
LEGACY_SPEECH_TOGGLE_PATH = Path.home() / ".claude" / "handsfree"
WAKE_TOGGLE_PATH = Path.home() / ".handsfree" / "wake-enabled"
CONSUME_AFTER_PATH = Path.home() / ".handsfree" / "consume-after"
VOICE_CONFIG_PATH = Path.home() / ".claude" / "voice-config.json"
SOUND_THEME_PATH = Path.home() / ".claude" / "theme"
SOUND_THEME_ROOT = Path.home() / ".claude" / "hooks" / "sounds"
HANDSFREE_REPO_ROOT = Path(
    os.environ.get("HANDSFREE_REPO_ROOT", str(Path.home() / "Code" / "handsfree"))
).expanduser()
PROVIDER_ORDER: tuple[ProviderName, ...] = ("claude", "codex", "gemini")
PROVIDER_LABELS: dict[ProviderName, str] = {
    "claude": "Claude",
    "codex": "Codex",
    "gemini": "Gemini",
}
KOKORO_VOICE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "American Female",
        (
            "af_heart",
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_shimmer",
            "af_sky",
        ),
    ),
    (
        "American Male",
        (
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
        ),
    ),
    (
        "British Female",
        (
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
        ),
    ),
    (
        "British Male",
        (
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
        ),
    ),
)
KOKORO_VOICES = tuple(
    voice for _group_name, voices in KOKORO_VOICE_GROUPS for voice in voices
)
DEFAULT_KOKORO_VOICE = "af_heart"
TTS_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("kokoro", "Kokoro"),
    ("chatterbox", "Chatterbox"),
)
DEFAULT_TTS_PROVIDER = "kokoro"
CHATTERBOX_VOICE_OPTIONS: tuple[tuple[str, str, bool], ...] = (
    ("p238 clone", "default", True),
)
DEFAULT_CHATTERBOX_VOICE = "default"
CHATTERBOX_STYLE_OPTIONS: tuple[tuple[str, str, bool], ...] = (
    ("Auto (context)", "auto", True),
    ("Off", "neutral", True),
    ("Force Happy", "happy", True),
    ("Force Sarcastic", "sarcastic", True),
    ("Force Dramatic", "dramatic", True),
    ("Force Narration", "narration", True),
    ("Force Angry", "angry", True),
    ("Force Whispering", "whispering", True),
    ("Force Surprised", "surprised", True),
    ("Force Sigh", "sigh", True),
    ("Force Chuckle", "chuckle", True),
    ("Force Laugh", "laugh", True),
)
DEFAULT_CHATTERBOX_STYLE = "auto"
INTERACTION_MODES: tuple[tuple[str, str, bool], ...] = (
    ("Off", "off", True),
    ("On Demand", "on_demand", True),
    ("Direct", "direct", True),
    ("Conductor", "conductor", True),
)
SPEECH_VERBOSITY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Original", "direct"),
    ("Detailed", "detailed"),
    ("Brief", "terse"),
    ("Tiny", "tiny"),
)
DEFAULT_SPEECH_VERBOSITY = "detailed"
CONDUCTOR_MODELS: tuple[tuple[str, str, bool], ...] = (
    ("Qwen 2B", "mlx-community/Qwen3.5-2B-OptiQ-4bit", True),
)
DEFAULT_INTERACTION_MODE = "off"
DEFAULT_ACTIVE_INTERACTION_MODE = "direct"

# Grid column indices — one source of truth for layout math.
# ``COL_METRIC`` used to be two columns (``COL_LABEL`` + ``COL_VALUE``)
# but keeping them separate forced a ~2-character gap between the label
# and the value (trailing empty in the label cell + leading empty in the
# right-aligned value cell). They're now a single cell containing a
# monospaced rich-text blob so the gap is exactly one space character
# while the trailing ``%`` of each row still line up vertically via a
# fixed-width pad on the value segment.
COL_LOGO = 0
COL_METRIC = 1
COL_BAR = 2
COL_DETAIL = 3
# Mute used to live in a trailing column, but providers had inconsistent
# support (Gemini has no mute file) and the bell column added dead space
# on every row. It is now a single global control in the top bar.

# Visual tuning.
PROVIDER_LOGO_PX = 22
UI_ICON_PX = 14
CONTROL_GROUP_GAP_PX = max(6, UI_ICON_PX // 2)
PACE_BAR_CELLS = 16
MAX_WINDOW_WIDTH_PX = 520


@dataclass(slots=True)
class AppConfig:
    selected_providers: set[str]
    codex_sessions_dir: Path
    all_limits: bool
    interval_seconds: float
    always_on_top: bool
    frameless: bool
    font_size: float
    # ``None`` means "auto-fit from sizeHint after the first refresh". Any
    # explicit geometry (user-provided or remembered) is honored as-is.
    geometry: str | None
    geometry_explicit: bool
    force: bool
    # True when the user passed ``--font-size`` on the CLI. A persisted
    # zoom level in UiState should lose to an explicit CLI override but
    # win over the compiled-in default.
    font_size_explicit: bool = False
    providers_explicit: bool = False


def _style_name_for_metric(row: MetricRow) -> str:
    if row.utilization is None:
        return "yellow"
    if row.stale:
        return "orange"
    return usage_hud.usage_style(int(round(float(row.utilization))))


def _style_name_for_delta(row: MetricRow) -> str:
    if row.delta is None:
        return "dim"
    if row.stale:
        return "orange"
    return usage_hud.pace_style(int(round(float(row.delta))))


def _format_detail_html(row: MetricRow) -> str:
    """Pack delta/target/speedometer into a colored rich-text detail cell.

    The delta uses the pace color (green/yellow/red) so it pops, while
    the parenthetical expected value is muted grey — the expected pct is
    reference context, not the primary signal. Returns an empty string
    when the row has no delta or target to display.
    """

    parts: list[str] = []
    if row.delta is not None and row.expected_utilization is not None:
        delta_value = row.delta if row.delta is not None else 0
        delta_text = html.escape(usage_hud.fmt_delta(delta_value).strip())
        delta_color = color_for_style(_style_name_for_delta(row))
        target_text = html.escape(
            f"({usage_hud.fmt_pct(row.expected_utilization).strip()})"
        )
        parts.append(
            f'<span style="color:{delta_color}">{delta_text}</span>'
            f'&nbsp;<span style="color:{COLORS["muted"]}">{target_text}</span>'
        )
    if usage_hud.SPEEDOMETER_ENABLED and row.burn_rate_per_hour is not None:
        eta_text = ""
        if row.eta_hours is not None:
            eta_text = f" {usage_hud._format_eta(row.eta_hours)}"
        speed = html.escape(f"\u23F1 +{row.burn_rate_per_hour:.0f}%/h{eta_text}")
        parts.append(f'<span style="color:{COLORS["muted"]}">{speed}</span>')
    return " ".join(parts)


def _clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


def _mark_consume_after(timestamp: float | None = None) -> None:
    value = time.time() if timestamp is None else float(timestamp)
    CONSUME_AFTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSUME_AFTER_PATH.write_text(f"{value:.6f}\n")
    os.utime(CONSUME_AFTER_PATH, (value, value))


def _parse_geometry(geometry: str) -> tuple[int, int, int, int]:
    match = GEOMETRY_RE.match(geometry.strip())
    if not match:
        raise ValueError(f"Invalid geometry string: {geometry}")
    return (
        int(match.group("w")),
        int(match.group("h")),
        int(match.group("x")),
        int(match.group("y")),
    )


class RefreshSignals(QObject):
    """Signals emitted by the background refresh worker."""

    succeeded = Signal(object, object)
    failed = Signal(str)


class RefreshWorker(QRunnable):
    """Fetch provider sections without blocking the GUI thread."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.signals = RefreshSignals()

    def run(self) -> None:
        try:
            bundle, sections = usage_hud.fetch_provider_section_models(
                selected_providers=self.config.selected_providers,
                codex_sessions_dir=self.config.codex_sessions_dir,
                all_limits=self.config.all_limits,
            )
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(str(exc))
            return
        self.signals.succeeded.emit(bundle, sections)


class HandsfreeActionSignals(QObject):
    """Signals emitted by background Handsfree control commands."""

    succeeded = Signal(str, object)
    failed = Signal(str, str)


def _uv_command() -> str:
    for candidate in (
        os.environ.get("UV"),
        shutil.which("uv"),
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError("uv not found")


class HandsfreeActionWorker(QRunnable):
    """Run a Handsfree broker command without blocking the HUD."""

    def __init__(
        self,
        action: str,
        broker_args: list[str],
        *,
        timeout: float = 120.0,
    ):
        super().__init__()
        self.action = action
        self.broker_args = broker_args
        self.timeout = timeout
        self.signals = HandsfreeActionSignals()

    def run(self) -> None:
        try:
            if not (HANDSFREE_REPO_ROOT / "src" / "broker.py").exists():
                raise RuntimeError(f"Handsfree repo not found: {HANDSFREE_REPO_ROOT}")
            env = {
                **os.environ,
                "PYTHONPATH": str(HANDSFREE_REPO_ROOT / "src"),
            }
            result = subprocess.run(
                [_uv_command(), "run", "python", "-m", "broker", *self.broker_args],
                cwd=str(HANDSFREE_REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.action, str(exc))
            return

        if result.returncode != 0:
            message = (result.stderr or result.stdout or "Handsfree command failed").strip()
            self.signals.failed.emit(self.action, message)
            return
        self.signals.succeeded.emit(self.action, result.stdout)


class HudWindow(QWidget):
    """Minimal always-on-top HUD window driven by provider view models.

    The top-level widget is a plain ``QWidget``; the visible chrome lives
    inside a child ``QFrame#root`` that owns the rounded border, background
    fill, and content layout. This split is required so frameless mode can
    set ``WA_TranslucentBackground`` on the window while the inner frame's
    stylesheet background still paints cleanly to a fixed rect — Qt won't
    paint a stylesheet background on a top-level translucent widget.
    """

    # Zoom bounds — the QFontMetrics columns go unreadable below ~7pt
    # and start stealing screen real estate above 22pt.
    MIN_FONT_SIZE = 8.0
    MAX_FONT_SIZE = 22.0
    ZOOM_STEP = 1.0

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._thread_pool = QThreadPool.globalInstance()
        self._refresh_worker: RefreshWorker | None = None
        self._refresh_in_flight = False
        self._handsfree_action_workers: dict[str, HandsfreeActionWorker] = {}
        self._drag_origin = None
        self._ui_state = load_ui_state()
        self._last_sections: tuple[ProviderSection, ...] = ()
        self._provider_buttons: dict[ProviderName, QPushButton] = {}
        self._select_popup: QFrame | None = None
        # Cached tuple of (provider, rows, notes) counts from the last render;
        # changes here gate calls to ``adjustSize`` so routine refreshes don't
        # resize the window just because a percentage moved.
        self._last_layout_key: tuple[tuple[str, int, int], ...] = ()

        if (
            self._ui_state.selected_providers is not None
            and not getattr(config, "providers_explicit", False)
        ):
            self.config.selected_providers = set(self._ui_state.selected_providers)
        if not self.config.selected_providers:
            self.config.selected_providers = set(PROVIDER_ORDER)

        # Capture the "reset target" *before* applying any persisted zoom
        # so ⌘0 always snaps back to the CLI/default size, not whatever
        # the user last zoomed to. Persisted zoom wins over the compiled
        # default but never over an explicit ``--font-size`` flag.
        self._default_font_size = float(config.font_size)
        if (
            self._ui_state.font_size is not None
            and not getattr(config, "font_size_explicit", False)
        ):
            self.config.font_size = float(self._ui_state.font_size)

        self.setWindowTitle("usage-hud-app")
        self.setStyleSheet(build_stylesheet(self.config.font_size))
        # Cap overall width so long notes (e.g. HTTP 429 error bodies) wrap
        # rather than stretching the whole panel. The minimum width comes
        # from the grid's QFontMetrics columns — this ceiling only kicks in
        # when a note is longer than those fixed columns would support.
        self.setMaximumWidth(MAX_WINDOW_WIDTH_PX)
        self._apply_window_flags()
        self._build_layout()
        self._apply_geometry()
        self._bind_shortcuts()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_now)
        self._timer.start(max(1, int(config.interval_seconds * 1000)))
        self.refresh_now()

    def _apply_window_flags(self) -> None:
        flags = Qt.WindowType.Window
        flags |= Qt.WindowType.WindowMinimizeButtonHint
        if self.config.always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        if self.config.frameless:
            flags |= Qt.WindowType.FramelessWindowHint
            # Translucent so the rounded QFrame#root clips cleanly and the
            # rectangular window surface doesn't show through the corners.
            # WA_StyledBackground tells Qt to honor our stylesheet for bg.
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowFlags(flags)

    def _build_layout(self) -> None:
        # Zero-margin outer layout so the inner QFrame fills the entire
        # window rect; the translucent top-level is purely a carrier for
        # the window flags and drag handling.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        self._root_frame = QFrame(self)
        self._root_frame.setObjectName("root")
        self._root_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        outer.addWidget(self._root_frame)

        inner = QVBoxLayout(self._root_frame)
        inner.setContentsMargins(8, 4, 8, 6)
        inner.setSpacing(2)
        inner.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        # Thin top bar: provider visibility + notification/mode/mic controls
        # on the left, window controls on the right. No title/timestamp; the
        # refresh metadata goes into the window tooltip.
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(4)
        for provider in PROVIDER_ORDER:
            button = self._build_provider_button(provider)
            button.clicked.connect(
                lambda _checked=False, p=provider: self._toggle_provider(p)
            )
            self._provider_buttons[provider] = button
            top_bar.addWidget(button)
        self._refresh_provider_buttons()
        top_bar.addSpacing(CONTROL_GROUP_GAP_PX)
        # Global mute toggle — one button flips *all* providers at once.
        # The per-row bells were inconsistent (Gemini has no mute file) and
        # chewed up a dedicated grid column for ~two providers. A single
        # header control matches how the user thinks about notifications:
        # on or off for the whole HUD, not per-engine.
        self.mute_button = self._build_icon_button(
            "bell",
            tooltip="Mute notifications",
            accessible="Toggle notifications",
            checkable=True,
        )
        self.mute_button.clicked.connect(self._toggle_mute_all)
        self._attach_settings_menu(self.mute_button, "notifications")
        top_bar.addWidget(self.mute_button)
        self._refresh_mute_icon()
        self.mode_button = self._build_icon_button(
            "user-sound",
            tooltip="Turn voice mode on",
            accessible="Toggle voice interaction mode",
            checkable=True,
        )
        self.mode_button.clicked.connect(self._toggle_mode)
        self._attach_settings_menu(self.mode_button, "mode")
        top_bar.addWidget(self.mode_button)
        self._refresh_mode_icon()
        self.wake_button = self._build_icon_button(
            "microphone",
            tooltip="Turn mic off",
            accessible="Toggle mic",
            checkable=True,
        )
        self.wake_button.clicked.connect(self._toggle_mic)
        self._attach_settings_menu(self.wake_button, "wake")
        top_bar.addWidget(self.wake_button)
        self._refresh_wake_icon()
        top_bar.addStretch(1)
        self.minimize_button = self._build_icon_button(
            "minus",
            tooltip="Minimize · \u2318M",
            accessible="Minimize window",
        )
        self.minimize_button.clicked.connect(self._minimize_window)
        top_bar.addWidget(self.minimize_button)
        self.close_button = self._build_icon_button(
            "x",
            tooltip="Close · \u2318W",
            accessible="Close window",
        )
        self.close_button.clicked.connect(self.close)
        top_bar.addWidget(self.close_button)
        inner.addLayout(top_bar)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(4)
        self.grid.setVerticalSpacing(2)
        self.grid.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        inner.addLayout(self.grid)
        self._configure_grid_columns()

        self.error_label = QLabel("")
        self.error_label.setObjectName("status")
        self.error_label.hide()
        inner.addWidget(self.error_label)

        self.settings_panel = self._build_settings_panel()
        self.settings_panel.hide()
        inner.addWidget(self.settings_panel)

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame(self._root_frame)
        panel.setObjectName("settingsPanel")
        panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.settings_panel_layout = QVBoxLayout(panel)
        self.settings_panel_layout.setContentsMargins(6, 8, 6, 8)
        self.settings_panel_layout.setSpacing(5)
        self.settings_panel_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self._active_settings_panel: str | None = None
        return panel

    def _attach_settings_menu(self, button: QPushButton, panel_name: str) -> None:
        button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        button.customContextMenuRequested.connect(
            lambda _pos, name=panel_name: self._toggle_settings_panel(name)
        )

    def _settings_row(self, icon_name: str) -> tuple[QHBoxLayout, QVBoxLayout]:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        icon_label = QLabel(self.settings_panel)
        icon_label.setPixmap(
            ui_pixmap(
                icon_name,
                size_px=UI_ICON_PX,
                color=COLORS["fg"],
                dpr=self.devicePixelRatioF(),
            )
        )
        icon_label.setFixedSize(UI_ICON_PX + 10, UI_ICON_PX + 10)
        icon_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter
        )
        row.addWidget(icon_label)

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(3)
        column.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        row.addLayout(column, 1)
        row.setAlignment(icon_label, Qt.AlignmentFlag.AlignTop)
        return row, column

    def _clear_settings_panel(self) -> None:
        _clear_layout(self.settings_panel_layout)

    def _show_settings_panel(self, panel_name: str) -> None:
        self._close_select_popup()
        self.settings_panel.setMaximumHeight(16777215)
        self._clear_settings_panel()
        self._active_settings_panel = panel_name
        if panel_name == "notifications":
            self._build_notifications_panel()
        elif panel_name == "mode":
            self._build_mode_panel()
        elif panel_name == "wake":
            self._build_wake_panel()
        else:
            self._build_mode_panel()
        self.settings_panel.show()
        self._refit_window(force=True)

    def _hide_settings_panel(self) -> None:
        self._close_select_popup()
        self.settings_panel.hide()
        self._active_settings_panel = None
        self._clear_settings_panel()
        self.settings_panel.setMaximumHeight(0)
        self._refit_window(force=True)

    def _toggle_settings_panel(self, panel_name: str | None = None) -> None:
        target = panel_name or self._active_settings_panel or "mode"
        if self.settings_panel.isVisible() and self._active_settings_panel == target:
            self._hide_settings_panel()
            return
        self._show_settings_panel(target)

    def _refit_window(self, *, force: bool = False) -> None:
        if self.config.geometry is not None:
            return
        self.layout().invalidate()
        self._root_frame.layout().invalidate()
        self.layout().activate()
        self._root_frame.layout().activate()
        hint = self.sizeHint()
        width = min(hint.width(), MAX_WINDOW_WIDTH_PX)
        height = hint.height()
        if force:
            self.setMinimumSize(width, height)
            self.setMaximumSize(width, height)
            self.resize(width, height)
            QTimer.singleShot(0, self._clear_forced_window_size)
            return
        self.setMinimumSize(width, height)
        self.resize(width, height)
        QTimer.singleShot(0, self._clear_forced_window_size)

    def _clear_forced_window_size(self) -> None:
        self.setMinimumSize(0, 0)
        self.setMaximumWidth(MAX_WINDOW_WIDTH_PX)
        self.setMaximumHeight(16777215)

    def _build_notifications_panel(self) -> None:
        row, column = self._settings_row("bell")
        options = self._sound_theme_options()
        column.addWidget(
            self._build_select_button(
                self._label_for_value(options, self._current_sound_theme(), "Default"),
                options,
                self._sound_theme_selected,
                setting_label="Sound",
            )
        )

        self.settings_panel_layout.addLayout(row)

    def _build_mode_panel(self) -> None:
        row, column = self._settings_row("user-sound")

        mode = self._current_interaction_mode()
        column.addWidget(
            self._build_select_button(
                self._label_for_value(INTERACTION_MODES, mode, "Off"),
                INTERACTION_MODES,
                self._set_interaction_mode,
                setting_label="Mode",
            )
        )
        column.addWidget(
            self._build_slider(
                "Readout",
                float(self._speech_verbosity_index()),
                self._speech_verbosity_changed,
                minimum=0,
                maximum=len(SPEECH_VERBOSITY_OPTIONS) - 1,
                steps=len(SPEECH_VERBOSITY_OPTIONS) - 1,
                formatter=lambda value: self._speech_verbosity_label_for_index(value),
            )
        )

        provider_options = tuple((label, provider, True) for provider, label in TTS_PROVIDERS)
        provider = self._current_tts_provider()
        column.addWidget(
            self._build_select_button(
                self._label_for_value(
                    provider_options,
                    provider,
                    "Kokoro",
                ),
                provider_options,
                self._tts_provider_selected,
                setting_label="Model",
            )
        )

        if provider == "chatterbox":
            chatterbox_voice = self._current_chatterbox_voice()
            column.addWidget(
                self._build_select_button(
                    self._label_for_value(
                        CHATTERBOX_VOICE_OPTIONS,
                        chatterbox_voice,
                        "p238 clone",
                    ),
                    CHATTERBOX_VOICE_OPTIONS,
                    self._chatterbox_voice_selected,
                    setting_label="Voice",
                )
            )
            chatterbox_style = self._current_chatterbox_style()
            column.addWidget(
                self._build_select_button(
                    self._label_for_value(
                        CHATTERBOX_STYLE_OPTIONS,
                        chatterbox_style,
                        "Auto (context)",
                    ),
                    CHATTERBOX_STYLE_OPTIONS,
                    self._chatterbox_style_selected,
                    setting_label="Emotion",
                )
            )
            column.addWidget(
                self._build_slider(
                    "Load",
                    self._current_chatterbox_style_strength(),
                    self._chatterbox_style_strength_changed,
                    formatter=lambda value: f"{round(value * 100):.0f}%",
                )
            )
            column.addWidget(
                self._build_slider(
                    "Temp",
                    self._current_chatterbox_temperature(),
                    self._chatterbox_temperature_changed,
                    minimum=0.4,
                    maximum=1.2,
                    formatter=lambda value: f"{value:.2f}",
                )
            )
        else:
            voice_spec = self._current_voice_spec()
            selected_voice = (
                self._first_voice_from_spec(voice_spec)
                if self._is_blend_voice_spec(voice_spec)
                else voice_spec
            )
            voice_options = self._voice_options()
            column.addWidget(
                self._build_select_button(
                    self._label_for_value(voice_options, selected_voice, DEFAULT_KOKORO_VOICE),
                    voice_options,
                    self._voice_selected,
                    setting_label="Voice",
                )
            )

            self.voice_blend_label = QLabel("", self.settings_panel)
            self.voice_blend_label.setObjectName("status")
            self.voice_blend_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            if self._is_blend_voice_spec(voice_spec):
                self.voice_blend_label.setText(voice_spec)
                self.voice_blend_label.show()
            else:
                self.voice_blend_label.hide()
            column.addWidget(self.voice_blend_label)

            speed_row = QHBoxLayout()
            speed_row.setContentsMargins(0, 0, 0, 0)
            speed_row.setSpacing(4)
            speed_label = QLabel("Speed:", self.settings_panel)
            speed_label.setObjectName("status")
            speed_row.addWidget(speed_label)
            self.kokoro_speed_spinbox = QDoubleSpinBox(self.settings_panel)
            self.kokoro_speed_spinbox.setObjectName("settingsSpinBox")
            self.kokoro_speed_spinbox.setRange(0.75, 1.6)
            self.kokoro_speed_spinbox.setSingleStep(0.05)
            self.kokoro_speed_spinbox.setDecimals(2)
            self.kokoro_speed_spinbox.setValue(self._current_kokoro_speed())
            self.kokoro_speed_spinbox.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            self.kokoro_speed_spinbox.valueChanged.connect(self._kokoro_speed_changed)
            speed_row.addWidget(self.kokoro_speed_spinbox)
            speed_row.addStretch(1)
            column.addLayout(speed_row)

        if mode == "conductor":
            config = self._read_voice_config()
            model = config.get("conductor_model")
            model_value = (
                model
                if isinstance(model, str) and model.strip()
                else "mlx-community/Qwen3.5-2B-OptiQ-4bit"
            )
            column.addWidget(
                self._build_select_button(
                    self._label_for_value(CONDUCTOR_MODELS, model_value, "Qwen 2B"),
                    CONDUCTOR_MODELS,
                    self._conductor_model_selected,
                    setting_label="Conductor",
                )
            )

        self.settings_panel_layout.addLayout(row)

    def _build_wake_panel(self) -> None:
        row, column = self._settings_row("microphone")
        config = self._read_voice_config()

        wake_options = (
            ("OpenWakeWord", "openwakeword", True),
            ("Whisper", "whisper", True),
        )
        engine = config.get("wake_engine")
        engine_value = engine if isinstance(engine, str) else "openwakeword"
        column.addWidget(
            self._build_select_button(
                self._label_for_value(wake_options, engine_value, "OpenWakeWord"),
                wake_options,
                self._wake_engine_selected,
                setting_label="Wake",
            )
        )
        self.settings_panel_layout.addLayout(row)

    @staticmethod
    def _setting_button_text(setting_label: str | None, value_label: str) -> str:
        if not setting_label:
            return value_label
        return f"{setting_label}: {value_label}"

    def _build_select_button(
        self,
        label: str,
        options: tuple[tuple[str, str | None, bool], ...],
        callback,
        *,
        setting_label: str | None = None,
    ) -> QWidget:
        container = QWidget(self.settings_panel)
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)
        container_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        button = QPushButton(self._setting_button_text(setting_label, label), container)
        button.setObjectName("settingsSelect")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button._select_container_layout = container_layout  # type: ignore[attr-defined]

        def open_select(_checked: bool = False) -> None:
            del _checked
            self._show_select_popup(button, options, callback, setting_label)

        button.clicked.connect(open_select)
        container_layout.addWidget(button)
        return container

    def _build_slider(
        self,
        label: str,
        value: float,
        callback,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        steps: int = 100,
        formatter=None,
    ) -> QWidget:
        container = QWidget(self.settings_panel)
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        formatter = formatter or (lambda slider_value: f"{slider_value:.2f}")
        status = QLabel("", container)
        status.setObjectName("status")

        slider = QSlider(Qt.Orientation.Horizontal, container)
        slider.setObjectName("settingsSlider")
        slider.setRange(0, steps)
        slider.setFixedWidth(170)
        slider.setFocusPolicy(Qt.FocusPolicy.TabFocus)

        def to_slider(raw_value: float) -> int:
            if maximum <= minimum:
                return 0
            clamped = max(minimum, min(maximum, float(raw_value)))
            return round(((clamped - minimum) / (maximum - minimum)) * steps)

        def from_slider(position: int) -> float:
            if maximum <= minimum:
                return minimum
            return minimum + ((float(position) / steps) * (maximum - minimum))

        def refresh(position: int) -> None:
            slider_value = from_slider(position)
            status.setText(f"{label}: {formatter(slider_value)}")
            callback(slider_value)

        slider.setValue(to_slider(value))
        status.setText(f"{label}: {formatter(from_slider(slider.value()))}")
        slider.valueChanged.connect(refresh)

        layout.addWidget(status)
        layout.addWidget(slider)
        return container

    def _show_select_popup(
        self,
        anchor: QPushButton,
        options: tuple[tuple[str, str | None, bool], ...],
        callback,
        setting_label: str | None = None,
    ) -> None:
        if self._select_popup is not None:
            owner = self._select_popup.property("anchor")
            if owner is anchor:
                self._close_select_popup()
                self._refit_window()
                return
        self._close_select_popup()
        popup = QFrame(anchor.parentWidget())
        popup.setObjectName("settingsPopup")
        popup.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        popup.setProperty("anchor", anchor)
        popup.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        popup_layout = QVBoxLayout(popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        popup_layout.setSpacing(0)
        popup_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        scroll = QScrollArea(popup)
        scroll.setObjectName("settingsPopupScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget(scroll)
        content.setObjectName("settingsPopupContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 2, 2)
        content_layout.setSpacing(0)

        def choose(value: str, label: str) -> None:
            anchor.setText(self._setting_button_text(setting_label, label))
            callback(value)
            self._close_select_popup()
            self._refit_window()

        for label, value, enabled in options:
            if value is None or not enabled:
                header = QLabel(label, content)
                header.setObjectName("settingsPopupHeader")
                content_layout.addWidget(header)
                continue
            item = QPushButton(label, content)
            item.setObjectName("settingsPopupItem")
            item.setCursor(Qt.CursorShape.PointingHandCursor)
            item.clicked.connect(
                lambda _checked=False, v=value, text=label: choose(v, text)
            )
            content_layout.addWidget(item)

        scroll.setWidget(content)
        visible_rows = min(10, max(1, len(options)))
        row_height = 28
        popup_width = max(anchor.width(), 180)
        scroll.setFixedWidth(popup_width)
        scroll.setMaximumHeight(visible_rows * row_height + 6)
        popup_layout.addWidget(scroll)

        popup.setFixedWidth(popup_width)
        popup.setFixedHeight(scroll.maximumHeight())
        container_layout = getattr(anchor, "_select_container_layout", None)
        if container_layout is None:
            popup.deleteLater()
            return
        container_layout.addWidget(popup)
        self._select_popup = popup
        popup.show()
        self._refit_window()

    def _close_select_popup(self) -> None:
        if self._select_popup is not None:
            self._select_popup.close()
            self._select_popup.deleteLater()
            self._select_popup = None

    @staticmethod
    def _label_for_value(
        options: tuple[tuple[str, str | None, bool], ...],
        value: str,
        fallback: str,
    ) -> str:
        for label, option_value, _enabled in options:
            if option_value == value:
                return label
        return fallback

    def _sound_theme_options(self) -> tuple[tuple[str, str | None, bool], ...]:
        themes = ["default"]
        try:
            themes.extend(
                path.name
                for path in sorted(SOUND_THEME_ROOT.iterdir(), key=lambda p: p.name.lower())
                if path.is_dir()
            )
        except OSError:
            pass
        current = self._current_sound_theme()
        if current not in themes:
            themes.append(current)
        return tuple(
            ("Default" if theme == "default" else theme, theme, True)
            for theme in themes
        )

    def _voice_options(self) -> tuple[tuple[str, str | None, bool], ...]:
        options: list[tuple[str, str | None, bool]] = []
        for group_name, voices in KOKORO_VOICE_GROUPS:
            options.append((group_name, None, False))
            for voice in voices:
                options.append((voice, voice, True))
        return tuple(options)

    def _current_sound_theme(self) -> str:
        try:
            theme = SOUND_THEME_PATH.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return "default"
        except OSError as exc:
            self._show_error(f"Could not read sound theme: {exc}")
            return "default"
        return theme or "default"

    def _sound_theme_selected(self, theme: str) -> None:
        try:
            if theme == "default":
                SOUND_THEME_PATH.unlink(missing_ok=True)
            else:
                SOUND_THEME_PATH.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = SOUND_THEME_PATH.with_name(f"{SOUND_THEME_PATH.name}.tmp")
                tmp_path.write_text(theme + "\n", encoding="utf-8")
                os.replace(tmp_path, SOUND_THEME_PATH)
        except OSError as exc:
            self._show_error(f"Could not write sound theme: {exc}")

    def _current_tts_provider(self) -> str:
        config = self._read_voice_config()
        provider = config.get("tts_provider")
        valid = {key for key, _label in TTS_PROVIDERS}
        if isinstance(provider, str) and provider in valid:
            return provider
        return DEFAULT_TTS_PROVIDER

    def _tts_provider_selected(self, provider: str) -> None:
        config = self._read_voice_config()
        config["tts_provider"] = provider
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")
            return
        QTimer.singleShot(0, lambda: self._show_settings_panel("mode"))

    def _current_chatterbox_voice(self) -> str:
        config = self._read_voice_config()
        voice = config.get("chatterbox_voice")
        valid = {value for _label, value, enabled in CHATTERBOX_VOICE_OPTIONS if enabled}
        if isinstance(voice, str) and voice in valid:
            return voice
        return DEFAULT_CHATTERBOX_VOICE

    def _chatterbox_voice_selected(self, voice: str) -> None:
        config = self._read_voice_config()
        config["chatterbox_voice"] = voice
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")

    def _current_chatterbox_style(self) -> str:
        config = self._read_voice_config()
        style = config.get("chatterbox_style")
        valid = {value for _label, value, enabled in CHATTERBOX_STYLE_OPTIONS if enabled}
        if isinstance(style, str) and style in valid:
            return style
        return DEFAULT_CHATTERBOX_STYLE

    def _chatterbox_style_selected(self, style: str) -> None:
        config = self._read_voice_config()
        config["chatterbox_style"] = style
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")

    def _current_chatterbox_style_strength(self) -> float:
        config = self._read_voice_config()
        try:
            strength = float(config.get("chatterbox_style_strength", 0.35))
        except (TypeError, ValueError):
            return 0.35
        return max(0.0, min(1.0, strength))

    def _chatterbox_style_strength_changed(self, strength: float) -> None:
        config = self._read_voice_config()
        config["chatterbox_style_strength"] = round(float(strength), 2)
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")

    def _current_chatterbox_temperature(self) -> float:
        config = self._read_voice_config()
        try:
            temperature = float(config.get("chatterbox_temperature", 0.8))
        except (TypeError, ValueError):
            return 0.8
        return max(0.4, min(1.2, temperature))

    def _chatterbox_temperature_changed(self, temperature: float) -> None:
        config = self._read_voice_config()
        config["chatterbox_temperature"] = round(float(temperature), 2)
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")

    def _current_kokoro_speed(self) -> float:
        config = self._read_voice_config()
        try:
            speed = float(config.get("kokoro_speed", 1.1))
        except (TypeError, ValueError):
            return 1.1
        return max(0.75, min(1.6, speed))

    def _kokoro_speed_changed(self, speed: float) -> None:
        config = self._read_voice_config()
        config["kokoro_speed"] = round(float(speed), 2)
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")

    def _wake_engine_selected(self, engine: str) -> None:
        config = self._read_voice_config()
        config["wake_engine"] = engine
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")

    def _current_interaction_mode(self) -> str:
        config = self._read_voice_config()
        mode = config.get("interaction_mode")
        valid = {value for _label, value, enabled in INTERACTION_MODES if enabled}
        if isinstance(mode, str) and mode in valid:
            if mode == "off":
                if self._is_speech_enabled():
                    return "direct"
                if self._is_wake_enabled():
                    return "on_demand"
            return mode
        if mode == "passive":
            return "direct"
        if mode == "hybrid":
            return "conductor"
        return DEFAULT_INTERACTION_MODE

    def _last_active_interaction_mode(self) -> str:
        config = self._read_voice_config()
        mode = config.get("last_interaction_mode")
        valid = {
            value
            for _label, value, enabled in INTERACTION_MODES
            if enabled and value != "off"
        }
        if isinstance(mode, str) and mode in valid:
            return mode
        return DEFAULT_ACTIVE_INTERACTION_MODE

    def _set_interaction_mode(self, mode: str) -> None:
        config = self._read_voice_config()
        config["interaction_mode"] = mode
        if mode != "off":
            config["last_interaction_mode"] = mode
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")
            return
        self._apply_interaction_mode_runtime(mode)
        self._refresh_mode_icon()
        self._refresh_wake_icon()
        if self.settings_panel.isVisible():
            QTimer.singleShot(0, lambda: self._show_settings_panel("mode"))

    def _current_speech_verbosity(self) -> str:
        config = self._read_voice_config()
        verbosity = config.get("verbosity")
        valid = {value for _label, value in SPEECH_VERBOSITY_OPTIONS}
        if isinstance(verbosity, str) and verbosity in valid:
            return verbosity
        return DEFAULT_SPEECH_VERBOSITY

    def _speech_verbosity_index(self) -> int:
        current = self._current_speech_verbosity()
        for index, (_label, value) in enumerate(SPEECH_VERBOSITY_OPTIONS):
            if value == current:
                return index
        return 1

    @staticmethod
    def _speech_verbosity_label_for_index(value: float) -> str:
        index = max(0, min(len(SPEECH_VERBOSITY_OPTIONS) - 1, round(float(value))))
        return SPEECH_VERBOSITY_OPTIONS[index][0]

    def _speech_verbosity_changed(self, value: float) -> None:
        index = max(0, min(len(SPEECH_VERBOSITY_OPTIONS) - 1, round(float(value))))
        config = self._read_voice_config()
        config["verbosity"] = SPEECH_VERBOSITY_OPTIONS[index][1]
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")

    def _conductor_model_selected(self, model: str) -> None:
        config = self._read_voice_config()
        config["conductor_model"] = model
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")

    def _current_voice_spec(self) -> str:
        config = self._read_voice_config()
        voice = config.get("kokoro_voice")
        if isinstance(voice, str) and voice.strip():
            return voice.strip()
        return DEFAULT_KOKORO_VOICE

    @staticmethod
    def _is_blend_voice_spec(voice_spec: str) -> bool:
        return ":" in voice_spec or "," in voice_spec

    @staticmethod
    def _first_voice_from_spec(voice_spec: str) -> str:
        first_part = voice_spec.split(",", 1)[0].strip()
        voice_name = first_part.split(":", 1)[0].strip()
        return voice_name or DEFAULT_KOKORO_VOICE

    def _read_voice_config(self) -> dict[str, object]:
        try:
            raw_config = VOICE_CONFIG_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            self._show_error(f"Could not read voice config: {exc}")
            return {}

        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            self._show_error(f"Could not parse voice config: {exc}")
            return {}
        if isinstance(config, dict):
            return config
        self._show_error("Voice config must be a JSON object.")
        return {}

    def _write_voice_config(self, config: dict[str, object]) -> None:
        VOICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = VOICE_CONFIG_PATH.with_name(f"{VOICE_CONFIG_PATH.name}.tmp")
        payload = json.dumps(config, indent=2) + "\n"
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, VOICE_CONFIG_PATH)

    def _voice_selected(self, voice: str) -> None:
        config = self._read_voice_config()
        config["kokoro_voice"] = voice
        try:
            self._write_voice_config(config)
        except OSError as exc:
            self._show_error(f"Could not write voice config: {exc}")
        if hasattr(self, "voice_blend_label"):
            self.voice_blend_label.hide()

    def _configure_grid_columns(self) -> None:
        """Pin column widths using QFontMetrics so refreshes don't reflow.

        The metric column reserves exactly ``"W 100%"`` (or the decimal
        equivalent) so every row's combined label+value blob has the
        same cell width, which keeps the pace bar anchored to a fixed
        x across refreshes. ``COL_DETAIL`` is left unpinned — Qt sizes
        it to the widest real delta/target cell per refresh so short
        deltas (``+2 (50%)``) don't leave dead space on the right.
        """

        fm = QFontMetrics(self.font())
        # Widest combined "label + space + value" blob for the current
        # decimals mode. This is the *only* metric-cell probe we need
        # now that label and value share a single QLabel.
        metric_probe = "W 100.0%" if usage_hud.DECIMALS else "W 100%"
        self.grid.setColumnMinimumWidth(COL_LOGO, PROVIDER_LOGO_PX + 2)
        self.grid.setColumnMinimumWidth(COL_METRIC, fm.horizontalAdvance(metric_probe) + 2)
        self.grid.setColumnMinimumWidth(COL_BAR, PaceBarWidget.WIDGET_WIDTH_PX + 2)
        self.grid.setColumnStretch(COL_DETAIL, 0)

    def _apply_geometry(self) -> None:
        if self.config.geometry is None:
            # Auto-fit mode: leave the initial size to Qt's sizeHint (the
            # grid has fixed column widths from QFontMetrics, so the first
            # adjustSize after the first refresh produces the right width).
            x, y = self._ui_state.window_position
            self.move(x, y)
            return
        width, height, x, y = _parse_geometry(self.config.geometry)
        if not self.config.geometry_explicit:
            x, y = self._ui_state.window_position
        self.resize(width, max(60, height))
        self.move(x, y)

    def _bind_shortcuts(self) -> None:
        # On macOS, Qt maps the string ``Ctrl+X`` to ⌘+X (the physical
        # Command key), and ``Meta+X`` to the *physical* Control key —
        # opposite of what the tooltips claim. Use ``Ctrl+...`` so ⌘W
        # and friends actually fire.
        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+W"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+M"), self, activated=self._minimize_window)
        QShortcut(QKeySequence("R"), self, activated=self.refresh_now)
        QShortcut(QKeySequence(","), self, activated=self._toggle_settings_panel)
        # Zoom — ⌘= / ⌘- / ⌘0 with the standard-key aliases so ⌘+ also
        # works without having to hold Shift. Bound individually so each
        # fires its own slot rather than going through a single shortcut
        # that would ambiguously match all three.
        QShortcut(
            QKeySequence.StandardKey.ZoomIn, self, activated=self._zoom_in
        )
        QShortcut(QKeySequence("Ctrl+="), self, activated=self._zoom_in)
        QShortcut(
            QKeySequence.StandardKey.ZoomOut, self, activated=self._zoom_out
        )
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self._zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._zoom_reset)

    # --- zoom ------------------------------------------------------------

    def _zoom_in(self) -> None:
        self._apply_font_size(self.config.font_size + self.ZOOM_STEP)

    def _zoom_out(self) -> None:
        self._apply_font_size(self.config.font_size - self.ZOOM_STEP)

    def _zoom_reset(self) -> None:
        self._apply_font_size(self._default_font_size, clear_persisted=True)

    def _apply_font_size(
        self, new_size: float, *, clear_persisted: bool = False
    ) -> None:
        """Rebuild the stylesheet at ``new_size`` and refit the grid.

        Has to do three things in order: (1) swap the stylesheet so every
        QLabel/QPushButton picks up the new font, (2) re-run the column
        configuration since the column minimums were derived from the
        *old* QFontMetrics and will otherwise leave the grid too wide or
        too narrow, and (3) re-render the current sections so note
        budgets and elide points are recomputed against the new metrics.
        The final ``adjustSize`` reshrinks the window to the new hint so
        zooming out actually reclaims pixels instead of leaving trailing
        whitespace.
        """

        clamped = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, float(new_size)))
        if abs(clamped - self.config.font_size) < 0.01 and not clear_persisted:
            return
        self.config.font_size = clamped
        self.setStyleSheet(build_stylesheet(clamped))
        self._configure_grid_columns()
        if self._last_sections:
            self._render_grid(self._last_sections)
        # Force a refit — the layout key hasn't changed (same provider
        # shapes) so ``_apply_refresh_result`` wouldn't adjustSize on its
        # own, but the cell widths definitely did.
        if self.config.geometry is None:
            self._refit_window()
        # Persist so the next launch remembers the zoom level. ``⌘0``
        # clears the persisted value so the CLI default takes over again.
        self._ui_state.font_size = None if clear_persisted else clamped
        self._save_current_state()

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        if self.settings_panel.isVisible():
            self._hide_settings_panel()
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self.config.frameless and event.button() == Qt.MouseButton.LeftButton:
            # Prefer startSystemMove — it lets the window manager handle the
            # drag (snap zones, multi-monitor) instead of us doing math.
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
            self._drag_origin = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if (
            self.config.frameless
            and self._drag_origin is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_origin is not None:
            self._save_current_state()
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_current_state()
        super().closeEvent(event)

    def _minimize_window(self) -> None:
        self._save_current_state()
        self.showMinimized()

    def refresh_now(self) -> None:
        if self._refresh_in_flight:
            return
        self._refresh_in_flight = True
        worker = RefreshWorker(self.config)
        worker.signals.succeeded.connect(self._apply_refresh_result)
        worker.signals.failed.connect(self._apply_refresh_error)
        self._refresh_worker = worker
        self._thread_pool.start(worker)

    def _apply_refresh_result(
        self, bundle: SnapshotBundle, sections: tuple[ProviderSection, ...]
    ) -> None:
        self._refresh_in_flight = False
        self._refresh_worker = None
        sections = tuple(
            section
            for section in sections
            if section.provider in self.config.selected_providers
        )
        self.error_label.hide()
        self._last_sections = sections
        self._render_grid(sections)
        self._apply_refresh_tooltip(bundle)

        # Only refit the window when auto-fit is on AND the grid's row
        # layout actually changed. Most refreshes just move percentages
        # around, which shouldn't cause the window to twitch. The first
        # refresh (empty → populated) always passes this check and
        # establishes the initial auto-fit size. When the user passed
        # --geometry, we honor it and never resize.
        layout_key = self._layout_key(sections)
        if layout_key != self._last_layout_key:
            self._last_layout_key = layout_key
            if self.config.geometry is None:
                self.adjustSize()

    @staticmethod
    def _layout_key(
        sections: tuple[ProviderSection, ...],
    ) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (section.provider, len(section.rows), len(section.notes))
            for section in sections
        )

    def _apply_refresh_tooltip(self, bundle: SnapshotBundle) -> None:
        try:
            timestamp = (
                datetime.fromisoformat(bundle.generated_at)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M:%S %Z")
            )
        except ValueError:
            timestamp = bundle.generated_at
        self.setToolTip(
            f"Updated {timestamp} · refresh every {self.config.interval_seconds:.0f}s\n"
            f"⌘+ / ⌘- / ⌘0 to zoom · {self.config.font_size:.0f}pt\n"
            "Left buttons toggle providers, notifications, voice mode, and mic"
        )

    def _apply_refresh_error(self, message: str) -> None:
        self._refresh_in_flight = False
        self._refresh_worker = None
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setStyleSheet(f"color: {COLORS['yellow']};")
        self.error_label.show()

    # --- grid rendering --------------------------------------------------

    # Pixels of vertical breathing room inserted between provider sections
    # so Claude / Codex / Gemini read as distinct groups at a glance.
    SECTION_GAP_PX = 6

    def _render_grid(self, sections: tuple[ProviderSection, ...]) -> None:
        _clear_layout(self.grid)
        row = 0
        for idx, section in enumerate(sections):
            # Insert a thin spacer row between providers so the sections
            # are visually distinct without inflating the per-row spacing.
            if idx > 0:
                spacer = QWidget()
                spacer.setFixedHeight(self.SECTION_GAP_PX)
                self.grid.addWidget(spacer, row, 0, 1, COL_DETAIL + 1)
                row += 1

            start_row = row
            row_count = max(1, len(section.rows) + len(section.notes))

            logo = QLabel()
            logo.setPixmap(
                provider_pixmap(
                    section.provider,
                    size_px=PROVIDER_LOGO_PX,
                    mode="color",
                    dpr=self.devicePixelRatioF(),
                )
            )
            logo.setFixedSize(PROVIDER_LOGO_PX, PROVIDER_LOGO_PX)
            logo.setAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter
            )
            logo.setToolTip(section.title)
            # Vert-center the logo inside its rowspan so it sits mid-section
            # instead of top-anchoring and creating ragged bell alignment
            # across providers with different row counts.
            self.grid.addWidget(
                logo,
                start_row,
                COL_LOGO,
                row_count,
                1,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter,
            )

            for metric_row in section.rows:
                self._add_metric_row(row, metric_row)
                row += 1
            for note in section.notes:
                self._add_note_row(row, note)
                row += 1

    def _build_metric_html(self, metric_row: MetricRow) -> str:
        """Render ``LABEL SPACE RIGHT_PADDED_VALUE`` as colored rich text.

        Label and value share a single QLabel so the gap between them is
        always exactly one space character — no trailing empty cell on
        the label side, no leading empty cell on the value side. The
        value segment is padded with non-breaking spaces to a fixed
        character count (``100%`` / ``100.0%``) so the trailing ``%`` of
        every row lines up vertically even though the column itself is
        left-aligned.

        Colors are applied per segment via inline ``<span>`` styles so
        label color (bold vs dim vs prefix accent) stays decoupled from
        value color (utilization-level tint).
        """

        # Label text (with optional prefix). ``html.escape`` keeps any
        # future ``<`` / ``&`` safe in rich-text rendering.
        if metric_row.prefix:
            label_text = f"{metric_row.prefix} {metric_row.label}"
            label_color = color_for_style(metric_row.prefix_style)
        else:
            label_text = metric_row.label
            label_color = (
                color_for_style("dim")
                if metric_row.display_mode == "value_only"
                else COLORS["fg"]
            )

        # Value text, right-padded with non-breaking spaces to the
        # widest possible width in the current decimals mode. ``&nbsp;``
        # is one monospace char wide in Menlo, so the padding holds
        # cross-row alignment without trailing-space collapsing.
        # ``fmt_pct`` uses ``{:>3}`` in non-decimals mode, which yields
        # a *wider* string for a float (``'42.0%'``) than for an int
        # (``' 42%'``) — coerce to ``int`` so ``value_max_chars`` of 4
        # always bounds the stripped output.
        if metric_row.utilization is None:
            value_text = "--"
        else:
            util_value = (
                float(metric_row.utilization)
                if usage_hud.DECIMALS
                else int(round(float(metric_row.utilization)))
            )
            value_text = usage_hud.fmt_pct(util_value).strip()
        value_color = COLORS["fg"]
        value_max_chars = 6 if usage_hud.DECIMALS else 4  # "100.0%" or "100%"
        pad = max(0, value_max_chars - len(value_text))
        value_padded = "&nbsp;" * pad + html.escape(value_text)

        return (
            f'<span style="color:{label_color}">{html.escape(label_text)}</span>'
            "&nbsp;"
            f'<span style="color:{value_color}">{value_padded}</span>'
        )

    def _add_metric_row(self, row: int, metric_row: MetricRow) -> None:
        metric_widget = QLabel()
        metric_widget.setTextFormat(Qt.TextFormat.RichText)
        metric_widget.setText(self._build_metric_html(metric_row))
        # Left-align inside the cell: the value segment is already
        # right-padded to a fixed character width so the ``%`` of every
        # row lines up without needing grid-level right-alignment.
        self.grid.addWidget(
            metric_widget,
            row,
            COL_METRIC,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        has_pace_bar = (
            metric_row.display_mode == "pace"
            and metric_row.utilization is not None
            and metric_row.expected_utilization is not None
        )
        if has_pace_bar:
            bar = PaceBarWidget(width_cells=PACE_BAR_CELLS)
            bar.set_pace(
                metric_row.utilization,
                metric_row.expected_utilization,
                stale=metric_row.stale,
            )
            self.grid.addWidget(
                bar,
                row,
                COL_BAR,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            detail = QLabel()
            detail.setTextFormat(Qt.TextFormat.RichText)
            detail.setText(_format_detail_html(metric_row))
            detail.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.grid.addWidget(
                detail,
                row,
                COL_DETAIL,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
        else:
            # Value-only rows (e.g. "Reset  12:34") don't get a pace bar —
            # fold the reset-at text into a span across the bar + detail
            # cells so the logo-anchored column alignment stays intact.
            text = str(metric_row.reset_at) if metric_row.reset_at is not None else ""
            aux = QLabel(text)
            aux.setStyleSheet(f"color: {color_for_style('dim')};")
            self.grid.addWidget(
                aux,
                row,
                COL_BAR,
                1,
                COL_DETAIL - COL_BAR + 1,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )

    def _add_note_row(self, row: int, note: NoteLine) -> None:
        # Notes are always single-line — long error bodies get elided with
        # an ellipsis and the full text lives in the tooltip. Wrapping
        # breaks the tight row rhythm (variable section heights = the
        # "disjointed" look) and also lets QLabel's heightForWidth interact
        # weirdly with grid spans, collapsing the label to ~50px wide.
        label = QLabel()
        label.setStyleSheet(f"color: {color_for_style(note.style)};")
        label.setWordWrap(False)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setToolTip(note.text)
        # Match the note row height to a metric row height (driven by the
        # pace widget) so a note-only section like Claude's "HTTP 429..."
        # has the same visual weight as a Codex S/W/M row. Without this,
        # the logo rowspan centers on a ~16px text-height row while the
        # logo is 22px, making the section look "disconnected".
        label.setMinimumHeight(PaceBarWidget.WIDGET_HEIGHT_PX)
        fm = QFontMetrics(self.font())
        # Elide budget = a realistic ceiling on the spanned cells. The
        # span covers metric+bar+detail (3 columns, so 2 internal gaps).
        # The metric probe tracks ``_configure_grid_columns`` exactly so
        # a note can never force the grid wider than a metric row; the
        # ``-100 (100%)`` detail probe is a soft upper bound since
        # COL_DETAIL has no pinned minimum and actual detail widths vary.
        metric_probe = "W 100.0%" if usage_hud.DECIMALS else "W 100%"
        note_budget_px = (
            fm.horizontalAdvance(metric_probe)
            + 2
            + PaceBarWidget.WIDGET_WIDTH_PX
            + 2
            + fm.horizontalAdvance("-100 (100%)")
            + 2 * self.grid.horizontalSpacing()
        )
        label.setText(
            fm.elidedText(note.text, Qt.TextElideMode.ElideRight, note_budget_px)
        )
        label.setMaximumWidth(note_budget_px)
        # Span from the metric cell through detail so notes read as an
        # inset comment under the provider's metric rows.
        self.grid.addWidget(
            label,
            row,
            COL_METRIC,
            1,
            COL_DETAIL - COL_METRIC + 1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

    # --- controls --------------------------------------------------------

    def _build_provider_button(self, provider: ProviderName) -> QPushButton:
        button = QPushButton()
        button.setObjectName("providerToggleButton")
        button.setFlat(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        button.setCheckable(True)
        button.setAccessibleName(f"Toggle {PROVIDER_LABELS[provider]} usage")
        button.setFixedSize(UI_ICON_PX + 10, UI_ICON_PX + 10)
        return button

    def _build_icon_button(
        self,
        icon_name: str,
        *,
        tooltip: str,
        accessible: str,
        checkable: bool = False,
    ) -> QPushButton:
        button = QPushButton()
        button.setObjectName("controlButton")
        button.setIcon(
            ui_icon(
                icon_name,
                size_px=UI_ICON_PX,
                color=COLORS["fg"],
                dpr=self.devicePixelRatioF(),
            )
        )
        button.setFlat(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        button.setToolTip(tooltip)
        button.setAccessibleName(accessible)
        button.setCheckable(checkable)
        button.setFixedSize(UI_ICON_PX + 10, UI_ICON_PX + 10)
        return button

    def _set_control_icon(
        self,
        button: QPushButton,
        icon_name: str,
        *,
        color: str | None = None,
    ) -> None:
        button.setIcon(
            ui_icon(
                icon_name,
                size_px=UI_ICON_PX,
                color=color or COLORS["fg"],
                dpr=self.devicePixelRatioF(),
            )
        )

    def _set_handsfree_loading(self, button: QPushButton, tooltip: str) -> None:
        self._set_control_icon(button, "spinner", color=COLORS["yellow"])
        button.setChecked(True)
        button.setEnabled(False)
        button.setToolTip(tooltip)

    def _start_handsfree_action(
        self,
        action: str,
        broker_args: list[str],
        *,
        button: QPushButton,
        loading_tooltip: str,
        timeout: float = 120.0,
    ) -> None:
        if action in self._handsfree_action_workers:
            return
        self._set_handsfree_loading(button, loading_tooltip)
        worker = HandsfreeActionWorker(action, broker_args, timeout=timeout)
        worker.signals.succeeded.connect(self._handsfree_action_succeeded)
        worker.signals.failed.connect(self._handsfree_action_failed)
        self._handsfree_action_workers[action] = worker
        self._thread_pool.start(worker)

    def _refresh_handsfree_button_for_action(self, action: str) -> None:
        if "speech" in action or "conductor" in action:
            self.mode_button.setEnabled(True)
            self._refresh_mode_icon()
        if "wake" in action:
            self.wake_button.setEnabled(True)
            self._refresh_wake_icon()

    def _handsfree_action_succeeded(self, action: str, _stdout: object) -> None:
        self._handsfree_action_workers.pop(action, None)
        self._refresh_handsfree_button_for_action(action)

    def _handsfree_action_failed(self, action: str, message: str) -> None:
        self._handsfree_action_workers.pop(action, None)
        self._refresh_handsfree_button_for_action(action)
        first_line = message.strip().splitlines()[0] if message.strip() else "unknown error"
        self._show_error(f"Handsfree {action} failed — {first_line[:180]}")

    def _toggle_provider(self, provider: ProviderName) -> None:
        selected = set(self.config.selected_providers)
        if provider in selected:
            if len(selected) == 1:
                self._show_error("Keep at least one provider visible.")
                self._refresh_provider_buttons()
                return
            selected.remove(provider)
        else:
            selected.add(provider)

        self.config.selected_providers = selected
        self._ui_state.selected_providers = set(selected)
        self._save_current_state()
        self._refresh_provider_buttons()
        if self._last_sections:
            visible = tuple(
                section
                for section in self._last_sections
                if section.provider in self.config.selected_providers
            )
            self._render_grid(visible)
            self._last_layout_key = ()
            if self.config.geometry is None:
                self.adjustSize()
        self.refresh_now()

    def _refresh_provider_buttons(self) -> None:
        for provider, button in self._provider_buttons.items():
            checked = provider in self.config.selected_providers
            label = PROVIDER_LABELS[provider]
            button.setChecked(checked)
            button.setToolTip(f"Hide {label}" if checked else f"Show {label}")
            mode = "color" if checked else "mono"
            button.setIcon(
                provider_icon(
                    provider,
                    size_px=UI_ICON_PX,
                    mode=mode,
                    mono_color=COLORS["muted"],
                    dpr=self.devicePixelRatioF(),
                )
            )

    def _is_any_muted(self) -> bool:
        """True when at least one provider mute flag is present on disk."""

        return any(path.exists() for path in MUTE_PATHS.values())

    def _toggle_mute_all(self) -> None:
        """Flip every provider mute file together.

        "Muted" is an all-or-nothing global state from the UI's point of
        view: if anything is currently muted the click clears them all,
        otherwise we create mute files for every provider that supports
        one. Gemini isn't in :data:`MUTE_PATHS` because it has no mute
        hook — that's intentional, not a bug.
        """

        target_muted = not self._is_any_muted()
        failures: list[str] = []
        for provider, path in MUTE_PATHS.items():
            try:
                if target_muted:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("")
                elif path.exists():
                    path.unlink()
            except OSError as exc:
                failures.append(f"{provider}: {exc}")
        if not target_muted and not failures:
            try:
                _mark_consume_after()
            except OSError as exc:
                failures.append(f"consume-after: {exc}")

        self._refresh_mute_icon()
        if failures:
            self._show_error("Could not update mute flag — " + "; ".join(failures))

    def _refresh_mute_icon(self) -> None:
        """Sync the global mute button's icon and tooltip to disk state."""

        muted = self._is_any_muted()
        icon_name = "bell-slash" if muted else "bell"
        self._set_control_icon(
            self.mute_button,
            icon_name,
            color=COLORS["muted"] if muted else COLORS["fg"],
        )
        self.mute_button.setChecked(muted)
        self.mute_button.setToolTip(
            "Unmute all notifications" if muted else "Mute all notifications"
        )

    def _is_speech_enabled(self) -> bool:
        return SPEECH_TOGGLE_PATH.exists() or LEGACY_SPEECH_TOGGLE_PATH.exists()

    def _enable_speech_if_needed(self) -> None:
        if not self._is_speech_enabled():
            self._start_handsfree_action(
                "enable-speech",
                ["enable", "speech", "--timeout", "120"],
                button=self.mode_button,
                loading_tooltip="Starting speech...",
                timeout=130.0,
            )

    def _disable_speech_if_needed(self) -> None:
        if self._is_speech_enabled():
            self._start_handsfree_action(
                "disable-speech",
                ["disable", "speech"],
                button=self.mode_button,
                loading_tooltip="Stopping speech...",
                timeout=15.0,
            )

    def _enable_wake_if_needed(self) -> None:
        if not self._is_wake_enabled():
            self._start_handsfree_action(
                "enable-wake",
                ["enable", "wake", "--timeout", "120"],
                button=self.wake_button,
                loading_tooltip="Starting mic...",
                timeout=130.0,
            )

    def _disable_wake_if_needed(self, *, button: QPushButton | None = None) -> None:
        if self._is_wake_enabled():
            self._start_handsfree_action(
                "disable-wake",
                ["disable", "wake"],
                button=button or self.wake_button,
                loading_tooltip="Stopping mic...",
                timeout=15.0,
            )

    def _warm_conductor_if_needed(self) -> None:
        self._start_handsfree_action(
            "warm-conductor",
            ["warm", "conductor", "--timeout", "120"],
            button=self.mode_button,
            loading_tooltip="Starting conductor...",
            timeout=130.0,
        )

    def _apply_interaction_mode_runtime(self, mode: str) -> None:
        if mode == "off":
            self._disable_speech_if_needed()
            self._disable_wake_if_needed(button=self.mode_button)
            return
        if mode == "on_demand":
            self._disable_speech_if_needed()
            self._enable_wake_if_needed()
            return
        if mode == "direct":
            self._enable_speech_if_needed()
            self._disable_wake_if_needed(button=self.mode_button)
            return
        if mode == "conductor":
            self._enable_speech_if_needed()
            self._enable_wake_if_needed()
            self._warm_conductor_if_needed()

    def _is_wake_enabled(self) -> bool:
        return WAKE_TOGGLE_PATH.exists()

    def _toggle_mic(self) -> None:
        if self._current_interaction_mode() == "off":
            self._set_interaction_mode("on_demand")
            return
        if self._is_wake_enabled():
            self._disable_wake_if_needed()
        else:
            self._enable_wake_if_needed()

    def _refresh_wake_icon(self) -> None:
        enabled = self._is_wake_enabled()
        icon_name = "microphone" if enabled else "microphone-slash"
        self._set_control_icon(
            self.wake_button,
            icon_name,
            color=COLORS["fg"] if enabled else COLORS["muted"],
        )
        self.wake_button.setChecked(enabled)
        self.wake_button.setToolTip(
            "Turn mic off" if enabled else "Turn mic on"
        )

    def _toggle_mode(self) -> None:
        mode = self._current_interaction_mode()
        next_mode = self._last_active_interaction_mode() if mode == "off" else "off"
        self._set_interaction_mode(next_mode)

    def _refresh_mode_icon(self) -> None:
        mode = self._current_interaction_mode()
        enabled = mode != "off"
        labels = {
            "off": "Off",
            "on_demand": "On Demand",
            "direct": "Direct",
            "conductor": "Conductor",
        }
        self._set_control_icon(
            self.mode_button,
            "user-sound",
            color=COLORS["fg"] if enabled else COLORS["muted"],
        )
        self.mode_button.setChecked(enabled)
        self.mode_button.setToolTip(
            f"Mode: {labels.get(mode, mode)}. "
            + ("Click for off" if enabled else "Click for last mode")
        )

    def _save_current_state(self) -> None:
        position = self.pos()
        self._ui_state.window_position = (position.x(), position.y())
        save_ui_state(self._ui_state)
