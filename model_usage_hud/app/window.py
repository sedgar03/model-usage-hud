"""Minimal PySide6 desktop window for the usage HUD.

The window is a dense ``QGridLayout`` where column 0 is a provider logo
spanning that provider's rows, and the remaining columns are label, value,
pace bar, and detail text. Drops the old card-per-provider chrome and the
header title / timestamp row in favor of a tooltip on the whole frame.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import usage_hud
from model_usage_hud.app.icons import provider_icon, provider_pixmap, ui_icon
from model_usage_hud.app.styles import COLORS, build_stylesheet, color_for_style
from model_usage_hud.app.widgets.gauge_bar import GaugeBarWidget
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
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


GEOMETRY_RE = re.compile(r"^(?P<w>\d+)x(?P<h>\d+)\+(?P<x>-?\d+)\+(?P<y>-?\d+)$")
MUTE_PATHS: dict[ProviderName, Path] = {
    "claude": Path.home() / ".claude" / "mute",
    "codex": Path.home() / ".codex" / "mute",
}
PROVIDER_ORDER: tuple[ProviderName, ...] = ("claude", "codex", "gemini", "system")
PROVIDER_LABELS: dict[ProviderName, str] = {
    "claude": "Claude",
    "codex": "Codex",
    "gemini": "Gemini",
    "system": "System",
}

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
    # Filesystem the System provider gauges for free space (see --disk-path).
    disk_path: str = "/"
    # (base_url, label) to read the System provider from a remote tailnet peer,
    # or None to read the local machine.
    system_remote: tuple[str, str] | None = None


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
                disk_path=self.config.disk_path,
                system_remote=self.config.system_remote,
            )
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(str(exc))
            return
        self.signals.succeeded.emit(bundle, sections)


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
        # A refresh requested while one is already running. The provider fetch
        # can be slow (the Claude usage API in particular), and toggling a
        # provider on mid-fetch must not be silently dropped — otherwise the
        # newly-shown section never appears. We remember the request and fire
        # it as soon as the in-flight refresh returns.
        self._refresh_pending = False
        self._drag_origin = None
        self._ui_state = load_ui_state()
        self._last_sections: tuple[ProviderSection, ...] = ()
        # Most recent section per provider, kept even while a provider is
        # hidden. Toggling a provider back on renders its cached section
        # instantly instead of waiting for the next (possibly slow) fetch.
        self._section_cache: dict[ProviderName, ProviderSection] = {}
        self._provider_buttons: dict[ProviderName, QPushButton] = {}
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

        # Thin top bar: provider visibility + notification mute on the left,
        # window controls on the right. No title/timestamp; the refresh
        # metadata goes into the window tooltip.
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
        top_bar.addWidget(self.mute_button)
        self._refresh_mute_icon()
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
            # Don't drop the request — the current fetch was started under an
            # older provider selection. Queue a follow-up so the latest
            # selection is fetched as soon as the in-flight one completes.
            self._refresh_pending = True
            return
        self._refresh_in_flight = True
        self._refresh_pending = False
        worker = RefreshWorker(self.config)
        worker.signals.succeeded.connect(self._apply_refresh_result)
        worker.signals.failed.connect(self._apply_refresh_error)
        self._refresh_worker = worker
        self._thread_pool.start(worker)

    def _drain_pending_refresh(self) -> None:
        """Kick a queued refresh after the current one settles."""

        if self._refresh_pending and not self._refresh_in_flight:
            self._refresh_pending = False
            QTimer.singleShot(0, self.refresh_now)

    def _apply_refresh_result(
        self, bundle: SnapshotBundle, sections: tuple[ProviderSection, ...]
    ) -> None:
        self._refresh_in_flight = False
        self._refresh_worker = None
        # Cache every returned section (before filtering) so a provider that
        # is later toggled off and back on can repaint from cache immediately.
        for section in sections:
            self._section_cache[section.provider] = section
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

        self._drain_pending_refresh()

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
            "Left buttons toggle providers and notifications"
        )

    def _apply_refresh_error(self, message: str) -> None:
        self._refresh_in_flight = False
        self._refresh_worker = None
        self._show_error(message)
        self._drain_pending_refresh()

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
        # A gauge row is a pace-mode row with a value but no expected target —
        # an instantaneous level (CPU/mem/disk), not a pace. It gets a fill
        # bar plus a free-form detail string (e.g. "263G free").
        has_gauge = (
            metric_row.display_mode == "pace"
            and metric_row.utilization is not None
            and metric_row.expected_utilization is None
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
        elif has_gauge:
            gauge = GaugeBarWidget()
            gauge.set_value(
                metric_row.utilization,
                stale=metric_row.stale,
                style=metric_row.gauge_style,
            )
            self.grid.addWidget(
                gauge,
                row,
                COL_BAR,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            if metric_row.detail:
                detail = QLabel(metric_row.detail)
                detail.setStyleSheet(
                    f"color: {color_for_style(metric_row.detail_style)};"
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
        # Repaint immediately from the per-provider cache (in canonical order)
        # so a provider toggled back on reappears at once instead of waiting
        # for the next fetch. Providers never fetched yet simply have no cached
        # section and fill in when the refresh below completes.
        visible = tuple(
            self._section_cache[provider]
            for provider in PROVIDER_ORDER
            if provider in self.config.selected_providers
            and provider in self._section_cache
        )
        self._last_sections = visible
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

    def _save_current_state(self) -> None:
        position = self.pos()
        self._ui_state.window_position = (position.x(), position.y())
        save_ui_state(self._ui_state)
