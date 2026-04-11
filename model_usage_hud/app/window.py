"""Minimal PySide6 desktop window for the usage HUD.

The window is a dense ``QGridLayout`` where column 0 is a provider logo
spanning that provider's rows, and the remaining columns are label, value,
pace bar, and detail text. Drops the old card-per-provider chrome and the
header title / timestamp row in favor of a tooltip on the whole frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import usage_hud
from model_usage_hud.app.icons import provider_pixmap, ui_icon
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
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


GEOMETRY_RE = re.compile(r"^(?P<w>\d+)x(?P<h>\d+)\+(?P<x>-?\d+)\+(?P<y>-?\d+)$")
MUTE_PATHS: dict[ProviderName, Path] = {
    "claude": Path.home() / ".claude" / "mute",
    "codex": Path.home() / ".codex" / "mute",
}

# Grid column indices — one source of truth for layout math.
COL_LOGO = 0
COL_LABEL = 1
COL_VALUE = 2
COL_BAR = 3
COL_DETAIL = 4
COL_MUTE = 5

# Visual tuning.
PROVIDER_LOGO_PX = 28
UI_ICON_PX = 14
PACE_BAR_CELLS = 16


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


def _format_detail(row: MetricRow) -> str:
    """Pack delta/target/speedometer into a single compact detail cell.

    Returns an empty string when the row has no delta or target to display
    (e.g. value-only rows or rows missing expected utilization).
    """

    pieces: list[str] = []
    if row.delta is not None and row.expected_utilization is not None:
        delta_value = row.delta if row.delta is not None else 0
        pieces.append(usage_hud.fmt_delta(delta_value).strip())
        pieces.append(f"({usage_hud.fmt_pct(row.expected_utilization).strip()})")
    if usage_hud.SPEEDOMETER_ENABLED and row.burn_rate_per_hour is not None:
        eta_text = ""
        if row.eta_hours is not None:
            eta_text = f" {usage_hud._format_eta(row.eta_hours)}"
        pieces.append(f"\u23F1 +{row.burn_rate_per_hour:.0f}%/h{eta_text}")
    return " ".join(pieces)


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

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._thread_pool = QThreadPool.globalInstance()
        self._refresh_in_flight = False
        self._drag_origin = None
        self._ui_state = load_ui_state()
        self._last_sections: tuple[ProviderSection, ...] = ()
        # Cached tuple of (provider, rows, notes) counts from the last render;
        # changes here gate calls to ``adjustSize`` so routine refreshes don't
        # resize the window just because a percentage moved.
        self._last_layout_key: tuple[tuple[str, int, int], ...] = ()

        self.setWindowTitle("usage-hud-app")
        self.setStyleSheet(build_stylesheet(config.font_size))
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

        self._root_frame = QFrame(self)
        self._root_frame.setObjectName("root")
        outer.addWidget(self._root_frame)

        inner = QVBoxLayout(self._root_frame)
        inner.setContentsMargins(10, 6, 10, 8)
        inner.setSpacing(4)

        # Thin top bar: just a minimize icon, pinned right. No "Unified Usage
        # HUD" title, no timestamp — the refresh metadata goes into the
        # window tooltip so it stays discoverable without eating a row.
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(4)
        top_bar.addStretch(1)
        self.minimize_button = self._build_icon_button(
            "minus",
            tooltip="Minimize · \u2318M",
            accessible="Minimize window",
        )
        self.minimize_button.clicked.connect(self._minimize_window)
        top_bar.addWidget(self.minimize_button)
        inner.addLayout(top_bar)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(2)
        inner.addLayout(self.grid)
        self._configure_grid_columns()

        self.error_label = QLabel("")
        self.error_label.setObjectName("status")
        self.error_label.hide()
        inner.addWidget(self.error_label)

    def _configure_grid_columns(self) -> None:
        """Pin column widths using QFontMetrics so refreshes don't reflow.

        Widths use the widest plausible strings ("Weekly", "100.0%",
        "+100pt (100%)") rather than the current cell text so the column
        doesn't twitch as values change. COL_DETAIL gets the grid stretch
        so any extra width flows into the detail column, keeping the bar
        anchored to a fixed pixel position.
        """

        fm = QFontMetrics(self.font())
        self.grid.setColumnMinimumWidth(COL_LOGO, PROVIDER_LOGO_PX + 6)
        self.grid.setColumnMinimumWidth(COL_LABEL, fm.horizontalAdvance("Weekly") + 4)
        self.grid.setColumnMinimumWidth(COL_VALUE, fm.horizontalAdvance("100.0%") + 4)
        self.grid.setColumnMinimumWidth(
            COL_BAR,
            PACE_BAR_CELLS * (PaceBarWidget.CELL_WIDTH_PX + PaceBarWidget.CELL_GAP_PX) + 4,
        )
        self.grid.setColumnMinimumWidth(COL_DETAIL, fm.horizontalAdvance("+100pt (100%)") + 4)
        self.grid.setColumnMinimumWidth(COL_MUTE, UI_ICON_PX + 10)
        self.grid.setColumnStretch(COL_DETAIL, 1)

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
        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence("Meta+W"), self, activated=self.close)
        QShortcut(QKeySequence("Meta+M"), self, activated=self._minimize_window)
        QShortcut(QKeySequence("R"), self, activated=self.refresh_now)

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
        self._thread_pool.start(worker)

    def _apply_refresh_result(
        self, bundle: SnapshotBundle, sections: tuple[ProviderSection, ...]
    ) -> None:
        self._refresh_in_flight = False
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
            f"Updated {timestamp} · refresh every {self.config.interval_seconds:.0f}s"
        )

    def _apply_refresh_error(self, message: str) -> None:
        self._refresh_in_flight = False
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setStyleSheet(f"color: {COLORS['yellow']};")
        self.error_label.show()

    # --- grid rendering --------------------------------------------------

    def _render_grid(self, sections: tuple[ProviderSection, ...]) -> None:
        _clear_layout(self.grid)
        row = 0
        for section in sections:
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
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
            )
            logo.setToolTip(section.title)
            self.grid.addWidget(
                logo,
                start_row,
                COL_LOGO,
                row_count,
                1,
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            )

            for metric_row in section.rows:
                self._add_metric_row(row, metric_row)
                row += 1
            for note in section.notes:
                self._add_note_row(row, note)
                row += 1

            if section.provider in MUTE_PATHS:
                self.grid.addWidget(
                    self._build_mute_button(section.provider),
                    start_row,
                    COL_MUTE,
                    1,
                    1,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                )

    def _add_metric_row(self, row: int, metric_row: MetricRow) -> None:
        label_text = metric_row.label
        if metric_row.prefix:
            label_text = f"{metric_row.prefix} {label_text}"
        label = QLabel(label_text)
        if metric_row.prefix:
            label.setStyleSheet(
                f"color: {color_for_style(metric_row.prefix_style)};"
            )
        else:
            style_name = "dim" if metric_row.display_mode == "value_only" else "bold"
            label.setStyleSheet(f"color: {color_for_style(style_name)};")
        self.grid.addWidget(
            label,
            row,
            COL_LABEL,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        if metric_row.utilization is None:
            value = QLabel("--")
        else:
            value = QLabel(usage_hud.fmt_pct(metric_row.utilization).strip())
        value.setStyleSheet(
            f"color: {color_for_style(_style_name_for_metric(metric_row))};"
        )
        self.grid.addWidget(
            value,
            row,
            COL_VALUE,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
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
            detail = QLabel(_format_detail(metric_row))
            detail.setStyleSheet(
                f"color: {color_for_style(_style_name_for_delta(metric_row))};"
            )
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
                COL_MUTE - COL_BAR,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )

    def _add_note_row(self, row: int, note: NoteLine) -> None:
        label = QLabel(note.text)
        label.setStyleSheet(f"color: {color_for_style(note.style)};")
        label.setWordWrap(False)
        # Span from label through detail so notes read as an inset comment
        # under the provider's metric rows without crossing the mute column.
        self.grid.addWidget(
            label,
            row,
            COL_LABEL,
            1,
            COL_MUTE - COL_LABEL,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

    # --- controls --------------------------------------------------------

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

    def _build_mute_button(self, provider: ProviderName) -> QPushButton:
        muted = self._is_muted(provider)
        icon_name = "bell-slash" if muted else "bell"
        action = "Unmute" if muted else "Mute"
        button = self._build_icon_button(
            icon_name,
            tooltip=f"{action} {provider.title()} notifications",
            accessible=f"{action} {provider} notifications",
            checkable=True,
        )
        button.setChecked(muted)
        button.clicked.connect(
            lambda _checked=False, provider=provider: self._toggle_mute(provider)
        )
        return button

    def _is_muted(self, provider: ProviderName) -> bool:
        path = MUTE_PATHS.get(provider)
        return bool(path and path.exists())

    def _toggle_mute(self, provider: ProviderName) -> None:
        path = MUTE_PATHS.get(provider)
        if path is None:
            return

        try:
            if path.exists():
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
        except OSError as exc:
            self._show_error(f"Could not update {provider} mute: {exc}")
            return

        self._render_grid(self._last_sections)

    def _save_current_state(self) -> None:
        position = self.pos()
        self._ui_state.window_position = (position.x(), position.y())
        save_ui_state(self._ui_state)
