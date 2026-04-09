"""Minimal PySide6 desktop window for the usage HUD."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import usage_hud
from model_usage_hud.app.styles import COLORS, build_stylesheet, color_for_style
from model_usage_hud.core.models import MetricRow, ProviderSection, SnapshotBundle
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


GEOMETRY_RE = re.compile(r"^(?P<w>\d+)x(?P<h>\d+)\+(?P<x>-?\d+)\+(?P<y>-?\d+)$")


@dataclass(slots=True)
class AppConfig:
    selected_providers: set[str]
    codex_sessions_dir: Path
    all_limits: bool
    interval_seconds: float
    always_on_top: bool
    frameless: bool
    font_size: float
    geometry: str
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


def _plain_fill_bar(pct: int | float, width: int) -> str:
    filled = int(round((max(0.0, min(float(pct), 100.0)) / 100.0) * width))
    return ("█" * filled) + ("░" * (width - filled))


def _plain_pace_bar(actual_pct: int | float, expected_pct: int | float, width: int) -> str:
    actual_units = int(round((max(0.0, min(float(actual_pct), 100.0)) / 100.0) * width))
    expected_units = int(round((max(0.0, min(float(expected_pct), 100.0)) / 100.0) * width))
    marker_idx = max(0, min(width - 1, expected_units - 1 if expected_units > 0 else 0))
    pieces: list[str] = []
    for i in range(width):
        pos = i + 1
        if i == marker_idx:
            pieces.append("│")
        elif pos <= actual_units:
            pieces.append("█")
        else:
            pieces.append("░")
    return "".join(pieces)


def _format_speedometer_text(row: MetricRow) -> str:
    rate = row.burn_rate_per_hour
    if not usage_hud.SPEEDOMETER_ENABLED or rate is None:
        return ""
    eta = row.eta_hours
    eta_text = f" {usage_hud._format_eta(eta)}" if eta is not None else ""
    return f"  \u23F1 +{rate:.0f}%/h{eta_text}"


def _format_metric_plain(row: MetricRow) -> tuple[str, str, str]:
    label = f"{row.label}"
    if row.prefix:
        label = f"{row.prefix} {label}"

    if row.utilization is None:
        return label, "--%", "no data"

    pct_text = usage_hud.fmt_pct(row.utilization)
    if row.display_mode == "value_only":
        return label, pct_text, ""

    if row.expected_utilization is None:
        return label, pct_text, _plain_fill_bar(row.utilization, 14) + _format_speedometer_text(row)

    delta_text = usage_hud.fmt_delta(row.delta if row.delta is not None else 0)
    target_text = f"({usage_hud.fmt_pct(row.expected_utilization).strip()})"
    bar = _plain_pace_bar(row.utilization, row.expected_utilization, 16)
    return label, pct_text, f"{bar} {delta_text} {target_text}{_format_speedometer_text(row)}"


def _clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            while child_layout.count():
                child_item = child_layout.takeAt(0)
                child_widget = child_item.widget()
                if child_widget is not None:
                    child_widget.deleteLater()


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
    """Minimal always-on-top HUD window driven by provider view models."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._thread_pool = QThreadPool.globalInstance()
        self._refresh_in_flight = False
        self._drag_origin = None

        self.setObjectName("root")
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
        if self.config.always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        if self.config.frameless:
            flags |= Qt.WindowType.FramelessWindowHint
        self.setWindowFlags(flags)

    def _build_layout(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        self.header_label = QLabel("Refreshing…")
        self.header_label.setObjectName("title")
        root_layout.addWidget(self.header_label)

        self.meta_label = QLabel("")
        self.meta_label.setObjectName("headerMeta")
        root_layout.addWidget(self.meta_label)

        self.sections_host = QVBoxLayout()
        self.sections_host.setSpacing(8)
        root_layout.addLayout(self.sections_host)

        self.error_label = QLabel("")
        self.error_label.setObjectName("status")
        self.error_label.hide()
        root_layout.addWidget(self.error_label)

    def _apply_geometry(self) -> None:
        width, height, x, y = _parse_geometry(self.config.geometry)
        self.resize(width, max(60, height))
        self.move(x, y)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence("Meta+W"), self, activated=self.close)
        QShortcut(QKeySequence("R"), self, activated=self.refresh_now)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self.config.frameless and event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.config.frameless and self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def refresh_now(self) -> None:
        if self._refresh_in_flight:
            return
        self._refresh_in_flight = True
        worker = RefreshWorker(self.config)
        worker.signals.succeeded.connect(self._apply_refresh_result)
        worker.signals.failed.connect(self._apply_refresh_error)
        self._thread_pool.start(worker)

    def _apply_refresh_result(self, bundle: SnapshotBundle, sections: tuple[ProviderSection, ...]) -> None:
        self._refresh_in_flight = False
        self.error_label.hide()
        timestamp = datetime.fromisoformat(bundle.generated_at).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        self.header_label.setText("Unified Usage HUD")
        self.meta_label.setText(f"Updated {timestamp}")
        _clear_layout(self.sections_host)
        for section in sections:
            self.sections_host.addWidget(self._build_section_card(section))
        self.adjustSize()

    def _apply_refresh_error(self, message: str) -> None:
        self._refresh_in_flight = False
        self.error_label.setText(message)
        self.error_label.setStyleSheet(f"color: {COLORS['yellow']};")
        self.error_label.show()

    def _build_section_card(self, section: ProviderSection) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel(section.title)
        title.setObjectName("title")
        title.setStyleSheet(f"color: {COLORS.get(section.accent, COLORS['fg'])};")
        header.addWidget(title)
        header.addStretch(1)

        meta_parts: list[str] = []
        if section.highest_utilization is not None:
            meta_parts.append(usage_hud.fmt_pct(section.highest_utilization).strip())
        if section.status:
            meta_parts.append(section.status)
        header_meta = QLabel("  ".join(meta_parts))
        header_meta.setObjectName("headerMeta")
        header.addWidget(header_meta)
        layout.addLayout(header)

        for row in section.rows:
            layout.addWidget(self._build_metric_widget(row))
        for note in section.notes:
            note_label = QLabel(note.text)
            note_label.setStyleSheet(f"color: {color_for_style(note.style)};")
            layout.addWidget(note_label)

        return card

    def _build_metric_widget(self, row: MetricRow) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        label_text, value_text, detail_text = _format_metric_plain(row)
        label = QLabel(label_text)
        label.setMinimumWidth(58)
        if row.prefix:
            label.setStyleSheet(f"color: {color_for_style(row.prefix_style)};")
        else:
            label.setStyleSheet(f"color: {color_for_style('dim' if row.display_mode == 'value_only' else 'bold')};")
        layout.addWidget(label)

        value = QLabel(value_text)
        value.setMinimumWidth(34)
        value.setStyleSheet(f"color: {color_for_style(_style_name_for_metric(row))};")
        layout.addWidget(value)

        detail = QLabel(detail_text)
        detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        detail.setStyleSheet(f"color: {color_for_style(_style_name_for_delta(row))};")
        layout.addWidget(detail, 1)

        return widget
