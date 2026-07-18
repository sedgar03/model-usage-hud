"""Simple level gauge for instantaneous metrics (no pace target).

Where :class:`~model_usage_hud.app.widgets.pace_bar.PaceBarWidget` visualizes
*actual vs. expected* pace (fills, a gap segment, and a status dot), a gauge is
for values that are just a level right now — CPU %, memory %, disk %. It paints
one thing: a dim track with a fill from 0 to the value, tinted green / yellow /
red by the same thresholds the CLI uses (:func:`usage_hud.usage_style`) so a
"redlining" bar looks identical across both HUDs.

Geometry is kept pixel-compatible with ``PaceBarWidget`` (same width, height,
and track thickness) so gauge rows and pace rows line up in the shared grid.
"""

from __future__ import annotations

import usage_hud
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from model_usage_hud.app.styles import COLORS, color_for_style


class GaugeBarWidget(QWidget):
    """Paint a single-level fill gauge tinted by utilization threshold."""

    # Match PaceBarWidget so the two bar kinds align in the grid.
    WIDGET_WIDTH_PX = 120
    WIDGET_HEIGHT_PX = 14
    TRACK_HEIGHT_PX = 4

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pct: float | None = None
        self._stale = False
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_value(self, pct: float | int | None, *, stale: bool = False) -> None:
        """Update the gauge. ``None`` clears the fill."""

        if pct is None:
            if self._pct is not None:
                self._pct = None
                self._stale = False
                self.update()
            return
        value = max(0.0, min(100.0, float(pct)))
        if self._pct == value and self._stale == stale:
            return
        self._pct = value
        self._stale = stale
        self.update()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(self.WIDGET_WIDTH_PX, self.WIDGET_HEIGHT_PX)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return self.sizeHint()

    def _fill_color(self) -> QColor:
        if self._stale:
            return QColor(COLORS["orange"])
        assert self._pct is not None
        # Reuse the CLI's threshold mapping so both HUDs redline together.
        style = usage_hud.usage_style(int(round(self._pct)))
        return QColor(color_for_style(style))

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        if self._pct is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            rect = self.rect()
            w = float(rect.width())
            h = float(rect.height())
            track_h = float(self.TRACK_HEIGHT_PX)
            track_y = (h - track_h) / 2.0
            radius = track_h / 2.0

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["border"]))
            painter.drawRoundedRect(QRectF(0.0, track_y, w, track_h), radius, radius)

            fill_w = w * (self._pct / 100.0)
            if fill_w > 0.0:
                painter.setBrush(self._fill_color())
                painter.drawRoundedRect(
                    QRectF(0.0, track_y, fill_w, track_h), radius, radius
                )
        finally:
            painter.end()
