"""Custom QWidget that paints a pace bar from shared ``pace_bar_runs`` cells.

A pace bar shows three things at once:

- actual utilization (how many cells are filled)
- expected utilization (a vertical marker line)
- whether the caller is on/under/over pace (the tone of the filled cells)

In the text HUD those three are packed into one line of Unicode block
characters; here we draw them with ``QPainter`` so the result is pixel
accurate at any DPR and doesn't rely on rich-text ``QLabel`` kerning.

The pacing logic lives in :func:`model_usage_hud.core.builders.pace_bar_runs`
so the CLI ANSI renderer and this widget cannot drift — we only translate
the resulting ``PaceBarCell`` tones into concrete ``QColor`` fills.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from model_usage_hud.app.styles import COLORS
from model_usage_hud.core.builders import pace_bar_runs
from model_usage_hud.core.models import PaceBarCell, PaceBarTone


# Map abstract tones to concrete fills. ``dim`` is intentionally darker than
# the text HUD's ``muted`` so empty cells read as faint placeholders rather
# than as another layer of body text next to the colored cells.
_TONE_TO_HEX: dict[PaceBarTone, str] = {
    "cyan": COLORS["cyan"],
    "red": COLORS["red"],
    "green": COLORS["green"],
    "orange": COLORS["orange"],
    "white": COLORS["white"],
    "dim": COLORS["border"],
}


class PaceBarWidget(QWidget):
    """Paint a pace bar driven by ``pace_bar_runs`` output.

    Callers push raw percentages via :meth:`set_pace`; the widget owns its
    own call to ``pace_bar_runs`` so the host layout never has to think
    about cells. Sizing is fixed via :meth:`sizeHint` so column widths in
    the parent grid stay predictable across refreshes.
    """

    # Visual tuning — picked to match the OG ANSI bar's density while being
    # pixel-accurate. Each cell is a small vertical block; the marker is a
    # thin line drawn inside its own cell slot.
    CELL_WIDTH_PX = 5
    CELL_GAP_PX = 1
    BAR_HEIGHT_PX = 12
    MARKER_WIDTH_PX = 2

    def __init__(
        self,
        width_cells: int = 16,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._width_cells = max(1, int(width_cells))
        self._cells: tuple[PaceBarCell, ...] = ()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # ------------------------------------------------------------------ API

    def set_pace(
        self,
        actual_pct: float | int | None,
        expected_pct: float | int | None,
        *,
        stale: bool = False,
    ) -> None:
        """Update the bar. Passing ``None`` for either value clears the bar."""

        if actual_pct is None or expected_pct is None:
            if self._cells:
                self._cells = ()
                self.update()
            return

        cells = pace_bar_runs(
            actual_pct,
            expected_pct,
            width=self._width_cells,
            stale=stale,
        )
        if cells != self._cells:
            self._cells = cells
            self.update()

    def set_width_cells(self, width_cells: int) -> None:
        width_cells = max(1, int(width_cells))
        if width_cells == self._width_cells:
            return
        self._width_cells = width_cells
        # Keep the existing cell tuple in shape by recomputing if we have one,
        # otherwise leave it empty until the next ``set_pace`` call.
        self._cells = ()
        self.updateGeometry()
        self.update()

    # -------------------------------------------------------------- sizing

    def sizeHint(self) -> QSize:  # type: ignore[override]
        total_w = (
            self._width_cells * self.CELL_WIDTH_PX
            + max(0, self._width_cells - 1) * self.CELL_GAP_PX
        )
        return QSize(total_w, self.BAR_HEIGHT_PX)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return self.sizeHint()

    # -------------------------------------------------------------- paint

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        if not self._cells:
            return
        painter = QPainter(self)
        # Cells are axis-aligned rectangles — antialiasing just blurs them.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        try:
            self._paint_cells(painter)
        finally:
            painter.end()

    def _paint_cells(self, painter: QPainter) -> None:
        rect = self.rect()
        bar_h = float(min(rect.height(), self.BAR_HEIGHT_PX))
        top = (rect.height() - bar_h) / 2.0
        cell_w = float(self.CELL_WIDTH_PX)
        gap = float(self.CELL_GAP_PX)
        marker_w = float(self.MARKER_WIDTH_PX)

        x = 0.0
        for cell in self._cells:
            color = QColor(_TONE_TO_HEX.get(cell.tone, COLORS["fg"]))
            if cell.kind == "marker":
                # Center a thin line inside the cell's allotted horizontal slot,
                # full bar height. Reads as a tick against the fill behind it.
                painter.fillRect(
                    QRectF(
                        x + (cell_w - marker_w) / 2.0,
                        top,
                        marker_w,
                        bar_h,
                    ),
                    color,
                )
            else:
                painter.fillRect(
                    QRectF(x, top, cell_w, bar_h),
                    color,
                )
            x += cell_w + gap
