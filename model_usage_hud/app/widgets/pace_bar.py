"""Continuous-track pace visualization for the desktop HUD.

Unlike the CLI's ANSI ``build_pace_bar``, which renders 16 discrete Unicode
block cells because text is its only tool, this widget paints a single
horizontal track with three layers:

- A **blue fill** from 0 to ``min(actual, expected)`` — the "safe
  consumption envelope" that elapsed time has unlocked for you.
- A **red overshoot segment** from ``expected`` to ``actual`` when you're
  over pace — a concrete visual of *how much* you've exceeded the
  envelope, not just the binary "over or not". Drops to orange when the
  payload is stale so a missing refresh doesn't masquerade as over-pace.
- A **green remaining segment** from ``actual`` to ``expected`` when
  under pace — shows how much allowance is left.
- A **status-colored dot** at ``actual_pct`` — cyan/green when under
  pace, red/yellow when over, white when on pace, orange when stale.
  The dot is the crisp read-out; the segments behind it supply
  magnitude.

The on-pace tolerance is tight (±1 pt) because the user cares about small
habit drifts — being 3% over is a different signal than being on pace.

Width scales with the grid column so the track is always readable; the
dot diameter is fixed because it's the visual anchor. Callers push raw
percentages via :meth:`set_pace`; everything inside is just geometry.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from model_usage_hud.app.styles import COLORS


class PaceBarWidget(QWidget):
    """Paint a pace track: dim background, blue "used" fill, green "remaining", status dot."""

    # Visual tuning — tight enough to read at a glance without dominating the row.
    WIDGET_WIDTH_PX = 120
    WIDGET_HEIGHT_PX = 14
    TRACK_HEIGHT_PX = 4
    DOT_DIAMETER_PX = 10
    # Tolerance in percentage points for calling a dot "on pace" vs
    # under/over. Deliberately tight (±1) — the user tracks habit drift
    # and cares about a few percent delta, so "close enough" has to be
    # nearly exactly on pace.
    ON_PACE_TOLERANCE = 1.0
    # Threshold in percentage points for flipping the dot to the
    # "comfortably under" / "materially over" high-contrast colors.
    STRONG_DELTA = 5.0

    def __init__(
        self,
        width_cells: int | None = None,  # legacy — ignored, kept for call-site compat
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        del width_cells
        self._actual_pct: float | None = None
        self._expected_pct: float | None = None
        self._stale = False
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
        """Update the pace values. Passing ``None`` for either clears the track."""

        if actual_pct is None or expected_pct is None:
            if self._actual_pct is not None or self._expected_pct is not None:
                self._actual_pct = None
                self._expected_pct = None
                self._stale = False
                self.update()
            return

        actual = max(0.0, min(100.0, float(actual_pct)))
        expected = max(0.0, min(100.0, float(expected_pct)))
        if (
            self._actual_pct == actual
            and self._expected_pct == expected
            and self._stale == stale
        ):
            return
        self._actual_pct = actual
        self._expected_pct = expected
        self._stale = stale
        self.update()

    # -------------------------------------------------------------- sizing

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(self.WIDGET_WIDTH_PX, self.WIDGET_HEIGHT_PX)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return self.sizeHint()

    # -------------------------------------------------------------- paint

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        if self._actual_pct is None or self._expected_pct is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        rect = self.rect()
        w = float(rect.width())
        h = float(rect.height())

        # Reserve half a dot-diameter of padding on each side so the dot
        # never clips at 0% or 100% — the track visually shortens slightly
        # but stays pixel-aligned with the label row above/below it.
        pad = self.DOT_DIAMETER_PX / 2.0
        track_w = max(1.0, w - 2.0 * pad)
        track_h = float(self.TRACK_HEIGHT_PX)
        track_y = (h - track_h) / 2.0
        track_x = pad
        radius = track_h / 2.0

        # 1. Dim background track — establishes the 0-100% range visually.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["border"]))
        painter.drawRoundedRect(
            QRectF(track_x, track_y, track_w, track_h),
            radius,
            radius,
        )

        expected_frac = self._expected_pct / 100.0 if self._expected_pct else 0.0
        actual_frac = self._actual_pct / 100.0 if self._actual_pct else 0.0

        # 2. Blue "used" fill — from 0 to min(actual, expected). When
        # under-pace this is just ``actual`` and the dot sits inside the
        # blue; when over-pace it stops at ``expected`` and the red
        # overshoot segment takes over from there. Orange when stale so
        # the user doesn't mistake missing data for a pace reading.
        safe_frac = min(actual_frac, expected_frac)
        safe_w = track_w * safe_frac
        if safe_w > 0.0:
            safe_color = QColor(
                COLORS["orange"] if self._stale else COLORS["cyan"]
            )
            painter.setBrush(safe_color)
            painter.drawRoundedRect(
                QRectF(track_x, track_y, safe_w, track_h),
                radius,
                radius,
            )

        # 3. Pace-gap segment — shows the distance between actual and
        # expected as a colored fill *past* the dot. Green when under-pace
        # (the allowance you haven't used yet), red when over-pace (the
        # amount you've exceeded). This makes the relationship between
        # actual and expected immediately visible: at 16% actual with 35%
        # expected, a green bar stretches from 16% to 35%.
        if not self._stale and self._expected_pct is not None:
            if actual_frac < expected_frac:
                gap_start = track_x + track_w * actual_frac
                gap_end = track_x + track_w * expected_frac
                gap_w = max(0.0, gap_end - gap_start)
                if gap_w > 0.0:
                    painter.setBrush(QColor(COLORS["green"]))
                    painter.drawRoundedRect(
                        QRectF(gap_start, track_y, gap_w, track_h),
                        radius,
                        radius,
                    )
            elif actual_frac > expected_frac:
                over_start = track_x + track_w * expected_frac
                over_end = track_x + track_w * actual_frac
                over_w = max(0.0, over_end - over_start)
                if over_w > 0.0:
                    painter.setBrush(QColor(COLORS["red"]))
                    painter.drawRoundedRect(
                        QRectF(over_start, track_y, over_w, track_h),
                        radius,
                        radius,
                    )

        # 4. Actual dot — position = actual_pct along the track, color =
        # pacing state relative to expected. A thin background-color ring
        # around the dot keeps it crisp when it overlaps fills behind it.
        dot_cx = track_x + track_w * actual_frac
        dot_cy = h / 2.0
        dot_color = self._dot_color()

        painter.setPen(QPen(QColor(COLORS["bg"]), 1.5))
        painter.setBrush(dot_color)
        painter.drawEllipse(
            QPointF(dot_cx, dot_cy),
            self.DOT_DIAMETER_PX / 2.0,
            self.DOT_DIAMETER_PX / 2.0,
        )

    def _dot_color(self) -> QColor:
        """Map the actual-vs-expected delta to a 5-step color scale.

        Bucketed finer than the old ±3/on-pace split so ``+2`` reads
        differently from ``+0`` — the user uses the bar to nudge habits
        and a "close enough" zone hid that feedback. Scale:

        =======================  =================================
        delta (pct points)       color
        =======================  =================================
        stale data               orange (status, not pace)
        >= +STRONG_DELTA (+5)    red       (materially over)
        > +ON_PACE_TOLERANCE     yellow    (slightly over)
        within ±ON_PACE_TOL      white     (on pace)
        < -ON_PACE_TOLERANCE     cyan      (slightly under)
        <= -STRONG_DELTA (-5)    green     (comfortably under)
        =======================  =================================
        """

        if self._stale:
            return QColor(COLORS["orange"])
        assert self._actual_pct is not None and self._expected_pct is not None
        delta = self._actual_pct - self._expected_pct
        if delta >= self.STRONG_DELTA:
            return QColor(COLORS["red"])
        if delta > self.ON_PACE_TOLERANCE:
            return QColor(COLORS["yellow"])
        if delta <= -self.STRONG_DELTA:
            return QColor(COLORS["green"])
        if delta < -self.ON_PACE_TOLERANCE:
            return QColor(COLORS["cyan"])
        return QColor(COLORS["white"])
