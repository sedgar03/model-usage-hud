"""Shared styles for the minimal desktop app."""

from __future__ import annotations

from model_usage_hud.core.models import TextStyle

COLORS = {
    "bg": "#111315",
    "panel": "#171a1d",
    "border": "#2a2d30",
    "fg": "#e6edf3",
    "muted": "#8b949e",
    "red": "#f85149",
    "green": "#3fb950",
    "yellow": "#d29922",
    "cyan": "#58a6ff",
    "white": "#e6edf3",
    "orange": "#d18616",
    "brown": "#b87333",
}

STYLE_TO_COLOR: dict[TextStyle, str] = {
    "plain": COLORS["fg"],
    "bold": COLORS["fg"],
    "dim": COLORS["muted"],
    "red": COLORS["red"],
    "green": COLORS["green"],
    "yellow": COLORS["yellow"],
    "cyan": COLORS["cyan"],
    "white": COLORS["white"],
    "orange": COLORS["orange"],
    "brown": COLORS["brown"],
    "bold_red": COLORS["red"],
    "bold_green": COLORS["green"],
    "bold_white": COLORS["white"],
}


def color_for_style(style: TextStyle) -> str:
    return STYLE_TO_COLOR.get(style, COLORS["fg"])


def build_stylesheet(font_size: float) -> str:
    # The universal ``QWidget`` rule only sets typography and text color —
    # it deliberately omits ``background`` so frameless-translucent windows
    # don't double-paint over the QFrame#root rounded fill. Widgets that
    # need a panel fill (the root frame) set it explicitly below.
    return f"""
QWidget {{
    color: {COLORS["fg"]};
    font-family: Menlo;
    font-size: {font_size}pt;
}}

QFrame#root {{
    background: {COLORS["bg"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
}}

QFrame#card {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
}}

QLabel#title {{
    font-weight: 600;
}}

QLabel#status {{
    color: {COLORS["muted"]};
}}

QLabel#headerMeta {{
    color: {COLORS["muted"]};
}}

QPushButton#controlButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    color: {COLORS["fg"]};
    padding: 1px;
}}

QPushButton#controlButton:hover {{
    background: {COLORS["panel"]};
    border-color: {COLORS["border"]};
}}

QPushButton#controlButton:focus {{
    border-color: {COLORS["cyan"]};
    outline: none;
}}

QPushButton#controlButton:checked {{
    border-color: {COLORS["yellow"]};
    color: {COLORS["yellow"]};
}}
"""
