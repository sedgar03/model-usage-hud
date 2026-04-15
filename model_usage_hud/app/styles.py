"""Shared styles for the minimal desktop app."""

from __future__ import annotations

from model_usage_hud.core.models import TextStyle

COLORS = {
    "bg": "#111315",
    "panel": "#171a1d",
    "border": "#2a2d30",
    "fg": "#f0f6fc",
    "muted": "#8b949e",
    "red": "#ff7b72",
    "green": "#56d364",
    "yellow": "#e3b341",
    "cyan": "#79c0ff",
    "white": "#ffffff",
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

QFrame#settingsPanel {{
    background: {COLORS["bg"]};
    border-top: 1px solid {COLORS["border"]};
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}}

QPushButton#settingsSelect {{
    background: {COLORS["bg"]};
    color: {COLORS["fg"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 2px 6px;
    font-family: Menlo;
    text-align: left;
    min-width: 170px;
}}

QPushButton#settingsSelect:hover,
QPushButton#settingsSelect:focus {{
    border-color: {COLORS["cyan"]};
    outline: none;
}}

QFrame#settingsPopup {{
    background: {COLORS["bg"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
}}

QScrollArea#settingsPopupScroll,
QWidget#settingsPopupContent {{
    background: {COLORS["bg"]};
    border: none;
}}

QLabel#settingsPopupHeader {{
    background: {COLORS["bg"]};
    color: {COLORS["muted"]};
    padding: 5px 8px 2px 8px;
}}

QPushButton#settingsPopupItem {{
    background: {COLORS["bg"]};
    color: {COLORS["fg"]};
    border: none;
    border-radius: 3px;
    padding: 4px 8px;
    text-align: left;
    font-family: Menlo;
}}

QPushButton#settingsPopupItem:hover,
QPushButton#settingsPopupItem:focus {{
    background: {COLORS["border"]};
    color: {COLORS["fg"]};
    outline: none;
}}

QDoubleSpinBox#settingsSpinBox {{
    background: {COLORS["bg"]};
    color: {COLORS["fg"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 2px 6px;
    font-family: Menlo;
    min-width: 70px;
}}

QSlider#settingsSlider::groove:horizontal {{
    background: {COLORS["border"]};
    border-radius: 2px;
    height: 4px;
}}

QSlider#settingsSlider::sub-page:horizontal {{
    background: {COLORS["green"]};
    border-radius: 2px;
}}

QSlider#settingsSlider::handle:horizontal {{
    background: {COLORS["fg"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    width: 12px;
    margin: -5px 0;
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
    /* No extra border/tint on checked — the icon swap (e.g. bell ↔
       bell-slash) already signals state, and a yellow ring around a
       header control reads as a warning/error. */
    background: transparent;
    border-color: transparent;
    color: {COLORS["fg"]};
}}

QPushButton#providerToggleButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 1px;
}}

QPushButton#providerToggleButton:hover {{
    background: {COLORS["panel"]};
    border-color: {COLORS["border"]};
}}

QPushButton#providerToggleButton:focus {{
    border-color: {COLORS["cyan"]};
    outline: none;
}}
"""
