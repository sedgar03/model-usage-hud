"""SVG icon loader and DPR-aware QIcon/QPixmap cache for the HUD app.

Assets are loaded via ``importlib.resources`` so they work inside wheels and
packaged ``.app`` bundles (Briefcase, see ``SPEC.md`` Phase 4). Icons are
rendered through ``QSvgRenderer`` into a pixmap sized for the caller's device
pixel ratio, then cached by ``(category, name, size, color, dpr_bucket)``.

Tinting model
-------------
The monochrome provider SVGs (Lobe Icons) and all Phosphor UI icons use
literal ``fill="currentColor"``. We recolor them by byte-substituting the
token in the SVG payload before handing it to ``QSvgRenderer`` — the most
reliable way to tint a single-color SVG without dragging in a full parser.

The color provider variants (``{name}-color.svg``) have hardcoded fills and
gradients and are rendered as-is.

Attribution
-----------
Provider logos come from Lobe Icons (https://github.com/lobehub/lobe-icons),
MIT-licensed, vendored under ``assets/providers/LICENSE-lobe-icons.txt``.
Provider logos are trademarks of their respective owners and are used here
solely to identify the services this HUD monitors (nominative fair use).

UI control icons come from Phosphor Icons
(https://github.com/phosphor-icons/core), MIT-licensed, vendored under
``assets/icons/LICENSE-phosphor.txt``.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from model_usage_hud.core.models import ProviderName

ProviderLogoMode = Literal["color", "mono"]

_ASSETS_ROOT = files(__package__) / "assets"
_PROVIDERS_ROOT = _ASSETS_ROOT / "providers"
_UI_ICONS_ROOT = _ASSETS_ROOT / "icons"

# Keyed by (category, name, size_px, color_or_none, dpr_bucket).
_PIXMAP_CACHE: dict[tuple[str, str, int, str | None, float], QPixmap] = {}


def _load_svg_bytes(root, name: str) -> bytes:
    return (root / f"{name}.svg").read_bytes()


def _tint_svg(svg_bytes: bytes, color: str) -> bytes:
    return svg_bytes.replace(b"currentColor", color.encode("ascii"))


def _dpr_bucket(dpr: float) -> float:
    # Round to quarter-step so tiny float jitter between monitors doesn't bust
    # the cache on every refresh.
    return round(float(dpr) * 4.0) / 4.0


def _render_pixmap(svg_bytes: bytes, size_px: int, dpr: float) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(svg_bytes))
    pixel_size = max(1, int(round(size_px * dpr)))
    pm = QPixmap(pixel_size, pixel_size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    renderer.render(painter, QRectF(0.0, 0.0, float(pixel_size), float(pixel_size)))
    painter.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def provider_pixmap(
    provider: ProviderName,
    *,
    size_px: int = 28,
    mode: ProviderLogoMode = "color",
    mono_color: str | None = None,
    dpr: float = 1.0,
) -> QPixmap:
    """Return a pixmap of the provider logo, rendered for the given DPR.

    ``mode="color"`` loads the brand-colored variant as-is.
    ``mode="mono"`` loads the monochrome variant and tints with ``mono_color``
    (required — the caller owns the palette, this module does not).
    """

    if mode == "mono" and not mono_color:
        raise ValueError("mode='mono' requires mono_color (e.g. '#D97757')")

    bucket = _dpr_bucket(dpr)
    color_key = mono_color if mode == "mono" else None
    key = ("provider", provider, size_px, color_key, bucket)
    cached = _PIXMAP_CACHE.get(key)
    if cached is not None:
        return cached

    if mode == "color":
        svg_bytes = _load_svg_bytes(_PROVIDERS_ROOT, f"{provider}-color")
    else:
        raw = _load_svg_bytes(_PROVIDERS_ROOT, provider)
        svg_bytes = _tint_svg(raw, mono_color)  # type: ignore[arg-type]

    pm = _render_pixmap(svg_bytes, size_px, dpr)
    _PIXMAP_CACHE[key] = pm
    return pm


def provider_icon(
    provider: ProviderName,
    *,
    size_px: int = 28,
    mode: ProviderLogoMode = "color",
    mono_color: str | None = None,
    dpr: float = 1.0,
) -> QIcon:
    return QIcon(
        provider_pixmap(
            provider,
            size_px=size_px,
            mode=mode,
            mono_color=mono_color,
            dpr=dpr,
        )
    )


def ui_pixmap(
    name: str,
    *,
    size_px: int = 14,
    color: str = "#e6edf3",
    dpr: float = 1.0,
) -> QPixmap:
    bucket = _dpr_bucket(dpr)
    key = ("ui", name, size_px, color, bucket)
    cached = _PIXMAP_CACHE.get(key)
    if cached is not None:
        return cached

    raw = _load_svg_bytes(_UI_ICONS_ROOT, name)
    svg_bytes = _tint_svg(raw, color)
    pm = _render_pixmap(svg_bytes, size_px, dpr)
    _PIXMAP_CACHE[key] = pm
    return pm


def ui_icon(
    name: str,
    *,
    size_px: int = 14,
    color: str = "#e6edf3",
    dpr: float = 1.0,
) -> QIcon:
    return QIcon(ui_pixmap(name, size_px=size_px, color=color, dpr=dpr))


def clear_cache() -> None:
    """Drop all cached pixmaps. Useful after a theme change or DPR change."""

    _PIXMAP_CACHE.clear()
