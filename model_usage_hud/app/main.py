"""PySide6 desktop app entrypoint for the usage HUD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import usage_hud


DEFAULT_FONT_SIZE = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the PySide6 desktop HUD.")
    parser.add_argument(
        "--codex-sessions-dir",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
        help="Codex sessions directory (default: ~/.codex/sessions)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Refresh interval in seconds (default: 30 with --decimals, 60 otherwise)",
    )
    parser.add_argument(
        "--providers",
        default="all",
        help="Comma-separated providers to display: claude,codex,gemini or all (default: all)",
    )
    parser.add_argument(
        "--all-limits",
        action="store_true",
        help="Show all Codex limit buckets",
    )
    parser.add_argument(
        "--speedometer",
        action="store_true",
        help="Show burn-rate and ETA metadata in the app rows",
    )
    parser.add_argument(
        "--decimals",
        action="store_true",
        help="Show one decimal place on percentages",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing HUD lock",
    )
    parser.add_argument(
        "--always-on-top",
        dest="always_on_top",
        action="store_true",
        default=sys.platform == "darwin",
        help="Keep the app window above other windows",
    )
    parser.add_argument(
        "--no-always-on-top",
        dest="always_on_top",
        action="store_false",
        help="Disable always-on-top mode",
    )
    parser.add_argument(
        "--frameless",
        dest="frameless",
        action="store_true",
        default=sys.platform == "darwin",
        help="Hide the title bar and allow drag-to-move",
    )
    parser.add_argument(
        "--framed",
        dest="frameless",
        action="store_false",
        help="Show the normal window frame",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=None,
        help=(
            f"Base font size for the app window (default {DEFAULT_FONT_SIZE:.0f}pt, "
            "⌘+/⌘-/⌘0 to zoom at runtime)"
        ),
    )
    parser.add_argument(
        "--geometry",
        default=None,
        help="Initial geometry in WIDTHxHEIGHT+X+Y format",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry_explicit = args.geometry is not None
    font_size_explicit = args.font_size is not None
    providers_explicit = any(
        arg == "--providers" or arg.startswith("--providers=")
        for arg in sys.argv[1:]
    )

    try:
        selected_providers = usage_hud.parse_provider_selection(args.providers)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.interval is None:
        args.interval = 30.0 if args.decimals else 60.0
    if args.interval <= 0:
        print("--interval must be > 0", file=sys.stderr)
        return 2
    if args.font_size is None:
        args.font_size = DEFAULT_FONT_SIZE
    if args.font_size <= 0:
        print("--font-size must be > 0", file=sys.stderr)
        return 2

    usage_hud.DECIMALS = bool(args.decimals)
    usage_hud.SPEEDOMETER_ENABLED = bool(args.speedometer)

    # When the user didn't pass --geometry we leave it as None and let the
    # window auto-fit from its grid sizeHint after the first refresh. The
    # Tk HUD's build_default_topmost_geometry is tuned for text metrics and
    # doesn't match the PySide grid, so we no longer use it here.

    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        print(
            "PySide6 is not installed. Install it with `pip install -e '.[gui]'`.",
            file=sys.stderr,
        )
        return 1

    from model_usage_hud.app.window import AppConfig, HudWindow

    try:
        with usage_hud.single_instance_lock(force=args.force):
            app = QApplication(sys.argv)
            window = HudWindow(
                AppConfig(
                    selected_providers=selected_providers,
                    codex_sessions_dir=args.codex_sessions_dir.expanduser(),
                    all_limits=bool(args.all_limits),
                    interval_seconds=float(args.interval),
                    always_on_top=bool(args.always_on_top),
                    frameless=bool(args.frameless),
                    font_size=float(args.font_size),
                    geometry=str(args.geometry) if args.geometry is not None else None,
                    geometry_explicit=geometry_explicit,
                    force=bool(args.force),
                    font_size_explicit=font_size_explicit,
                    providers_explicit=providers_explicit,
                )
            )
            window.show()
            return app.exec()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
