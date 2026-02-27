#!/usr/bin/env python3
"""Unified usage HUD for Claude Code + Codex + Gemini."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any

KEYCHAIN_SERVICE = "Claude Code-credentials"
CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
ANTHROPIC_BETA = "oauth-2025-04-20"

RESET = "\033[0m"
STYLE = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "brown": "\033[38;5;130m",
    "bold_red": "\033[1;31m",
    "bold_green": "\033[1;32m",
    "bold_white": "\033[1;37m",
}


@dataclass
class CodexRateWindow:
    used_percent: float
    window_minutes: int
    resets_at: int


@dataclass
class CodexSnapshot:
    timestamp: str
    limit_id: str
    primary: CodexRateWindow | None
    secondary: CodexRateWindow | None
    total_tokens: int | None
    model_context_window: int | None


class Ansi:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def style(self, text: str, style_name: str) -> str:
        if not self.enabled:
            return text
        code = STYLE.get(style_name)
        if not code:
            return text
        return f"{code}{text}{RESET}"


def supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if term.lower() == "dumb":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


ANSI = Ansi(supports_color())
BAR_STYLE = "legacy"
PROVIDER_ORDER = ("claude", "codex", "gemini")
DEFAULT_TOPMOST_GEOMETRY = "320x130+40+40"
SINGLE_PROVIDER_TOPMOST_WIDTH_SCALE = 0.65


def _default_lock_path() -> Path:
    env_path = os.environ.get("USAGE_HUD_LOCK_PATH")
    if env_path:
        return Path(env_path)
    return Path.home() / ".usage-hud" / "usage-hud.lock"


LOCK_PATH = _default_lock_path()


# ---------------------------------------------------------------------------
# Burn-rate tracker (speedometer)
# ---------------------------------------------------------------------------

@dataclass
class _Sample:
    timestamp: float
    pct: int


class BurnTracker:
    def __init__(self, maxlen: int = 15):
        self._maxlen = maxlen
        self._buffers: dict[tuple[str, str], deque[_Sample]] = {}

    def record(self, key: tuple[str, str], pct: int) -> None:
        buf = self._buffers.get(key)
        if buf and buf[-1].pct - pct > 5:
            buf.clear()
        if buf is None:
            buf = deque(maxlen=self._maxlen)
            self._buffers[key] = buf
        buf.append(_Sample(time.time(), pct))

    def burn_rate(self, key: tuple[str, str]) -> float | None:
        buf = self._buffers.get(key)
        if not buf or len(buf) < 2:
            return None
        oldest, newest = buf[0], buf[-1]
        dt_hours = (newest.timestamp - oldest.timestamp) / 3600
        if dt_hours <= 0:
            return None
        rate = (newest.pct - oldest.pct) / dt_hours
        return rate if rate > 0 else None

    def eta_hours(self, key: tuple[str, str]) -> float | None:
        rate = self.burn_rate(key)
        if rate is None:
            return None
        buf = self._buffers[key]
        remaining = 100 - buf[-1].pct
        if remaining <= 0:
            return 0.0
        return remaining / rate


BURN_TRACKER = BurnTracker(maxlen=15)
SPEEDOMETER_ENABLED = False
DECIMALS = False


def _format_eta(hours: float) -> str:
    if hours < 1:
        minutes = max(1, int(hours * 60))
        return f"~{minutes}m"
    if hours <= 48:
        if hours == int(hours):
            return f"~{int(hours)}h"
        return f"~{hours:.1f}h"
    days = hours / 24
    if days == int(days):
        return f"~{int(days)}d"
    return f"~{days:.1f}d"


def format_speedometer(key: tuple[str, str] | None) -> str:
    if key is None or not SPEEDOMETER_ENABLED:
        return ""
    rate = BURN_TRACKER.burn_rate(key)
    if rate is None:
        return ""
    eta = BURN_TRACKER.eta_hours(key)
    eta_str = f" {_format_eta(eta)}" if eta is not None else ""
    return ANSI.style(f" \u23F1 +{rate:.0f}%/h{eta_str}", "dim")


def clamp_pct(value: Any) -> int | float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0 if DECIMALS else 0
    if DECIMALS:
        pct = round(v, 1)
        return max(0.0, min(pct, 100.0))
    pct = int(round(v))
    return max(0, min(pct, 100))


def fmt_pct(value: int | float) -> str:
    if DECIMALS:
        return f"{value:>5.1f}%"
    return f"{value:>3}%"


def fmt_delta(value: int | float) -> str:
    if DECIMALS:
        return f"{value:+.1f}"
    return f"{value:+d}"


def parse_iso(ts: str | None) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    value = ts.strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()


def format_iso_reset(ts: str | None) -> str:
    dt = parse_iso(ts)
    if dt is None:
        return "--"
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_epoch_reset(epoch_seconds: int) -> str:
    if epoch_seconds <= 0:
        return "--"
    return datetime.fromtimestamp(epoch_seconds).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def format_reset_short_iso(ts: str | None) -> str:
    dt = parse_iso(ts)
    if dt is None:
        return "--"
    now_local = datetime.now().astimezone()
    if dt.date() == now_local.date():
        return dt.strftime("%I:%M%p").replace(" 0", " ").lower()
    return dt.strftime("%b%d %I:%M%p").replace(" 0", " ").lower()


def format_reset_short_epoch(epoch_seconds: int) -> str:
    if epoch_seconds <= 0:
        return "--"
    dt = datetime.fromtimestamp(epoch_seconds).astimezone()
    now_local = datetime.now().astimezone()
    if dt.date() == now_local.date():
        return dt.strftime("%I:%M%p").replace(" 0", " ").lower()
    return dt.strftime("%b%d %I:%M%p").replace(" 0", " ").lower()


def _seconds_until_epoch(epoch_seconds: int) -> int | None:
    if epoch_seconds <= 0:
        return None
    seconds_left = int(epoch_seconds - time.time())
    return max(0, seconds_left)


def _seconds_until_iso(ts: str | None) -> int | None:
    dt = parse_iso(ts)
    if dt is None:
        return None
    seconds_left = int(dt.timestamp() - time.time())
    return max(0, seconds_left)


def format_time_left(seconds_left: int | None) -> str:
    if seconds_left is None:
        return "--"
    if seconds_left == 0:
        return "resetting now"
    days, rem = divmod(seconds_left, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    pieces: list[str] = []
    if days:
        pieces.append(f"{days}d")
    if hours or days:
        pieces.append(f"{hours}h")
    pieces.append(f"{minutes}m")
    return "in " + " ".join(pieces)


def expected_pct_from_iso_reset(resets_at: str | None, window: timedelta, now_local: datetime) -> int | float | None:
    reset_dt = parse_iso(resets_at)
    if reset_dt is None:
        return None
    now = now_local.astimezone(reset_dt.tzinfo)
    start_dt = reset_dt - window
    elapsed = (now - start_dt).total_seconds() / window.total_seconds()
    return clamp_pct(elapsed * 100.0)


def expected_pct_from_epoch_reset(resets_at: int, window_minutes: int, now_epoch: float) -> int | float | None:
    if resets_at <= 0 or window_minutes <= 0:
        return None
    window_seconds = window_minutes * 60
    start_ts = resets_at - window_seconds
    elapsed = (now_epoch - start_ts) / window_seconds
    return clamp_pct(elapsed * 100.0)


def usage_style(pct: int) -> str:
    if pct >= 95:
        return "bold_red"
    if pct >= 80:
        return "yellow"
    return "green"


def pace_style(delta: int) -> str:
    if delta >= 15:
        return "bold_red"
    if delta >= 6:
        return "yellow"
    if delta <= -15:
        return "green"
    if delta <= -6:
        return "green"
    return "cyan"


def pct_bar(pct: int, width: int = 16) -> str:
    pct = max(0, min(pct, 100))
    filled = int(round((pct / 100.0) * width))
    if BAR_STYLE == "solid":
        fill = "█" * filled
        empty = "█" * (width - filled)
    else:
        fill = "▓" * filled
        empty = "░" * (width - filled)
    return ANSI.style(fill, usage_style(pct)) + ANSI.style(empty, "dim")


def build_pace_bar(actual_pct: int, expected_pct: int, width: int = 24) -> str:
    actual_units = int(round((max(0, min(actual_pct, 100)) / 100.0) * width))
    expected_units = int(round((max(0, min(expected_pct, 100)) / 100.0) * width))
    marker_idx = max(0, min(width - 1, expected_units - 1 if expected_units > 0 else 0))

    if BAR_STYLE == "solid":
        pieces: list[str] = []
        for i in range(width):
            pos = i + 1
            cell_style = "dim"
            if pos <= min(actual_units, expected_units):
                cell_style = "cyan"
            elif actual_units > expected_units and expected_units < pos <= actual_units:
                cell_style = "red"
            elif expected_units > actual_units and actual_units < pos <= expected_units:
                cell_style = "green"
            elif pos <= actual_units:
                cell_style = "cyan"

            if i == marker_idx:
                marker_style = "red" if actual_units > expected_units else "green"
                if abs(actual_units - expected_units) <= 1:
                    marker_style = "white"
                pieces.append(ANSI.style("│", marker_style))
                continue

            pieces.append(ANSI.style("█", cell_style))
        return "".join(pieces)

    pieces: list[str] = []
    for i in range(width):
        pos = i + 1
        if i == marker_idx:
            marker_style = "red" if actual_units > expected_units else "green"
            if abs(actual_units - expected_units) <= 1:
                marker_style = "white"
            pieces.append(ANSI.style("│", marker_style))
            continue

        if pos <= min(actual_units, expected_units):
            pieces.append(ANSI.style("▓", "cyan"))
        elif actual_units > expected_units and expected_units < pos <= actual_units:
            pieces.append(ANSI.style("▓", "red"))
        elif expected_units > actual_units and actual_units < pos <= expected_units:
            pieces.append(ANSI.style("▒", "green"))
        elif pos <= actual_units:
            pieces.append(ANSI.style("▓", "cyan"))
        else:
            pieces.append(ANSI.style("░", "dim"))
    return "".join(pieces)


def status(pct: int, warn: int, critical: int) -> str:
    if pct >= critical:
        return ANSI.style("CRIT", "bold_red")
    if pct >= warn:
        return ANSI.style("WARN", "yellow")
    return ANSI.style("OK", "green")



ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
ANSI_CODE_RE = re.compile(r"\x1b\[([0-9;]*)m")

ANSI_TK_TAG_BY_CODE = {
    "1": "ansi_bold",
    "2": "ansi_dim",
    "31": "ansi_red",
    "32": "ansi_green",
    "33": "ansi_yellow",
    "36": "ansi_cyan",
    "37": "ansi_white",
    "38;5;130": "ansi_brown",
    "1;31": "ansi_bold_red",
    "1;32": "ansi_bold_green",
    "1;37": "ansi_bold_white",
}


def provider_badge_lines(provider: str, show_badge: bool = True) -> tuple[str, str]:
    if not show_badge:
        return "", ""

    if provider == "claude":
        top = ANSI.style("▀▄▄▄▀", "brown")
        bottom = ANSI.style("█▀█▀█", "brown")
    elif provider == "codex":
        top = ANSI.style("▄▀▀▀▄", "white")
        bottom = ANSI.style("▀▄█▄▀", "white")
    else:
        top = ANSI.style("▀█▀█▀", "cyan")
        bottom = ANSI.style("▄█▄█▄", "cyan")

    return f"{top} ", f"{bottom} "

def visible_len(text: str) -> int:
    return len(ANSI_ESCAPE_RE.sub("", text))


def insert_ansi_text_tk(widget: Any, text: str, font_size: int, default_fg: str) -> None:
    base_font = ("Menlo", font_size)
    bold_font = ("Menlo", font_size, "bold")

    widget.tag_configure("ansi_default", foreground=default_fg, font=base_font)
    widget.tag_configure("ansi_dim", foreground="#8b949e", font=base_font)
    widget.tag_configure("ansi_bold", foreground=default_fg, font=bold_font)
    widget.tag_configure("ansi_red", foreground="#d94848", font=base_font)
    widget.tag_configure("ansi_green", foreground="#78c28f", font=base_font)
    widget.tag_configure("ansi_yellow", foreground="#ffd43b", font=base_font)
    widget.tag_configure("ansi_cyan", foreground="#4f86b8", font=base_font)
    widget.tag_configure("ansi_white", foreground="#f8f9fa", font=base_font)
    widget.tag_configure("ansi_brown", foreground="#d9a066", font=base_font)
    widget.tag_configure("ansi_bold_red", foreground="#e03131", font=bold_font)
    widget.tag_configure("ansi_bold_green", foreground="#86cfa0", font=bold_font)
    widget.tag_configure("ansi_bold_white", foreground="#ffffff", font=bold_font)

    current_tag = "ansi_default"
    start = 0
    for match in ANSI_CODE_RE.finditer(text):
        segment = text[start : match.start()]
        if segment:
            widget.insert("end", segment, current_tag)
        code = match.group(1) or "0"
        if code == "0":
            current_tag = "ansi_default"
        else:
            current_tag = ANSI_TK_TAG_BY_CODE.get(code, "ansi_default")
        start = match.end()

    tail = text[start:]
    if tail:
        widget.insert("end", tail, current_tag)


def resolve_tk_font_size(root: Any, requested_size: float) -> int:
    # Tk font sizes are integer points; emulate fractional requests with tk scaling.
    target = float(requested_size)
    if target <= 0:
        return 1

    rounded = max(1, int(round(target)))
    if abs(target - rounded) < 1e-9:
        return rounded

    try:
        root.tk.call("tk", "scaling", target / float(rounded))
    except Exception:  # noqa: BLE001
        pass
    return rounded


def estimate_topmost_rows(selected_providers: set[str], mini: bool) -> int:
    provider_count = max(1, len(selected_providers))
    rows = (provider_count * 2) + max(0, provider_count - 1)
    if not mini:
        rows += 2
    return rows


def estimate_topmost_height(selected_providers: set[str], mini: bool, font_size: float) -> int:
    rows = estimate_topmost_rows(selected_providers=selected_providers, mini=mini)
    scale = max(1.0, float(font_size)) / 7.5
    base_px = 34.0 * scale
    row_px = 12.0 * scale
    return max(52, int(round(base_px + (rows * row_px))))


def build_default_topmost_geometry(
    selected_providers: set[str],
    mini: bool,
    font_size: float,
    speedometer: bool,
) -> str:
    width = 400 if speedometer else 320
    if DECIMALS:
        width = int(round(width * 1.05))
    if len(selected_providers) == 1:
        width = max(200, int(round(width * SINGLE_PROVIDER_TOPMOST_WIDTH_SCALE)))
    height = estimate_topmost_height(
        selected_providers=selected_providers,
        mini=mini,
        font_size=font_size,
    )
    return f"{width}x{height}+40+40"


def enable_frameless_controls(root: Any, drag_widget: Any) -> None:
    drag_origin: dict[str, int] = {"x": 0, "y": 0}

    def on_press(event: Any) -> None:
        drag_origin["x"] = int(event.x_root)
        drag_origin["y"] = int(event.y_root)

    def on_drag(event: Any) -> None:
        dx = int(event.x_root) - drag_origin["x"]
        dy = int(event.y_root) - drag_origin["y"]
        x = int(root.winfo_x()) + dx
        y = int(root.winfo_y()) + dy
        root.geometry(f"+{x}+{y}")
        drag_origin["x"] = int(event.x_root)
        drag_origin["y"] = int(event.y_root)

    def on_close(_: Any = None) -> str:
        root.destroy()
        return "break"

    drag_widget.bind("<ButtonPress-1>", on_press)
    drag_widget.bind("<B1-Motion>", on_drag)
    root.bind("<Escape>", on_close)
    root.bind("<Command-w>", on_close)
    root.bind("q", on_close)


def apply_frameless_style(root: Any) -> None:
    # Tk on macOS can require multiple calls (before and after mapping) to reliably
    # remove the window chrome.
    try:
        root.overrideredirect(True)
    except Exception:  # noqa: BLE001
        pass

    try:
        root.tk.call(
            "::tk::unsupported::MacWindowStyle",
            "style",
            root._w,
            "help",
            "none",
        )
    except Exception:  # noqa: BLE001
        pass

    try:
        root.update_idletasks()
        root.overrideredirect(True)
    except Exception:  # noqa: BLE001
        pass


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def single_instance_lock(force: bool):
    lock_path = LOCK_PATH
    fallback_path = Path("/tmp/usage-hud.lock")

    for candidate in (lock_path, fallback_path):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            lock_path = candidate
            break
        except OSError:
            continue
    else:
        raise RuntimeError("Could not create lock directory.")

    if lock_path.exists() and not force:
        try:
            existing_pid = int(lock_path.read_text().strip() or "0")
        except ValueError:
            existing_pid = 0
        if pid_is_running(existing_pid):
            raise RuntimeError(
                f"usage-hud already running (pid {existing_pid}). Use --force to replace."
            )
        lock_path.unlink(missing_ok=True)

    lock_path.write_text(str(os.getpid()))
    try:
        yield
    finally:
        try:
            current = int(lock_path.read_text().strip() or "0")
        except (ValueError, OSError):
            current = 0
        if current == os.getpid():
            lock_path.unlink(missing_ok=True)


def enter_alt_screen() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[?1049h\x1b[2J\x1b[H")
        sys.stdout.flush()


def leave_alt_screen() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[?1049l")
        sys.stdout.flush()


def read_keychain_secret() -> str:
    result = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("No Claude Code credentials. Run `claude login`.")
    secret = result.stdout.strip()
    if not secret:
        raise RuntimeError("Claude credentials item is empty.")
    return secret


def claude_token_from_secret(secret: str) -> str:
    try:
        data = json.loads(secret)
    except json.JSONDecodeError:
        return secret

    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    if not isinstance(oauth, dict):
        oauth = data if isinstance(data, dict) else {}

    token = oauth.get("accessToken") or oauth.get("access_token")
    if not token or not isinstance(token, str):
        raise RuntimeError("Could not read Claude access token from Keychain item.")
    return token


def fetch_claude_usage_payload(token: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "anthropic-beta": ANTHROPIC_BETA,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "usage-hud/1.0",
    }
    req = urllib.request.Request(CLAUDE_USAGE_URL, method="GET", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read()
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Claude usage API returned unexpected payload.")
    return payload


def normalize_claude_window(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    window = payload.get(key)
    if not isinstance(window, dict):
        return None
    return {
        "utilization": clamp_pct(window.get("utilization")),
        "resets_at": window.get("resets_at"),
    }


def fetch_claude_snapshot() -> dict[str, Any]:
    token = claude_token_from_secret(read_keychain_secret())
    payload = fetch_claude_usage_payload(token)
    return {
        "five_hour": normalize_claude_window(payload, "five_hour"),
        "seven_day": normalize_claude_window(payload, "seven_day"),
        "seven_day_opus": normalize_claude_window(payload, "seven_day_opus"),
        "updated_at": datetime.now().astimezone().isoformat(),
    }


def parse_codex_window(raw_window: Any) -> CodexRateWindow | None:
    if not isinstance(raw_window, dict):
        return None
    try:
        used_percent = float(raw_window.get("used_percent", 0.0))
    except (TypeError, ValueError):
        used_percent = 0.0
    try:
        window_minutes = int(raw_window.get("window_minutes", 0))
    except (TypeError, ValueError):
        window_minutes = 0
    try:
        resets_at = int(raw_window.get("resets_at", 0))
    except (TypeError, ValueError):
        resets_at = 0
    return CodexRateWindow(
        used_percent=max(0.0, min(100.0, used_percent)),
        window_minutes=window_minutes,
        resets_at=resets_at,
    )


def parse_codex_snapshot(raw_event: Any) -> CodexSnapshot | None:
    if not isinstance(raw_event, dict):
        return None
    if raw_event.get("type") != "event_msg":
        return None

    payload = raw_event.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != "token_count":
        return None

    rate_limits = payload.get("rate_limits")
    if not isinstance(rate_limits, dict):
        return None

    limit_id = rate_limits.get("limit_id")
    if not isinstance(limit_id, str) or not limit_id:
        return None

    info = payload.get("info")
    total_tokens: int | None = None
    model_context_window: int | None = None
    if isinstance(info, dict):
        usage = info.get("total_token_usage")
        if isinstance(usage, dict):
            try:
                total_tokens = int(usage.get("total_tokens", 0))
            except (TypeError, ValueError):
                total_tokens = 0
        try:
            model_context_window = int(info.get("model_context_window", 0))
        except (TypeError, ValueError):
            model_context_window = 0

    return CodexSnapshot(
        timestamp=str(raw_event.get("timestamp", "")),
        limit_id=limit_id,
        primary=parse_codex_window(rate_limits.get("primary")),
        secondary=parse_codex_window(rate_limits.get("secondary")),
        total_tokens=total_tokens,
        model_context_window=model_context_window,
    )


def load_latest_codex_snapshots(sessions_dir: Path) -> dict[str, CodexSnapshot]:
    latest: dict[str, CodexSnapshot] = {}
    if not sessions_dir.exists():
        return latest

    for session_log in sessions_dir.rglob("*.jsonl"):
        try:
            with session_log.open("r", encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    snapshot = parse_codex_snapshot(event)
                    if snapshot is None:
                        continue
                    current = latest.get(snapshot.limit_id)
                    if current is None or snapshot.timestamp > current.timestamp:
                        latest[snapshot.limit_id] = snapshot
        except OSError:
            continue
    return latest


def fetch_codex_snapshot(sessions_dir: Path) -> dict[str, Any]:
    latest = load_latest_codex_snapshots(sessions_dir)
    items: dict[str, Any] = {}
    for limit_id, snapshot in latest.items():
        items[limit_id] = {
            "timestamp": snapshot.timestamp,
            "total_tokens": snapshot.total_tokens,
            "model_context_window": snapshot.model_context_window,
            "primary": None
            if snapshot.primary is None
            else {
                "used_percent": snapshot.primary.used_percent,
                "window_minutes": snapshot.primary.window_minutes,
                "resets_at": snapshot.primary.resets_at,
            },
            "secondary": None
            if snapshot.secondary is None
            else {
                "used_percent": snapshot.secondary.used_percent,
                "window_minutes": snapshot.secondary.window_minutes,
                "resets_at": snapshot.secondary.resets_at,
            },
        }
    return {
        "sources": items,
        "updated_at": datetime.now().astimezone().isoformat(),
    }


def _start_of_local_day(now_local: datetime) -> datetime:
    return now_local.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_local_minute(now_local: datetime) -> datetime:
    return now_local.replace(second=0, microsecond=0)


def _utilization_from_count(used_count: int, limit_count: int) -> int | float | None:
    if limit_count <= 0:
        return None
    pct = (used_count / float(limit_count)) * 100.0
    if not DECIMALS and used_count > 0 and pct < 1.0:
        return 1
    return clamp_pct(pct)


def fetch_gemini_snapshot(
    gemini_tmp_dir: Path,
    minute_limit_requests: int,
    day_limit_requests: int,
) -> dict[str, Any]:
    now_local = datetime.now().astimezone()
    minute_start = _start_of_local_minute(now_local)
    day_start = _start_of_local_day(now_local)
    minute_reset = minute_start + timedelta(minutes=1)
    day_reset = day_start + timedelta(days=1)

    sessions_scanned = 0
    minute_requests = 0
    day_requests = 0
    minute_tokens = 0
    day_tokens = 0
    model_totals: dict[str, int] = {}

    if gemini_tmp_dir.exists():
        for session_file in gemini_tmp_dir.rglob("session-*.json"):
            if session_file.parent.name != "chats":
                continue

            try:
                with session_file.open("r", encoding="utf-8", errors="replace") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            sessions_scanned += 1
            messages = payload.get("messages")
            if not isinstance(messages, list):
                continue

            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("type") != "gemini":
                    continue

                ts = parse_iso(message.get("timestamp"))
                if ts is None:
                    continue

                total_tokens = 0
                tokens = message.get("tokens")
                if isinstance(tokens, dict):
                    try:
                        total_tokens = int(tokens.get("total", 0))
                    except (TypeError, ValueError):
                        total_tokens = 0
                total_tokens = max(0, total_tokens)

                model = message.get("model")
                if not isinstance(model, str) or not model:
                    model = "unknown"
                model_totals[model] = model_totals.get(model, 0) + total_tokens

                if ts >= minute_start:
                    minute_requests += 1
                    minute_tokens += total_tokens
                if ts >= day_start:
                    day_tokens += total_tokens
                    day_requests += 1

    return {
        "minute": {
            "used_requests": minute_requests,
            "limit_requests": max(0, minute_limit_requests),
            "utilization": _utilization_from_count(minute_requests, minute_limit_requests),
            "used_tokens": minute_tokens,
            "resets_at": minute_reset.isoformat(),
        },
        "day": {
            "used_tokens": day_tokens,
            "used_requests": day_requests,
            "limit_requests": max(0, day_limit_requests),
            "utilization": _utilization_from_count(day_requests, day_limit_requests),
            "resets_at": day_reset.isoformat(),
        },
        "models": {
            model: model_totals[model]
            for model in sorted(model_totals)
        },
        "sessions_scanned": sessions_scanned,
        "updated_at": now_local.isoformat(),
    }


def render_window_compact(
    label: str,
    pct: int | float | None,
    expected: int | float | None,
    warn: int,
    critical: int,
    burn_key: tuple[str, str] | None = None,
) -> str:
    speedo = format_speedometer(burn_key)
    if pct is None:
        return f"{ANSI.style(label, 'bold')} --% {ANSI.style('no data', 'yellow')}"

    pct_text = ANSI.style(fmt_pct(pct), "white")
    if expected is None:
        return f"{ANSI.style(label, 'bold')} {pct_text} {pct_bar(pct, 14)}{speedo}"

    delta = pct - expected
    delta_text = ANSI.style(fmt_delta(delta), pace_style(delta))
    target_text = ANSI.style(f"({fmt_pct(expected).strip()})", "dim")
    return (
        f"{ANSI.style(label, 'bold')} {pct_text} {build_pace_bar(pct, expected, 16)} "
        f"{delta_text} {target_text}{speedo}"
    )


def render_claude_mini(
    snapshot: dict[str, Any] | None,
    status_line: str,
    warn: int,
    critical: int,
    show_badge: bool = True,
) -> list[str]:
    first_prefix, second_prefix = provider_badge_lines("claude", show_badge=show_badge)
    cont_prefix = " " * visible_len(first_prefix)

    if not snapshot:
        return [f"{first_prefix}{ANSI.style(status_line, 'yellow')}"]

    now_local = datetime.now().astimezone()
    five = snapshot.get("five_hour") or {}
    seven = snapshot.get("seven_day") or {}
    opus = snapshot.get("seven_day_opus") or {}

    five_pct = clamp_pct(five.get("utilization")) if five else None
    seven_pct = clamp_pct(seven.get("utilization")) if seven else None
    five_expected = expected_pct_from_iso_reset(five.get("resets_at"), timedelta(hours=5), now_local)
    seven_expected = expected_pct_from_iso_reset(
        seven.get("resets_at"), timedelta(days=7), now_local
    )

    key_s = ("claude", "S")
    key_w = ("claude", "W")
    if five_pct is not None:
        BURN_TRACKER.record(key_s, five_pct)
    if seven_pct is not None:
        BURN_TRACKER.record(key_w, seven_pct)

    row_s = render_window_compact(
        "S",
        five_pct,
        five_expected,
        warn,
        critical,
        burn_key=key_s,
    )
    row_w = render_window_compact(
        "W",
        seven_pct,
        seven_expected,
        warn,
        critical,
        burn_key=key_w,
    )

    lines = [f"{first_prefix}{row_s}", f"{second_prefix}{row_w}"]

    if isinstance(opus, dict) and opus.get("utilization") is not None:
        opus_pct = clamp_pct(opus.get("utilization"))
        lines.append(f"{cont_prefix}{ANSI.style('O', 'dim')} {fmt_pct(opus_pct)}")

    if status_line != "Live usage":
        lines.append(f"{cont_prefix}{ANSI.style(status_line, 'yellow')}")

    return lines


def select_codex_sources(sources: dict[str, Any], all_limits: bool) -> list[tuple[str, dict[str, Any]]]:
    if all_limits:
        return [
            (limit_id, item)
            for limit_id, item in sorted(sources.items())
            if isinstance(item, dict)
        ]

    preferred = sources.get("codex")
    if isinstance(preferred, dict):
        return [("codex", preferred)]

    for limit_id in sorted(sources):
        item = sources.get(limit_id)
        if isinstance(item, dict):
            return [(limit_id, item)]

    return []


def render_codex_mini(
    snapshot: dict[str, Any] | None,
    status_line: str,
    warn: int,
    critical: int,
    all_limits: bool,
    show_badge: bool = True,
) -> list[str]:
    first_prefix, second_prefix = provider_badge_lines("codex", show_badge=show_badge)
    cont_prefix = " " * visible_len(first_prefix)

    if not snapshot:
        return [f"{first_prefix}{ANSI.style(status_line, 'yellow')}"]

    sources = snapshot.get("sources")
    if not isinstance(sources, dict) or not sources:
        return [f"{first_prefix}{ANSI.style('no token_count events in ~/.codex/sessions', 'yellow')}"]

    lines: list[str] = []
    now_epoch = time.time()
    selected_sources = select_codex_sources(sources, all_limits)

    for idx, (limit_id, item) in enumerate(selected_sources):
        primary = item.get("primary")
        secondary = item.get("secondary")

        p_pct = clamp_pct(primary.get("used_percent")) if isinstance(primary, dict) else None
        p_window = int(primary.get("window_minutes") or 0) if isinstance(primary, dict) else 0
        p_reset = int(primary.get("resets_at") or 0) if isinstance(primary, dict) else 0
        p_expected = expected_pct_from_epoch_reset(p_reset, p_window, now_epoch)

        s_pct = clamp_pct(secondary.get("used_percent")) if isinstance(secondary, dict) else None
        s_window = int(secondary.get("window_minutes") or 0) if isinstance(secondary, dict) else 0
        s_reset = int(secondary.get("resets_at") or 0) if isinstance(secondary, dict) else 0
        s_expected = expected_pct_from_epoch_reset(s_reset, s_window, now_epoch)

        key_s = ("codex", f"{limit_id}/S")
        key_w = ("codex", f"{limit_id}/W")
        if p_pct is not None:
            BURN_TRACKER.record(key_s, p_pct)
        if s_pct is not None:
            BURN_TRACKER.record(key_w, s_pct)

        row_s = render_window_compact(
            "S",
            p_pct,
            p_expected,
            warn,
            critical,
            burn_key=key_s,
        )
        row_w = render_window_compact(
            "W",
            s_pct,
            s_expected,
            warn,
            critical,
            burn_key=key_w,
        )

        if all_limits:
            label = ANSI.style(limit_id.upper(), "dim")
            if idx == 0:
                lines.append(f"{first_prefix}{label} {row_s}")
                lines.append(f"{second_prefix}{row_w}")
            else:
                lines.append(f"{cont_prefix}{label} {row_s}")
                lines.append(f"{cont_prefix}{row_w}")
        else:
            if idx == 0:
                lines.append(f"{first_prefix}{row_s}")
                lines.append(f"{second_prefix}{row_w}")
            else:
                lines.append(f"{cont_prefix}{row_s}")
                lines.append(f"{cont_prefix}{row_w}")

    if status_line != "Local usage":
        lines.append(f"{cont_prefix}{ANSI.style(status_line, 'yellow')}")

    return lines


def render_gemini_mini(
    snapshot: dict[str, Any] | None,
    status_line: str,
    warn: int,
    critical: int,
    show_badge: bool = True,
) -> list[str]:
    first_prefix, second_prefix = provider_badge_lines("gemini", show_badge=show_badge)
    cont_prefix = " " * visible_len(first_prefix)

    if not snapshot:
        return [f"{first_prefix}{ANSI.style(status_line, 'yellow')}"]

    minute = snapshot.get("minute") if isinstance(snapshot.get("minute"), dict) else {}
    day = snapshot.get("day") if isinstance(snapshot.get("day"), dict) else {}

    minute_pct = (
        clamp_pct(minute.get("utilization"))
        if minute and minute.get("utilization") is not None
        else None
    )
    day_pct = (
        clamp_pct(day.get("utilization"))
        if day and day.get("utilization") is not None
        else None
    )

    key_m = ("gemini", "M")
    key_d = ("gemini", "D")
    if minute_pct is not None:
        BURN_TRACKER.record(key_m, minute_pct)
    if day_pct is not None:
        BURN_TRACKER.record(key_d, day_pct)

    now_local = datetime.now().astimezone()
    minute_expected = (
        expected_pct_from_iso_reset(minute.get("resets_at"), timedelta(minutes=1), now_local)
        if minute_pct is not None
        else None
    )
    day_expected = (
        expected_pct_from_iso_reset(day.get("resets_at"), timedelta(days=1), now_local)
        if day_pct is not None
        else None
    )

    row_s = render_window_compact(
        "M",
        minute_pct,
        minute_expected,
        warn,
        critical,
        burn_key=key_m,
    )
    row_w = render_window_compact(
        "D",
        day_pct,
        day_expected,
        warn,
        critical,
        burn_key=key_d,
    )

    lines = [f"{first_prefix}{row_s}", f"{second_prefix}{row_w}"]

    if status_line != "Local usage":
        lines.append(f"{cont_prefix}{ANSI.style(status_line, 'yellow')}")

    return lines


def parse_provider_selection(raw_value: str) -> set[str]:
    value = str(raw_value or "").strip().lower()
    if value in {"all", "*"}:
        return set(PROVIDER_ORDER)

    providers = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not providers:
        raise ValueError(
            "--providers cannot be empty. Use 'all' or a comma-separated list like 'codex,gemini'."
        )

    invalid = sorted(set(providers) - set(PROVIDER_ORDER))
    if invalid:
        valid = ", ".join(PROVIDER_ORDER)
        bad = ", ".join(invalid)
        raise ValueError(f"Invalid --providers value: {bad}. Valid values: {valid}, all.")

    return set(providers)


def build_provider_sections(
    selected_providers: set[str],
    claude_snapshot: dict[str, Any] | None,
    codex_snapshot: dict[str, Any] | None,
    gemini_snapshot: dict[str, Any] | None,
    claude_status: str,
    codex_status: str,
    gemini_status: str,
    warn: int,
    critical: int,
    all_limits: bool,
    show_badges: bool,
) -> list[list[str]]:
    sections: list[list[str]] = []
    if "claude" in selected_providers:
        sections.append(
            render_claude_mini(
                claude_snapshot,
                claude_status,
                warn,
                critical,
                show_badge=show_badges,
            )
        )
    if "codex" in selected_providers:
        sections.append(
            render_codex_mini(
                codex_snapshot,
                codex_status,
                warn,
                critical,
                all_limits,
                show_badge=show_badges,
            )
        )
    if "gemini" in selected_providers:
        sections.append(
            render_gemini_mini(
                gemini_snapshot,
                gemini_status,
                warn,
                critical,
                show_badge=show_badges,
            )
        )
    return sections


def render_full(
    selected_providers: set[str],
    claude_snapshot: dict[str, Any] | None,
    codex_snapshot: dict[str, Any] | None,
    gemini_snapshot: dict[str, Any] | None,
    claude_status: str,
    codex_status: str,
    gemini_status: str,
    warn: int,
    critical: int,
    all_limits: bool,
) -> str:
    now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: list[str] = [ANSI.style(f"UNIFIED USAGE HUD  ({now_local})", "bold")]
    show_badges = len(selected_providers) > 1
    sections = build_provider_sections(
        selected_providers=selected_providers,
        claude_snapshot=claude_snapshot,
        codex_snapshot=codex_snapshot,
        gemini_snapshot=gemini_snapshot,
        claude_status=claude_status,
        codex_status=codex_status,
        gemini_status=gemini_status,
        warn=warn,
        critical=critical,
        all_limits=all_limits,
        show_badges=show_badges,
    )
    if sections:
        lines.append("")
    for idx, section in enumerate(sections):
        lines.extend(section)
        if idx < len(sections) - 1:
            lines.append("")

    return "\n".join(lines)


def render_mini(
    selected_providers: set[str],
    claude_snapshot: dict[str, Any] | None,
    codex_snapshot: dict[str, Any] | None,
    gemini_snapshot: dict[str, Any] | None,
    claude_status: str,
    codex_status: str,
    gemini_status: str,
    warn: int,
    critical: int,
    all_limits: bool,
) -> str:
    lines: list[str] = []
    show_badges = len(selected_providers) > 1
    sections = build_provider_sections(
        selected_providers=selected_providers,
        claude_snapshot=claude_snapshot,
        codex_snapshot=codex_snapshot,
        gemini_snapshot=gemini_snapshot,
        claude_status=claude_status,
        codex_status=codex_status,
        gemini_status=gemini_status,
        warn=warn,
        critical=critical,
        all_limits=all_limits,
        show_badges=show_badges,
    )
    for idx, section in enumerate(sections):
        lines.extend(section)
        if idx < len(sections) - 1:
            lines.append("")
    return "\n".join(lines)


def fetch_all_snapshots(
    selected_providers: set[str],
    codex_sessions_dir: Path,
    gemini_tmp_dir: Path,
    gemini_minute_limit_requests: int,
    gemini_day_limit_requests: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, str, str, str]:
    claude_snapshot: dict[str, Any] | None = None
    codex_snapshot: dict[str, Any] | None = None
    gemini_snapshot: dict[str, Any] | None = None
    claude_status = "Disabled by --providers"
    codex_status = "Disabled by --providers"
    gemini_status = "Disabled by --providers"

    if "claude" in selected_providers:
        claude_status = "Initializing"
        try:
            claude_snapshot = fetch_claude_snapshot()
            claude_status = "Live usage"
        except urllib.error.HTTPError as exc:
            claude_status = f"HTTP {exc.code} from Claude usage API"
        except urllib.error.URLError:
            claude_status = "Network error reaching Claude usage API"
        except Exception as exc:  # noqa: BLE001
            claude_status = str(exc)

    if "codex" in selected_providers:
        codex_status = "Initializing"
        try:
            codex_snapshot = fetch_codex_snapshot(codex_sessions_dir.expanduser())
            codex_status = "Local usage"
        except Exception as exc:  # noqa: BLE001
            codex_status = str(exc)

    if "gemini" in selected_providers:
        gemini_status = "Initializing"
        try:
            gemini_snapshot = fetch_gemini_snapshot(
                gemini_tmp_dir.expanduser(),
                gemini_minute_limit_requests,
                gemini_day_limit_requests,
            )
            gemini_status = "Local usage"
        except Exception as exc:  # noqa: BLE001
            gemini_status = str(exc)

    return (
        claude_snapshot,
        codex_snapshot,
        gemini_snapshot,
        claude_status,
        codex_status,
        gemini_status,
    )


def render_output(
    selected_providers: set[str],
    mini: bool,
    all_limits: bool,
    warn: int,
    critical: int,
    claude_snapshot: dict[str, Any] | None,
    codex_snapshot: dict[str, Any] | None,
    gemini_snapshot: dict[str, Any] | None,
    claude_status: str,
    codex_status: str,
    gemini_status: str,
) -> str:
    if mini:
        return render_mini(
            selected_providers=selected_providers,
            claude_snapshot=claude_snapshot,
            codex_snapshot=codex_snapshot,
            gemini_snapshot=gemini_snapshot,
            claude_status=claude_status,
            codex_status=codex_status,
            gemini_status=gemini_status,
            warn=warn,
            critical=critical,
            all_limits=all_limits,
        )

    return render_full(
        selected_providers=selected_providers,
        claude_snapshot=claude_snapshot,
        codex_snapshot=codex_snapshot,
        gemini_snapshot=gemini_snapshot,
        claude_status=claude_status,
        codex_status=codex_status,
        gemini_status=gemini_status,
        warn=warn,
        critical=critical,
        all_limits=all_limits,
    )


def run_topmost_window(args: argparse.Namespace, selected_providers: set[str]) -> int:
    try:
        import tkinter as tk
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") == "_tkinter":
            py_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
            print(
                (
                    "Could not initialize topmost HUD window: missing Tk bindings for this Python.\n"
                    f"Install them with: brew install python-tk@{py_tag}\n"
                    "Then retry in a new shell."
                ),
                file=sys.stderr,
            )
            return 1
        print(f"Could not initialize topmost HUD window: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Could not initialize topmost HUD window: {exc}", file=sys.stderr)
        return 1

    try:
        bg_color = "#111315"
        fg_color = "#e6edf3"
        if args.no_color:
            bg_color = "#ffffff"
            fg_color = "#111111"

        root = tk.Tk()
        root.title("usage-hud (always-on-top)")
        root.configure(background=bg_color)
        root.attributes("-topmost", True)
        if args.always_on_top_frameless:
            apply_frameless_style(root)
        root.geometry(args.always_on_top_geometry)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not create topmost HUD window: {exc}", file=sys.stderr)
        return 1

    tk_font_size = resolve_tk_font_size(root, args.always_on_top_font_size)

    text = tk.Text(
        root,
        wrap="none",
        background=bg_color,
        foreground=fg_color,
        insertbackground=fg_color,
        borderwidth=0,
        highlightthickness=0,
        padx=12,
        pady=10,
        font=("Menlo", tk_font_size),
    )
    text.pack(fill="both", expand=True)
    text.configure(state="disabled")
    if args.always_on_top_frameless:
        enable_frameless_controls(root=root, drag_widget=text)

    interval_ms = max(1, int(args.interval * 1000))

    def refresh() -> None:
        (
            claude_snapshot,
            codex_snapshot,
            gemini_snapshot,
            claude_status,
            codex_status,
            gemini_status,
        ) = fetch_all_snapshots(
            selected_providers=selected_providers,
            codex_sessions_dir=args.codex_sessions_dir,
            gemini_tmp_dir=args.gemini_tmp_dir,
            gemini_minute_limit_requests=args.gemini_minute_limit_requests,
            gemini_day_limit_requests=args.gemini_day_limit_requests,
        )

        output = render_output(
            selected_providers=selected_providers,
            mini=args.mini,
            all_limits=args.all_limits,
            warn=args.warn_threshold,
            critical=args.critical_threshold,
            claude_snapshot=claude_snapshot,
            codex_snapshot=codex_snapshot,
            gemini_snapshot=gemini_snapshot,
            claude_status=claude_status,
            codex_status=codex_status,
            gemini_status=gemini_status,
        )

        text.configure(state="normal")
        text.delete("1.0", "end")
        if args.no_color:
            text.insert("1.0", ANSI_ESCAPE_RE.sub("", output))
        else:
            insert_ansi_text_tk(
                widget=text,
                text=output,
                font_size=tk_font_size,
                default_fg=fg_color,
            )
        text.configure(state="disabled")

        root.after(interval_ms, refresh)

    refresh()
    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Claude Code, Codex, and Gemini usage in one HUD.")
    mac_default_topmost = sys.platform == "darwin"
    mac_default_topmost_label = "enabled on macOS, disabled elsewhere"
    mac_default_frameless_label = "enabled on macOS, disabled elsewhere"
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
        "--once",
        action="store_true",
        help="Render once and exit",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Compact provider rows",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing HUD instance lock",
    )
    parser.add_argument(
        "--no-alt-screen",
        action="store_true",
        help="Disable alternate screen mode",
    )
    parser.add_argument(
        "--warn-threshold",
        type=int,
        default=80,
        help="Warn threshold percentage for windows (default: 80)",
    )
    parser.add_argument(
        "--critical-threshold",
        type=int,
        default=95,
        help="Critical threshold percentage for windows (default: 95)",
    )
    parser.add_argument(
        "--all-limits",
        action="store_true",
        help="Show all Codex limit buckets (default: show primary codex bucket only)",
    )
    parser.add_argument(
        "--providers",
        default="all",
        help="Comma-separated providers to display: claude,codex,gemini or all (default: all)",
    )
    parser.add_argument(
        "--bar-style",
        choices=["auto", "legacy", "solid"],
        default="auto",
        help="Bar rendering style (default: auto; solid in topmost mode, legacy in terminal mode)",
    )
    parser.add_argument(
        "--always-on-top",
        dest="always_on_top",
        action="store_true",
        default=mac_default_topmost,
        help=(
            "Render HUD in a small always-on-top window "
            f"(default: {mac_default_topmost_label})"
        ),
    )
    parser.add_argument(
        "--no-always-on-top",
        dest="always_on_top",
        action="store_false",
        help="Disable always-on-top window and render in the terminal",
    )
    parser.add_argument(
        "--always-on-top-font-size",
        type=float,
        default=7.5,
        help="Font size for --always-on-top window (default: 7.5)",
    )
    parser.add_argument(
        "--always-on-top-geometry",
        default=DEFAULT_TOPMOST_GEOMETRY,
        help=(
            "Initial geometry for --always-on-top, e.g. 320x130+40+40 "
            f"(default: {DEFAULT_TOPMOST_GEOMETRY}; size auto-scales for --providers when unchanged)"
        ),
    )
    parser.add_argument(
        "--always-on-top-frameless",
        dest="always_on_top_frameless",
        action="store_true",
        default=mac_default_topmost,
        help=(
            "Hide title bar in --always-on-top mode "
            f"(default: {mac_default_frameless_label}; drag to move, Esc/Cmd+W/q to close)"
        ),
    )
    parser.add_argument(
        "--always-on-top-framed",
        dest="always_on_top_frameless",
        action="store_false",
        help="Show title bar in --always-on-top mode",
    )
    parser.add_argument(
        "--gemini-tmp-dir",
        type=Path,
        default=Path.home() / ".gemini" / "tmp",
        help="Gemini CLI tmp directory (default: ~/.gemini/tmp)",
    )
    parser.add_argument(
        "--gemini-minute-limit-requests",
        type=int,
        default=120,
        help="Gemini request limit per minute for mini bars (default: 120)",
    )
    parser.add_argument(
        "--gemini-day-limit-requests",
        type=int,
        default=1500,
        help="Gemini request limit per day for mini bars (default: 1500)",
    )

    parser.add_argument(
        "--speedometer",
        action="store_true",
        help="Show burn-rate and ETA-to-throttle suffix on each window line",
    )
    parser.add_argument(
        "--decimals",
        action="store_true",
        help="Show one decimal place on percentages (e.g. 12.2%% instead of 12%%)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    return parser.parse_args()


def main() -> int:
    global ANSI
    global BAR_STYLE
    global SPEEDOMETER_ENABLED
    global DECIMALS

    args = parse_args()

    try:
        selected_providers = parse_provider_selection(args.providers)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.bar_style == "auto":
        BAR_STYLE = "solid" if args.always_on_top else "legacy"
    else:
        BAR_STYLE = args.bar_style

    if args.no_color:
        ANSI = Ansi(False)

    if args.speedometer:
        SPEEDOMETER_ENABLED = True

    if args.decimals:
        DECIMALS = True

    if args.interval is None:
        args.interval = 30.0 if args.decimals else 60.0

    if args.interval <= 0:
        print("--interval must be > 0", file=sys.stderr)
        return 2
    if args.critical_threshold < args.warn_threshold:
        print("--critical-threshold must be >= --warn-threshold", file=sys.stderr)
        return 2
    if args.gemini_minute_limit_requests < 0:
        print("--gemini-minute-limit-requests must be >= 0", file=sys.stderr)
        return 2
    if args.gemini_day_limit_requests < 0:
        print("--gemini-day-limit-requests must be >= 0", file=sys.stderr)
        return 2
    if args.always_on_top and sys.platform != "darwin":
        print("--always-on-top is currently supported on macOS only", file=sys.stderr)
        return 2
    if args.always_on_top and args.json:
        print("--always-on-top cannot be combined with --json", file=sys.stderr)
        return 2
    if args.always_on_top and args.once:
        print("--always-on-top cannot be combined with --once", file=sys.stderr)
        return 2
    if args.always_on_top and args.always_on_top_geometry == DEFAULT_TOPMOST_GEOMETRY:
        args.always_on_top_geometry = build_default_topmost_geometry(
            selected_providers=selected_providers,
            mini=args.mini,
            font_size=args.always_on_top_font_size,
            speedometer=args.speedometer,
        )

    if args.always_on_top_font_size <= 0:
        print("--always-on-top-font-size must be > 0", file=sys.stderr)
        return 2
    if not isinstance(args.always_on_top_geometry, str) or not args.always_on_top_geometry.strip():
        print("--always-on-top-geometry must be a non-empty geometry string", file=sys.stderr)
        return 2

    run_once = args.once or args.json
    use_alt_screen = (
        not run_once
        and sys.stdout.isatty()
        and not args.no_alt_screen
        and not args.json
        and not args.always_on_top
    )

    try:
        with single_instance_lock(force=args.force):
            if args.always_on_top:
                return run_topmost_window(args, selected_providers)

            if use_alt_screen:
                enter_alt_screen()

            while True:
                (
                    claude_snapshot,
                    codex_snapshot,
                    gemini_snapshot,
                    claude_status,
                    codex_status,
                    gemini_status,
                ) = fetch_all_snapshots(
                    selected_providers=selected_providers,
                    codex_sessions_dir=args.codex_sessions_dir,
                    gemini_tmp_dir=args.gemini_tmp_dir,
                    gemini_minute_limit_requests=args.gemini_minute_limit_requests,
                    gemini_day_limit_requests=args.gemini_day_limit_requests,
                )

                if args.json:
                    payload = {
                        "generated_at": datetime.now().astimezone().isoformat(),
                        "claude": {
                            "status": claude_status,
                            "snapshot": claude_snapshot,
                        },
                        "codex": {
                            "status": codex_status,
                            "snapshot": codex_snapshot,
                        },
                        "gemini": {
                            "status": gemini_status,
                            "snapshot": gemini_snapshot,
                        },
                    }
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    if sys.stdout.isatty():
                        sys.stdout.write("\x1b[2J\x1b[H")
                    print(
                        render_output(
                            selected_providers=selected_providers,
                            mini=args.mini,
                            all_limits=args.all_limits,
                            warn=args.warn_threshold,
                            critical=args.critical_threshold,
                            claude_snapshot=claude_snapshot,
                            codex_snapshot=codex_snapshot,
                            gemini_snapshot=gemini_snapshot,
                            claude_status=claude_status,
                            codex_status=codex_status,
                            gemini_status=gemini_status,
                        )
                    )
                    sys.stdout.flush()

                if run_once:
                    return 0
                time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if use_alt_screen:
            leave_alt_screen()


if __name__ == "__main__":
    sys.exit(main())
