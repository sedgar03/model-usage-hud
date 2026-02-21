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


def _default_lock_path() -> Path:
    env_path = os.environ.get("USAGE_HUD_LOCK_PATH")
    if env_path:
        return Path(env_path)
    return Path.home() / ".usage-hud" / "usage-hud.lock"


LOCK_PATH = _default_lock_path()


def clamp_pct(value: Any) -> int:
    try:
        pct = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(pct, 100))


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


def expected_pct_from_iso_reset(resets_at: str | None, window: timedelta, now_local: datetime) -> int | None:
    reset_dt = parse_iso(resets_at)
    if reset_dt is None:
        return None
    now = now_local.astimezone(reset_dt.tzinfo)
    start_dt = reset_dt - window
    elapsed = (now - start_dt).total_seconds() / window.total_seconds()
    return clamp_pct(elapsed * 100.0)


def expected_pct_from_epoch_reset(resets_at: int, window_minutes: int, now_epoch: float) -> int | None:
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
        return "bold_green"
    if delta <= -6:
        return "green"
    return "cyan"


def pct_bar(pct: int, width: int = 16) -> str:
    pct = max(0, min(pct, 100))
    filled = int(round((pct / 100.0) * width))
    fill = "▓" * filled
    empty = "░" * (width - filled)
    return ANSI.style(fill, usage_style(pct)) + ANSI.style(empty, "dim")


def build_pace_bar(actual_pct: int, expected_pct: int, width: int = 24) -> str:
    actual_units = int(round((max(0, min(actual_pct, 100)) / 100.0) * width))
    expected_units = int(round((max(0, min(expected_pct, 100)) / 100.0) * width))
    marker_idx = max(0, min(width - 1, expected_units - 1 if expected_units > 0 else 0))

    pieces: list[str] = []
    for i in range(width):
        pos = i + 1
        if i == marker_idx:
            marker_style = "bold_red" if actual_units > expected_units else "bold_green"
            if abs(actual_units - expected_units) <= 1:
                marker_style = "bold_white"
            pieces.append(ANSI.style("┊", marker_style))
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


def provider_badge_lines(provider: str) -> tuple[str, str]:
    if provider == "claude":
        top = ANSI.style("/CC\\", "brown")
        bottom = ANSI.style("\\CC/", "brown")
    elif provider == "codex":
        top = ANSI.style("/OA\\", "white")
        bottom = ANSI.style("\\OA/", "white")
    else:
        top = ANSI.style("/GM\\", "cyan")
        bottom = ANSI.style("\\GM/", "cyan")

    return f"{top} ", f"{bottom} "

def visible_len(text: str) -> int:
    return len(ANSI_ESCAPE_RE.sub("", text))


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


def _utilization_from_count(used_count: int, limit_count: int) -> int | None:
    if limit_count <= 0:
        return None
    pct = (used_count / float(limit_count)) * 100.0
    if used_count > 0 and pct < 1.0:
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
    pct: int | None,
    expected: int | None,
    warn: int,
    critical: int,
) -> str:
    if pct is None:
        return f"{ANSI.style(label, 'bold')} --% {ANSI.style('no data', 'yellow')}"

    pct_text = ANSI.style(f"{pct:>3}%", "bold")
    if expected is None:
        return f"{ANSI.style(label, 'bold')} {pct_text} {pct_bar(pct, 14)}"

    delta = pct - expected
    delta_text = ANSI.style(f"{delta:+d}", pace_style(delta))
    target_text = ANSI.style(f"({expected}%)", "dim")
    return (
        f"{ANSI.style(label, 'bold')} {pct_text} {build_pace_bar(pct, expected, 16)} "
        f"{delta_text} {target_text}"
    )


def render_claude_mini(
    snapshot: dict[str, Any] | None,
    status_line: str,
    warn: int,
    critical: int,
) -> list[str]:
    first_prefix, second_prefix = provider_badge_lines("claude")
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

    row_s = render_window_compact(
        "S",
        five_pct,
        five_expected,
        warn,
        critical,
    )
    row_w = render_window_compact(
        "W",
        seven_pct,
        seven_expected,
        warn,
        critical,
    )

    lines = [f"{first_prefix}{row_s}", f"{second_prefix}{row_w}"]

    if isinstance(opus, dict) and opus.get("utilization") is not None:
        opus_pct = clamp_pct(opus.get("utilization"))
        lines.append(f"{cont_prefix}{ANSI.style('O', 'dim')} {opus_pct:>3}%")

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
) -> list[str]:
    first_prefix, second_prefix = provider_badge_lines("codex")
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

        row_s = render_window_compact(
            "S",
            p_pct,
            p_expected,
            warn,
            critical,
        )
        row_w = render_window_compact(
            "W",
            s_pct,
            s_expected,
            warn,
            critical,
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
) -> list[str]:
    first_prefix, second_prefix = provider_badge_lines("gemini")
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
    )
    row_w = render_window_compact(
        "D",
        day_pct,
        day_expected,
        warn,
        critical,
    )

    lines = [f"{first_prefix}{row_s}", f"{second_prefix}{row_w}"]

    if status_line != "Local usage":
        lines.append(f"{cont_prefix}{ANSI.style(status_line, 'yellow')}")

    return lines


def render_full(
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
    lines: list[str] = [ANSI.style(f"UNIFIED USAGE HUD  ({now_local})", "bold"), ""]
    lines.extend(render_claude_mini(claude_snapshot, claude_status, warn, critical))
    lines.append("")
    lines.extend(render_codex_mini(codex_snapshot, codex_status, warn, critical, all_limits))
    lines.append("")
    lines.extend(render_gemini_mini(gemini_snapshot, gemini_status, warn, critical))

    return "\n".join(lines)


def render_mini(
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
    lines.extend(render_claude_mini(claude_snapshot, claude_status, warn, critical))
    lines.append("")
    lines.extend(render_codex_mini(codex_snapshot, codex_status, warn, critical, all_limits))
    lines.append("")
    lines.extend(render_gemini_mini(gemini_snapshot, gemini_status, warn, critical))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Claude Code, Codex, and Gemini usage in one HUD.")
    parser.add_argument(
        "--codex-sessions-dir",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
        help="Codex sessions directory (default: ~/.codex/sessions)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Refresh interval in seconds (default: 30)",
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
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    return parser.parse_args()


def main() -> int:
    global ANSI

    args = parse_args()

    if args.no_color:
        ANSI = Ansi(False)

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

    run_once = args.once or args.json
    use_alt_screen = (
        not run_once and sys.stdout.isatty() and not args.no_alt_screen and not args.json
    )

    claude_snapshot: dict[str, Any] | None = None
    codex_snapshot: dict[str, Any] | None = None
    gemini_snapshot: dict[str, Any] | None = None
    claude_status = "Initializing"
    codex_status = "Initializing"
    gemini_status = "Initializing"

    try:
        with single_instance_lock(force=args.force):
            if use_alt_screen:
                enter_alt_screen()

            while True:
                try:
                    claude_snapshot = fetch_claude_snapshot()
                    claude_status = "Live usage"
                except urllib.error.HTTPError as exc:
                    claude_status = f"HTTP {exc.code} from Claude usage API"
                except urllib.error.URLError:
                    claude_status = "Network error reaching Claude usage API"
                except Exception as exc:  # noqa: BLE001
                    claude_status = str(exc)

                try:
                    codex_snapshot = fetch_codex_snapshot(args.codex_sessions_dir.expanduser())
                    codex_status = "Local usage"
                except Exception as exc:  # noqa: BLE001
                    codex_status = str(exc)
                try:
                    gemini_snapshot = fetch_gemini_snapshot(
                        args.gemini_tmp_dir.expanduser(),
                        args.gemini_minute_limit_requests,
                        args.gemini_day_limit_requests,
                    )
                    gemini_status = "Local usage"
                except Exception as exc:  # noqa: BLE001
                    gemini_status = str(exc)

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
                    if args.mini:
                        print(
                            render_mini(
                                claude_snapshot=claude_snapshot,
                                codex_snapshot=codex_snapshot,
                                gemini_snapshot=gemini_snapshot,
                                claude_status=claude_status,
                                codex_status=codex_status,
                                gemini_status=gemini_status,
                                warn=args.warn_threshold,
                                critical=args.critical_threshold,
                                all_limits=args.all_limits,
                            )
                        )
                    else:
                        print(
                            render_full(
                                claude_snapshot=claude_snapshot,
                                codex_snapshot=codex_snapshot,
                                gemini_snapshot=gemini_snapshot,
                                claude_status=claude_status,
                                codex_status=codex_status,
                                gemini_status=gemini_status,
                                warn=args.warn_threshold,
                                critical=args.critical_threshold,
                                all_limits=args.all_limits,
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
