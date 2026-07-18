"""Local machine metrics for the ``system`` provider.

This module reads CPU, memory, memory-pressure, swap, and disk usage from the
host and returns a plain ``dict`` snapshot shaped like the other providers'
snapshots so it flows through the same view-model pipeline.

Design constraints
------------------
- **No third-party dependencies.** The rest of the HUD is stdlib-only by
  design (``urllib`` instead of ``requests``, local logs instead of API keys),
  so this stays that way. CPU ticks come from the Mach ``host_statistics``
  syscall via ``ctypes``; everything else is ``sysctl`` / ``vm_stat`` /
  ``shutil``.
- **macOS-first, degrade gracefully.** The target is an Apple-silicon Mac used
  as a tailnet server. On other platforms the disk and load-average fields
  still populate (they use ``shutil`` / ``os.getloadavg``); the macOS-only
  fields (memory pressure, swap, precise memory breakdown) come back ``None``.
- **Never raise.** A metric that can't be read is ``None`` in the snapshot, not
  an exception — a monitoring surface that crashes is worse than one with a
  gap.

Numbers, and how much to trust them
-----------------------------------
- **CPU busy %** is exact: a delta of kernel tick counters (user+system+nice
  vs. idle) sampled ``CPU_SAMPLE_SECONDS`` apart. This is the real utilization,
  not an estimate.
- **Load average** is reported as-is (1/5/15 min) plus a per-core normalization
  so a value >100% means the run queue is deeper than the core count — the
  classic "redlining" signal.
- **Memory used %** approximates Activity Monitor's "Memory Used"
  (``active + wired + compressor-occupied`` pages). It is an approximation; the
  *authoritative* pressure signal is ``kern.memorystatus_vm_pressure_level``,
  reported separately as ``pressure``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any

# Timeout for pulling a peer's snapshot over the tailnet. Kept short so a
# sleeping/offline server degrades to an "unreachable" note within a few
# seconds instead of stalling the whole HUD refresh.
REMOTE_TIMEOUT_SECONDS = 4.0

# Seconds between the two CPU-tick samples used to compute busy %. Short enough
# to be imperceptible on a 30-60s HUD refresh, long enough to be a stable read.
CPU_SAMPLE_SECONDS = 0.12

# ``kern.memorystatus_vm_pressure_level`` values. The kernel reports a bitmask-
# style level: 1 = normal, 2 = warning, 4 = critical/urgent.
_PRESSURE_NORMAL = 1
_PRESSURE_WARN = 2
_PRESSURE_CRITICAL = 4

# The disk whose free space actually matters. On modern macOS ``/`` is the
# sealed read-only system snapshot; user data lives on the Data volume, but both
# share one APFS container so ``shutil.disk_usage`` on either returns the shared
# container totals. ``/`` is the safe cross-platform default.
DEFAULT_DISK_PATH = "/"

# RAM is conventionally shown in binary GB (a "32 GB" Mac reports
# hw.memsize = 32 * 1024^3), while macOS shows storage in decimal GB (Finder's
# "995 GB"). Use each convention for its own metric so the numbers match what
# the user sees elsewhere on the machine.
_BYTES_PER_GIB = 1024 ** 3
_BYTES_PER_GB = 1000 ** 3

_IS_DARWIN = sys.platform == "darwin"


# ---------------------------------------------------------------------------
# CPU — Mach host_statistics(HOST_CPU_LOAD_INFO)
# ---------------------------------------------------------------------------

_HOST_CPU_LOAD_INFO = 3
_HOST_CPU_LOAD_INFO_COUNT = 4  # [user, system, idle, nice]


class _HostCpuLoadInfo(ctypes.Structure):
    _fields_ = [("cpu_ticks", ctypes.c_uint * _HOST_CPU_LOAD_INFO_COUNT)]


def _read_cpu_ticks() -> tuple[int, int] | None:
    """Return cumulative ``(busy_ticks, total_ticks)`` since boot, or ``None``.

    ``busy = user + system + nice``; ``total = busy + idle``. Both counters are
    monotonic, so two reads bracket an interval and their difference gives the
    busy fraction over that interval.
    """

    if not _IS_DARWIN:
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.mach_host_self.restype = ctypes.c_uint
        info = _HostCpuLoadInfo()
        count = ctypes.c_uint(_HOST_CPU_LOAD_INFO_COUNT)
        result = libc.host_statistics(
            libc.mach_host_self(),
            _HOST_CPU_LOAD_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if result != 0:
            return None
        user, system, idle, nice = list(info.cpu_ticks)
        busy = user + system + nice
        total = busy + idle
        if total <= 0:
            return None
        return busy, total
    except Exception:
        return None


def _cpu_busy_pct() -> float | None:
    """Instantaneous CPU busy percentage over a short sampling window.

    Stateless: samples ticks, sleeps ``CPU_SAMPLE_SECONDS``, samples again. The
    short block runs inside the refresh (background thread in the app, once per
    interval in the CLI) so it never touches interactive latency.
    """

    first = _read_cpu_ticks()
    if first is None:
        return None
    time.sleep(CPU_SAMPLE_SECONDS)
    second = _read_cpu_ticks()
    if second is None:
        return None
    busy_delta = second[0] - first[0]
    total_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, busy_delta / total_delta * 100.0))


# ---------------------------------------------------------------------------
# sysctl / vm_stat helpers
# ---------------------------------------------------------------------------


def _sysctl(name: str) -> str | None:
    if not _IS_DARWIN:
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return value or None


def _cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _load_average() -> tuple[float, float, float] | None:
    try:
        one, five, fifteen = os.getloadavg()
        return (float(one), float(five), float(fifteen))
    except (OSError, AttributeError, ValueError):
        return None


def _pressure_label(level: int | None) -> str | None:
    if level is None:
        return None
    if level >= _PRESSURE_CRITICAL:
        return "critical"
    if level >= _PRESSURE_WARN:
        return "warning"
    return "normal"


def _memory_pressure_level() -> int | None:
    raw = _sysctl("kern.memorystatus_vm_pressure_level")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


_VM_STAT_LINE = re.compile(r'^"?([^":]+)"?:\s+(\d+)\.?\s*$')
_VM_STAT_PAGESIZE = re.compile(r"page size of (\d+) bytes")


def _parse_vm_stat(text: str) -> tuple[int, dict[str, int]] | None:
    """Parse ``vm_stat`` output into ``(page_size_bytes, {label: pages})``.

    Labels are lower-cased and stripped (e.g. ``"Pages wired down"`` ->
    ``"pages wired down"``). Returns ``None`` if the page size can't be found.
    """

    lines = text.splitlines()
    if not lines:
        return None
    header = _VM_STAT_PAGESIZE.search(lines[0])
    if not header:
        return None
    page_size = int(header.group(1))
    counts: dict[str, int] = {}
    for line in lines[1:]:
        match = _VM_STAT_LINE.match(line.strip())
        if match:
            counts[match.group(1).strip().lower()] = int(match.group(2))
    return page_size, counts


def _memory_snapshot() -> dict[str, Any]:
    """Physical memory usage, breakdown, and macOS memory pressure."""

    total_raw = _sysctl("hw.memsize")
    total_bytes = None
    if total_raw is not None:
        try:
            total_bytes = int(total_raw)
        except ValueError:
            total_bytes = None

    level = _memory_pressure_level()
    result: dict[str, Any] = {
        "total_gb": round(total_bytes / _BYTES_PER_GIB, 2) if total_bytes else None,
        "used_gb": None,
        "available_gb": None,
        "used_pct": None,
        "pressure": _pressure_label(level),
        "pressure_level": level,
    }

    if not _IS_DARWIN or total_bytes is None:
        return result

    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2.0
        )
    except (OSError, subprocess.SubprocessError):
        return result
    if out.returncode != 0:
        return result

    parsed = _parse_vm_stat(out.stdout)
    if parsed is None:
        return result
    page_size, counts = parsed

    active = counts.get("pages active", 0)
    wired = counts.get("pages wired down", 0)
    compressed = counts.get("pages occupied by compressor", 0)

    # "Used" ~= Activity Monitor's Memory Used: app-resident + wired + the
    # compressor's physical footprint. Free/reclaimable = everything else
    # (free pages, file-cache-heavy inactive, speculative, purgeable).
    used_pages = active + wired + compressed
    used_bytes = used_pages * page_size
    available_bytes = max(0, total_bytes - used_bytes)

    result["used_gb"] = round(used_bytes / _BYTES_PER_GIB, 2)
    result["available_gb"] = round(available_bytes / _BYTES_PER_GIB, 2)
    result["used_pct"] = round(used_bytes / total_bytes * 100.0, 1)
    return result


_SWAP_RE = re.compile(
    r"total\s*=\s*([\d.]+)([KMG])\s+used\s*=\s*([\d.]+)([KMG])\s+free\s*=\s*([\d.]+)([KMG])"
)
_UNIT_TO_BYTES = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}


def _swap_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "total_gb": None,
        "used_gb": None,
        "free_gb": None,
        "used_pct": None,
    }
    raw = _sysctl("vm.swapusage")
    if raw is None:
        return result
    match = _SWAP_RE.search(raw)
    if not match:
        return result
    total = float(match.group(1)) * _UNIT_TO_BYTES[match.group(2)]
    used = float(match.group(3)) * _UNIT_TO_BYTES[match.group(4)]
    free = float(match.group(5)) * _UNIT_TO_BYTES[match.group(6)]
    result["total_gb"] = round(total / _BYTES_PER_GIB, 2)
    result["used_gb"] = round(used / _BYTES_PER_GIB, 2)
    result["free_gb"] = round(free / _BYTES_PER_GIB, 2)
    result["used_pct"] = round(used / total * 100.0, 1) if total > 0 else None
    return result


def _disk_snapshot(path: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path,
        "total_gb": None,
        "used_gb": None,
        "free_gb": None,
        "used_pct": None,
    }
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return result
    result["total_gb"] = round(usage.total / _BYTES_PER_GB, 2)
    result["used_gb"] = round(usage.used / _BYTES_PER_GB, 2)
    result["free_gb"] = round(usage.free / _BYTES_PER_GB, 2)
    result["used_pct"] = (
        round(usage.used / usage.total * 100.0, 1) if usage.total > 0 else None
    )
    return result


def _cpu_snapshot() -> dict[str, Any]:
    ncpu = _cpu_count()
    busy = _cpu_busy_pct()
    load = _load_average()
    load1 = load[0] if load else None

    # Fall back to normalized load if the tick sample is unavailable (e.g. the
    # Mach call failed) so the gauge still has a value to draw.
    used_pct = busy
    if used_pct is None and load1 is not None:
        used_pct = max(0.0, min(100.0, load1 / ncpu * 100.0))

    return {
        "used_pct": round(used_pct, 1) if used_pct is not None else None,
        "ncpu": ncpu,
        "load1": round(load1, 2) if load1 is not None else None,
        "load5": round(load[1], 2) if load else None,
        "load15": round(load[2], 2) if load else None,
        # Per-core load as a percentage; >100 means run queue deeper than cores.
        "load_pct": round(load1 / ncpu * 100.0, 1) if load1 is not None else None,
    }


def collect_system_snapshot(disk_path: str = DEFAULT_DISK_PATH) -> dict[str, Any]:
    """Return a snapshot of local machine health.

    Shape (any field may be ``None`` when unavailable)::

        {
          "cpu":    {"used_pct", "ncpu", "load1", "load5", "load15", "load_pct"},
          "memory": {"used_pct", "total_gb", "used_gb", "available_gb",
                     "pressure", "pressure_level"},
          "swap":   {"used_pct", "total_gb", "used_gb", "free_gb"},
          "disk":   {"used_pct", "total_gb", "used_gb", "free_gb", "path"},
          "platform": "darwin" | ...,
        }
    """

    return {
        "cpu": _cpu_snapshot(),
        "memory": _memory_snapshot(),
        "swap": _swap_snapshot(),
        "disk": _disk_snapshot(disk_path),
        "platform": sys.platform,
    }


def fetch_remote_system_snapshot(
    base_url: str, timeout: float = REMOTE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Pull another machine's snapshot from its ``usage-hud-serve`` endpoint.

    ``base_url`` is the server root (e.g. ``http://stevens-mac-studio:8787``);
    this GETs ``{base_url}/metrics`` and returns the ``system`` block, shaped
    identically to :func:`collect_system_snapshot`. Raises on any network,
    HTTP, or parse error so the caller can surface an "unreachable" note.
    """

    url = base_url.rstrip("/") + "/metrics"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    system = payload.get("system")
    if not isinstance(system, dict):
        raise ValueError("remote /metrics response has no 'system' block")
    return system


# ---------------------------------------------------------------------------
# Advisory budget — headroom a dependent tool can query before it launches
# ---------------------------------------------------------------------------

# Fractions of currently-available memory a tool should feel free to claim,
# scaled down as pressure rises. At "critical" the advice is: don't start.
_MEM_CLAIM_FRACTION = {"normal": 0.75, "warning": 0.4, "critical": 0.0}

# CPU headroom (in cores) below which we advise against starting new work, and
# disk free-space floor (GB) below which we flag the disk as constrained.
_MIN_FREE_CORES = 1.0
_DISK_FREE_FLOOR_GB = 20.0


def derive_budget(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Turn a raw snapshot into advisory headroom for dependent tools.

    This is *advisory*: it never reserves or enforces anything. A tool asks
    "how much can I safely take right now?" and decides for itself. The
    ``advice`` block folds the three signals into a single go/no-go plus a
    suggested memory ceiling so a caller can act on one field.
    """

    cpu = snapshot.get("cpu") or {}
    mem = snapshot.get("memory") or {}
    disk = snapshot.get("disk") or {}

    ncpu = cpu.get("ncpu") or 1
    cpu_used = cpu.get("used_pct")
    cpu_free_pct = round(100.0 - cpu_used, 1) if cpu_used is not None else None
    free_cores = (
        round(ncpu * (100.0 - cpu_used) / 100.0, 2) if cpu_used is not None else None
    )

    pressure = mem.get("pressure")
    available_gb = mem.get("available_gb")
    claim_fraction = _MEM_CLAIM_FRACTION.get(pressure or "normal", 0.75)
    suggested_mem_gb = (
        round(available_gb * claim_fraction, 1) if available_gb is not None else None
    )

    disk_free_gb = disk.get("free_gb")
    disk_used_pct = disk.get("used_pct")

    reasons: list[str] = []
    safe = True
    if pressure == "critical":
        safe = False
        reasons.append("memory pressure critical")
    elif pressure == "warning":
        reasons.append("memory pressure warning")
    if free_cores is not None and free_cores < _MIN_FREE_CORES:
        safe = False
        reasons.append(f"cpu busy ({cpu_used:.0f}%)")
    if disk_free_gb is not None and disk_free_gb < _DISK_FREE_FLOOR_GB:
        safe = False
        reasons.append(f"low disk ({disk_free_gb:.0f} GB free)")

    return {
        "cpu": {"free_pct": cpu_free_pct, "free_cores": free_cores},
        "mem": {
            "available_gb": available_gb,
            "pressure": pressure,
        },
        "disk": {
            "free_gb": disk_free_gb,
            "free_pct": round(100.0 - disk_used_pct, 1)
            if disk_used_pct is not None
            else None,
        },
        "advice": {
            "safe_to_start": safe,
            "suggested_mem_gb": suggested_mem_gb,
            "reason": "; ".join(reasons) if reasons else "headroom available",
        },
    }
