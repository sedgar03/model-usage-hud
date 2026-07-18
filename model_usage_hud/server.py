"""Tailnet server for the System provider — metrics, budget, and a web HUD.

Exposes the same local-machine snapshot the HUD renders over HTTP so it can be
watched from another device and consumed by tools that want to size their work
to available headroom.

Endpoints
---------
- ``GET /``         a self-contained dark web HUD that polls ``/metrics``
- ``GET /metrics``  the raw snapshot as JSON (CPU/memory/swap/disk)
- ``GET /budget``   advisory headroom derived from the snapshot (see
                    :func:`~model_usage_hud.core.system_metrics.derive_budget`)
- ``GET /healthz``  ``{"ok": true}`` for liveness checks

Binding
-------
By default the server binds to this machine's Tailscale IP (the 100.64.0.0/10
address), so it is reachable from other tailnet devices and from the host
itself, but not from the public internet or a coffee-shop LAN. If no Tailscale
address is found it falls back to ``127.0.0.1``. Pass ``--host 0.0.0.0`` to
bind every interface (only do this behind a firewall).

The payload is deliberately low-sensitivity (utilization percentages and free
space, no file names or process lists), but it is served without
authentication — keep it on the tailnet.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import subprocess
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from model_usage_hud.core.system_metrics import (
    DEFAULT_DISK_PATH,
    collect_system_snapshot,
    derive_budget,
)

DEFAULT_PORT = 8787
# Tailscale hands out addresses from the 100.64.0.0/10 CGNAT block.
_TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")

# Common install locations for the Tailscale CLI on macOS, where it is not
# usually on PATH (it ships inside the app bundle).
_TAILSCALE_BINARIES = (
    "tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    "/usr/local/bin/tailscale",
    "/opt/homebrew/bin/tailscale",
)


def _run_tailscale(args: list[str]) -> str | None:
    """Run the Tailscale CLI (wherever it lives) and return stdout, or None."""

    for binary in _TAILSCALE_BINARIES:
        try:
            out = subprocess.run(
                [binary, *args],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            return out.stdout
    return None


def _tailscale_ip_via_cli() -> str | None:
    stdout = _run_tailscale(["ip", "-4"])
    if stdout is None:
        return None
    for line in stdout.splitlines():
        candidate = line.strip()
        try:
            if ipaddress.ip_address(candidate) in _TAILSCALE_CGNAT:
                return candidate
        except ValueError:
            continue
    return None


def detect_peer_remote(
    name_hint: str, *, port: int = DEFAULT_PORT
) -> tuple[str, str] | None:
    """Find a tailnet peer whose hostname contains ``name_hint``.

    Returns ``(base_url, short_label)`` for the first match (e.g.
    ``("http://100.98.120.101:8787", "stevens-mac-studio")``) so the HUD can
    default its System provider to that machine, or ``None`` if no peer
    matches or the Tailscale CLI is unavailable. Offline peers still match — we
    want the HUD to keep pointing at the intended box and show "unreachable"
    rather than silently fall back to the local machine.
    """

    stdout = _run_tailscale(["status"])
    if stdout is None:
        return None
    hint = name_hint.lower()
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        ip, host = parts[0], parts[1]
        if hint not in host.lower():
            continue
        try:
            if ipaddress.ip_address(ip) not in _TAILSCALE_CGNAT:
                continue
        except ValueError:
            continue
        return f"http://{ip}:{port}", host
    return None


def _tailscale_ip_via_interfaces() -> str | None:
    """Fallback: scan local addresses for one in the Tailscale range."""

    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    candidates = {info[4][0] for info in infos}
    # getaddrinfo often misses the tailnet address; also probe the utun-style
    # addresses the OS knows about via a UDP connect trick per interface is
    # overkill, so we rely on the CLI first and this as a best effort.
    for addr in candidates:
        try:
            if ipaddress.ip_address(addr) in _TAILSCALE_CGNAT:
                return addr
        except ValueError:
            continue
    return None


def detect_bind_host() -> str:
    """Best-effort Tailscale IP, falling back to loopback."""

    return (
        _tailscale_ip_via_cli()
        or _tailscale_ip_via_interfaces()
        or "127.0.0.1"
    )


def build_metrics_payload(disk_path: str = DEFAULT_DISK_PATH) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "system": collect_system_snapshot(disk_path),
    }


def build_budget_payload(disk_path: str = DEFAULT_DISK_PATH) -> dict[str, Any]:
    snapshot = collect_system_snapshot(disk_path)
    payload = derive_budget(snapshot)
    payload["generated_at"] = datetime.now().astimezone().isoformat()
    return payload


def _make_handler(disk_path: str) -> type[BaseHTTPRequestHandler]:
    class MetricsHandler(BaseHTTPRequestHandler):
        server_version = "usage-hud-serve/1.0"

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                if path == "/":
                    self._send_html(WEB_HUD_HTML)
                elif path == "/metrics":
                    self._send_json(build_metrics_payload(disk_path))
                elif path == "/budget":
                    self._send_json(build_budget_payload(disk_path))
                elif path == "/healthz":
                    self._send_json({"ok": True})
                else:
                    self._send_json({"error": "not found", "path": path}, status=404)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, *args: Any) -> None:
            # Quiet by default — one request every few seconds from the web HUD
            # would otherwise spam the console.
            return

    return MetricsHandler


def serve(
    *,
    host: str | None = None,
    port: int = DEFAULT_PORT,
    disk_path: str = DEFAULT_DISK_PATH,
) -> int:
    bind_host = host or detect_bind_host()
    handler = _make_handler(disk_path)
    try:
        httpd = ThreadingHTTPServer((bind_host, port), handler)
    except OSError as exc:
        print(f"Could not bind {bind_host}:{port}: {exc}", file=sys.stderr)
        return 1

    scope = "tailnet + localhost" if bind_host.startswith("100.") else (
        "ALL interfaces" if bind_host == "0.0.0.0" else "localhost only"
    )
    print(f"usage-hud metrics server on http://{bind_host}:{port}  ({scope})")
    print(f"  web HUD : http://{bind_host}:{port}/")
    print(f"  metrics : http://{bind_host}:{port}/metrics")
    print(f"  budget  : http://{bind_host}:{port}/budget")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve local machine metrics + advisory budget over the tailnet."
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "Bind address (default: this machine's Tailscale IP, else 127.0.0.1). "
            "Use 0.0.0.0 to bind all interfaces."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--disk-path",
        default=DEFAULT_DISK_PATH,
        help=f"Filesystem to report free space for (default: {DEFAULT_DISK_PATH})",
    )
    args = parser.parse_args(argv)
    if args.port <= 0 or args.port > 65535:
        print("--port must be in 1..65535", file=sys.stderr)
        return 2
    return serve(host=args.host, port=args.port, disk_path=args.disk_path)


# Self-contained dark web HUD. No external assets — polls /metrics and /budget
# and paints threshold-tinted gauges matching the desktop app.
WEB_HUD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>System HUD</title>
<style>
  :root {
    --bg:#111315; --panel:#171a1d; --border:#2a2d30; --fg:#f0f6fc;
    --muted:#8b949e; --green:#56d364; --yellow:#e3b341; --red:#ff7b72;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--fg);
    font-family:ui-monospace,Menlo,Consolas,monospace;
    display:flex; justify-content:center; padding:24px;
  }
  .card {
    background:var(--panel); border:1px solid var(--border);
    border-radius:12px; padding:18px 20px; width:min(460px,100%);
  }
  h1 { font-size:15px; font-weight:600; margin:0 0 2px; }
  .meta { color:var(--muted); font-size:12px; margin-bottom:16px; }
  .row { display:grid; grid-template-columns:52px 54px 1fr; align-items:center;
         gap:10px; margin:10px 0; }
  .label { font-weight:600; }
  .value { text-align:right; font-variant-numeric:tabular-nums; }
  .track { position:relative; height:8px; border-radius:4px;
           background:var(--border); overflow:hidden; }
  .fill { position:absolute; inset:0 auto 0 0; border-radius:4px; width:0;
          transition:width .4s ease, background .4s ease; }
  .detail { color:var(--muted); font-size:12px; margin:2px 0 0 116px; }
  .pressure { margin-top:14px; font-size:13px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:6px;
           font-size:12px; font-weight:600; }
  .ok { color:var(--green); } .warn { color:var(--yellow); } .crit { color:var(--red); }
  .advice { margin-top:16px; padding-top:14px; border-top:1px solid var(--border);
            font-size:13px; color:var(--muted); }
  .stale { opacity:.5; }
</style>
</head>
<body>
<div class="card">
  <h1>System HUD</h1>
  <div class="meta" id="meta">connecting…</div>
  <div id="gauges"></div>
  <div class="advice" id="advice"></div>
</div>
<script>
const POLL_MS = 5000;
function tone(pct) { if (pct == null) return 'var(--border)'; if (pct >= 95) return 'var(--red)'; if (pct >= 80) return 'var(--yellow)'; return 'var(--green)'; }
function pressureColor(p) { return p === 'critical' ? 'var(--red)' : p === 'warning' ? 'var(--yellow)' : 'var(--green)'; }
function gauge(label, pct, detail, color) {
  const p = (pct == null) ? 0 : Math.max(0, Math.min(100, pct));
  const shown = (pct == null) ? '--' : Math.round(pct) + '%';
  return `<div class="row"><span class="label">${label}</span>`
    + `<span class="value">${shown}</span>`
    + `<span class="track"><span class="fill" style="width:${p}%;background:${color || tone(p)}"></span></span></div>`
    + (detail ? `<div class="detail">${detail}</div>` : '');
}
async function tick() {
  try {
    const [m, b] = await Promise.all([
      fetch('/metrics').then(r => r.json()),
      fetch('/budget').then(r => r.json()),
    ]);
    const s = m.system || {};
    const cpu = s.cpu || {}, mem = s.memory || {}, swp = s.swap || {};
    let html = '';
    html += gauge('CPU', cpu.used_pct, cpu.load1 != null ? ('load ' + cpu.load1) : '');
    // MEM: fill = used %, color = memory pressure (the health verdict). Swap
    // folds into the detail only when pressure is actually elevated.
    const pr = mem.pressure;
    let memDetail = mem.available_gb != null ? (mem.available_gb + 'G free') : '';
    if ((pr === 'warning' || pr === 'critical') && (swp.used_gb || 0) > 0.05)
      memDetail += ' · ' + Math.round(swp.used_gb) + 'G swap';
    let memColor = pressureColor(pr);
    if (pr === 'normal' && mem.used_pct != null && mem.used_pct >= 95) memColor = 'var(--yellow)';
    html += gauge('MEM', mem.used_pct, memDetail, memColor);
    document.getElementById('gauges').innerHTML = html;

    const adv = b.advice || {};
    document.getElementById('advice').innerHTML =
      `<b class="${adv.safe_to_start ? 'ok' : 'crit'}">${adv.safe_to_start ? 'OK to start work' : 'hold off'}</b>`
      + ` · ${adv.reason || ''}`
      + (adv.suggested_mem_gb != null ? ` · suggest ≤ ${adv.suggested_mem_gb}G` : '');

    const t = new Date(m.generated_at);
    document.getElementById('meta').textContent = 'updated ' + t.toLocaleTimeString();
    document.body.classList.remove('stale');
  } catch (e) {
    document.getElementById('meta').textContent = 'connection lost, retrying…';
    document.body.classList.add('stale');
  }
}
tick();
setInterval(tick, POLL_MS);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
