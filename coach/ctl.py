"""Start/stop the coach UI server without keeping a terminal open.

    coach start [--tier pro] [--no-jev] ...   # any coach.ui flag passes through
    coach stop
    coach status
    coach restart [...]
    coach logs [-f] [-n LINES]

The server is `python -m coach.ui`; the Swift fluid-poc worker is its child
and dies with it. `start` launches it detached in its own process group with
output going to <data_dir>/server.log; `stop` signals the group so a hung UI
cannot orphan the worker. The pid lives in <data_dir>/server.pid.

Run from anywhere; the venv's python is used (whoever runs this command).
Config comes from coach.yaml / FLUENCY_CONFIG / env as usual.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import default_data_dir

# UI startup imports torch/nicegui before the port opens; give it this long
# before "still starting" is reported instead of a URL.
STARTUP_WAIT_S = 20
STOP_WAIT_S = 10


def _pid_path() -> Path:
    return default_data_dir() / "server.pid"


def _info_path() -> Path:
    return default_data_dir() / "server.json"


def _write_info(pid: int, host: str, port: int) -> None:
    import json
    _info_path().write_text(json.dumps({"pid": pid, "host": host, "port": port}))


def _read_info() -> dict:
    import json
    try:
        info = json.loads(_info_path().read_text())
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {}


def _log_path() -> Path:
    return default_data_dir() / "server.log"


def _cleanup_files() -> None:
    _pid_path().unlink(missing_ok=True)
    _info_path().unlink(missing_ok=True)


def _read_pid() -> int | None:
    try:
        return int(_pid_path().read_text().strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, not ours — still "running" for status
    return True


def _url(host: str, port: int) -> str:
    display = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{display}:{port}"


def _settings(argv: list[str]):
    """Resolved settings for status display; never fatal on bad config."""
    try:
        from .config import load
        return load(argv)
    except Exception:
        return None


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host or "127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def cmd_start(argv: list[str]) -> int:
    pid = _read_pid()
    if pid is not None and _alive(pid):
        print(f"already running (pid {pid}) — try `coach status` or `coach stop`")
        return 1

    settings = _settings(argv)
    host = settings.host if settings else "127.0.0.1"
    port = settings.port if settings else 8080
    if _port_open(host, port):
        print(f"something is already listening on {host}:{port} (not started "
              f"by `coach`) — stop it or pass --port")
        return 1

    data_dir = default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path()
    log = open(log_path, "ab", buffering=0)
    log.write(f"\n--- coach start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode())

    proc = subprocess.Popen(
        [sys.executable, "-m", "coach.ui", *argv],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # own group: survives this terminal, stop hits worker too
        cwd=_project_root(),
    )
    _pid_path().write_text(f"{proc.pid}\n")
    _write_info(proc.pid, host, port)

    deadline = time.monotonic() + STARTUP_WAIT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"server exited during startup (code {proc.returncode}); "
                  f"last log lines:")
            _tail(log_path, 10)
            _cleanup_files()
            return 1
        if _port_open(host, port):
            if proc.poll() is not None:
                # port opened by someone else while ours died
                print(f"server exited during startup (code {proc.returncode}); "
                      f"last log lines:")
                _tail(log_path, 10)
                _cleanup_files()
                return 1
            print(f"fluency coach running — {_url(host, port)}  (pid {proc.pid})")
            print(f"log: {log_path}")
            return 0
        time.sleep(0.5)

    print(f"still starting (model imports are slow) — check `coach status` "
          f"in a moment; log: {log_path}")
    return 0


def cmd_stop(_argv: list[str]) -> int:
    pid = _read_pid()
    if pid is None or not _alive(pid):
        _cleanup_files()
        print("not running")
        return 0
    # The whole group: the UI plus any spawned fluid-poc worker.
    for sig, what in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
        deadline = time.monotonic() + (STOP_WAIT_S if sig == signal.SIGTERM else 3)
        while time.monotonic() < deadline:
            if not _alive(pid):
                _cleanup_files()
                print(f"stopped (pid {pid})")
                return 0
            time.sleep(0.2)
        if sig == signal.SIGTERM:
            print("did not exit after SIGTERM; sending SIGKILL…")
    _cleanup_files()
    print(f"stopped (pid {pid})")
    return 0


def cmd_status(_argv: list[str]) -> int:
    pid = _read_pid()
    if pid is None or not _alive(pid):
        _cleanup_files()
        settings = _settings([])
        if settings and _port_open(settings.host, settings.port):
            print(f"not tracked by `coach`, but something is listening on "
                  f"{_url(settings.host, settings.port)}")
            return 1
        print("not running")
        return 1
    info = _read_info()
    settings = _settings([])
    host = str(info.get("host") or (settings.host if settings else "127.0.0.1"))
    try:
        port = int(info.get("port") or (settings.port if settings else 8080))
    except (TypeError, ValueError):
        port = settings.port if settings else 8080
    state = "up" if _port_open(host, port) else "starting"
    print(f"running (pid {pid}, {state}) — {_url(host, port)}")
    print(f"log: {_log_path()}")
    return 0


def cmd_restart(argv: list[str]) -> int:
    cmd_stop([])
    return cmd_start(argv)


def _tail(path: Path, n: int) -> None:
    try:
        lines = path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return
    for line in lines:
        print(line)


def cmd_logs(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--follow", action="store_true")
    parser.add_argument("-n", "--lines", type=int, default=40)
    opts, _ = parser.parse_known_args(argv)

    path = _log_path()
    if not path.exists():
        print(f"no log yet ({path})")
        return 1
    _tail(path, opts.lines)
    if not opts.follow:
        return 0
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            while True:
                chunk = f.read(4096)
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def _project_root() -> Path:
    """cwd if it holds coach.yaml (config is picked up from cwd); otherwise
    the directory containing the coach package, so `python -m coach.ui`
    imports even when the project is not pip-installed."""
    if (Path.cwd() / "coach.yaml").is_file():
        return Path.cwd()
    return Path(__file__).resolve().parent.parent


def main() -> int | None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "start":
        return cmd_start(rest)
    if cmd == "stop":
        return cmd_stop(rest)
    if cmd == "status":
        return cmd_status(rest)
    if cmd == "restart":
        return cmd_restart(rest)
    if cmd == "logs":
        return cmd_logs(rest)
    print(f"unknown command {cmd!r} — try `coach --help`", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
