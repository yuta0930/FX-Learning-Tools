from __future__ import annotations

import atexit
import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(host: str, port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    port = _find_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    frozen = bool(getattr(sys, "frozen", False))

    stop_callbacks: list[callable[[], None]] = []

    def _stop_streamlit() -> None:
        for cb in stop_callbacks:
            with contextlib.suppress(Exception):
                cb()

    atexit.register(_stop_streamlit)

    if frozen:
        try:
            from streamlit.web import bootstrap
        except Exception as e:
            sys.stderr.write("Failed to import Streamlit bootstrap.\n")
            sys.stderr.write(f"Import error: {e}\n")
            return 2

        app_path = str((project_root / "app.py").resolve())
        cli_args = [
            "streamlit",
            "run",
            app_path,
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        sys.argv = cli_args

        import threading

        def _run() -> None:
            os.environ.update(env)
            bootstrap.run(app_path, "", [], {})

        t = threading.Thread(target=_run, name="streamlit-server", daemon=True)
        t.start()
        stop_callbacks.append(lambda: None)
    else:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(project_root / "app.py"),
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _stop_proc() -> None:
            if proc.poll() is not None:
                return
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                with contextlib.suppress(Exception):
                    proc.kill()

        stop_callbacks.append(_stop_proc)

    if not _wait_port(host, port, timeout_s=60.0):
        _stop_streamlit()
        sys.stderr.write("Failed to start Streamlit server.\n")
        return 2

    try:
        import webview  # type: ignore
    except Exception as e:
        _stop_streamlit()
        sys.stderr.write(
            "pywebview is not installed (or failed to import).\n"
            "Install desktop deps: pip install -r requirements_desktop.txt\n"
        )
        sys.stderr.write(f"Import error: {e}\n")
        return 3

    window = webview.create_window("FX-Learning-Tools", url)

    def _on_closed() -> None:
        _stop_streamlit()

    with contextlib.suppress(Exception):
        window.events.closed += _on_closed  # pywebview event hook

    try:
        webview.start()
    finally:
        _stop_streamlit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
