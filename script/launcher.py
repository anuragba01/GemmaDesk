import importlib.metadata
import multiprocessing
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import webview

# Ultimate Safety Net: Mock importlib.metadata.version to prevent PackageNotFoundError crash in PyInstaller environment
_orig_version = importlib.metadata.version


def _mock_version(package_name):
    try:
        return _orig_version(package_name)
    except Exception:
        if package_name == "streamlit":
            return "1.57.0"
        return "1.0.0"


importlib.metadata.version = _mock_version

try:
    import importlib_metadata

    _orig_im_version = importlib_metadata.version

    def _mock_im_version(package_name):
        try:
            return _orig_im_version(package_name)
        except Exception:
            if package_name == "streamlit":
                return "1.57.0"
            return "1.0.0"

    importlib_metadata.version = _mock_im_version
except ImportError:
    pass


def _choose_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _startup_log_path() -> str:
    runtime_dir = tempfile.gettempdir()
    return os.path.join(runtime_dir, "gemmadesk-launcher.log")


def _append_log(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(_startup_log_path(), "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _resolve_runtime_paths() -> tuple[str, str]:
    if getattr(sys, "frozen", False):
        base_dir = sys._MEIPASS
        app_path = os.path.join(base_dir, "app", "app.py")
        project_root = base_dir
        if not os.path.exists(app_path):
            app_path = os.path.join(base_dir, "_internal", "app", "app.py")
            project_root = os.path.join(base_dir, "_internal")
        return project_root, app_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_path = os.path.join(project_root, "app", "app.py")
    return project_root, app_path


def run_streamlit_server(app_path: str, port: int):
    """Directly runs the Streamlit server in this process."""
    import streamlit.web.cli as stcli
    from streamlit import config, file_util

    _append_log(
        "run_streamlit_server: "
        f"app_path={app_path} port={port} "
        f"streamlit_file={getattr(file_util, '__file__', 'unknown')} "
        f"static_dir={file_util.get_static_dir()} "
        f"static_exists={os.path.isdir(file_util.get_static_dir())}"
    )

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--global.developmentMode=false",
    ]
    _append_log(
        "streamlit argv prepared: "
        f"developmentMode={config.get_option('global.developmentMode')} "
        f"baseUrlPath={config.get_option('server.baseUrlPath')}"
    )
    sys.exit(stcli.main())


def start_streamlit(port: int):
    """Starts the Streamlit app in a background process."""
    print("Starting GemmaDesk Server...")
    project_root, app_path = _resolve_runtime_paths()
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Streamlit entrypoint not found: {app_path}")

    _append_log(f"Starting Streamlit with app_path={app_path} cwd={project_root} port={port}")

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--internal-streamlit", app_path, str(port)]
    else:
        cmd = [sys.executable, __file__, "--internal-streamlit", app_path, str(port)]

    log_handle = open(_startup_log_path(), "a", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        cwd=project_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle


def _wait_for_server(url: str, process: subprocess.Popen, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            _append_log(f"Backend process exited early with code {process.returncode}")
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status < 500:
                    _append_log(f"Server became ready at {url}")
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(0.25)
    _append_log(f"Timed out waiting for server at {url}")
    return False


def _show_startup_error(url: str):
    log_path = _startup_log_path()
    html = f"""
    <html>
      <body style=\"font-family: sans-serif; padding: 24px;\">
        <h2>GemmaDesk failed to start</h2>
        <p>The local server did not become ready at <code>{url}</code>.</p>
        <p>Startup logs were written to:</p>
        <pre>{log_path}</pre>
      </body>
    </html>
    """
    webview.create_window("GemmaDesk Startup Error", html=html, width=900, height=420)
    webview.start(gui="qt")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if len(sys.argv) > 3 and sys.argv[1] == "--internal-streamlit":
        run_streamlit_server(sys.argv[2], int(sys.argv[3]))
        sys.exit(0)

    port = _choose_free_port()
    url = f"http://127.0.0.1:{port}"
    server_process = None
    log_handle = None
    try:
        server_process, log_handle = start_streamlit(port)
        if not _wait_for_server(url, server_process):
            if server_process.poll() is None:
                server_process.terminate()
            _show_startup_error(url)
            sys.exit(1)

        window = webview.create_window("GemmaDesk - Offline AI", url, width=1280, height=800)
        webview.start(gui="qt")
    finally:
        print("Shutting down GemmaDesk...")
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        if log_handle is not None:
            log_handle.close()
