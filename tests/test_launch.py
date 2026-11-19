import socket

import launch


def test_browser_command_windows_prefers_app_mode():
    cmd = launch.browser_command("http://localhost:8501", "win32", exists=lambda p: p.endswith("msedge.exe"))
    assert cmd[0].endswith("msedge.exe") and "--app=http://localhost:8501" in cmd
    assert launch.browser_command("http://x", "win32", exists=lambda p: False) is None  # default browser


def test_browser_command_mac():
    cmd = launch.browser_command("http://x", "darwin", exists=lambda p: p == "/Applications/Google Chrome.app")
    assert cmd[:3] == ["open", "-na", "Google Chrome"] and "--app=http://x" in cmd
    assert launch.browser_command("http://x", "darwin", exists=lambda p: False) is None  # Safari: normal window


def test_free_port_skips_a_busy_port():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert launch.free_port(port) != port


def test_running_port_ignores_stale_file(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "PORT_FILE", tmp_path / "app.port")
    assert launch.running_port() is None  # no file
    (tmp_path / "app.port").write_text("1")  # nothing serves on port 1
    assert launch.running_port() is None


def test_mac_shell_quoting():
    assert launch._q("/Users/a b/it's") == "'/Users/a b/it'\\''s'"


def test_icon(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    out = launch.make_icon()
    assert (out / "icon.png").stat().st_size > 0 and (out / "icon.ico").stat().st_size > 0
