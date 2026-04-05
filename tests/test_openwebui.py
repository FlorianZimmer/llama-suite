from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from llama_suite.utils import openwebui


UPDATE_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "scripts" / "update.py"
INSTALL_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "scripts" / "install.py"


def load_update_script_module():
    spec = importlib.util.spec_from_file_location("llama_suite_update_script", UPDATE_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_install_script_module():
    spec = importlib.util.spec_from_file_location("llama_suite_install_script", INSTALL_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_container_uses_named_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(_rt_path: str, args: list[str], check: bool = False) -> subprocess.CompletedProcess:
        captured["args"] = args
        return subprocess.CompletedProcess(args=["docker", *args], returncode=0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(openwebui, "run", fake_run)

    openwebui.create_container(
        rt="docker",
        rt_path="docker",
        name="open-webui",
        host_port=3000,
        data_dir=None,
        data_volume="open-webui",
        image="ghcr.io/open-webui/open-webui:main",
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert "-v" in args
    assert "open-webui:/app/backend/data" in args


def test_create_container_uses_host_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(_rt_path: str, args: list[str], check: bool = False) -> subprocess.CompletedProcess:
        captured["args"] = args
        return subprocess.CompletedProcess(args=["docker", *args], returncode=0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(openwebui, "run", fake_run)

    openwebui.create_container(
        rt="docker",
        rt_path="docker",
        name="open-webui",
        host_port=3000,
        data_dir=tmp_path,
        data_volume=None,
        image="ghcr.io/open-webui/open-webui:main",
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert "-v" in args
    assert f"{str(tmp_path)}:/app/backend/data" in args


def test_create_container_rejects_both_dir_and_volume(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        openwebui.create_container(
            rt="docker",
            rt_path="docker",
            name="open-webui",
            host_port=3000,
            data_dir=tmp_path,
            data_volume="open-webui",
            image="ghcr.io/open-webui/open-webui:main",
        )


def test_runtime_available_reports_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(_rt_path: str, args: list[str], check: bool = False) -> subprocess.CompletedProcess:
        assert args == ["info"]
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon",
        )

    monkeypatch.setattr(openwebui, "run", fake_run)

    available, detail = openwebui.runtime_available("docker", "docker")

    assert available is False
    assert "Cannot connect to the Docker daemon" in detail


def test_main_exits_cleanly_when_runtime_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(openwebui, "detect_runtime", lambda explicit=None: ("docker", "/usr/local/bin/docker"))
    monkeypatch.setattr(
        openwebui,
        "runtime_available",
        lambda rt_name, rt_path: (False, "Cannot connect to the Docker daemon"),
    )

    openwebui.main(["--name", "open-webui", "--port", "3000", "--data-volume", "open-webui"])

    captured = capsys.readouterr()
    assert "Skipping Open WebUI management" in captured.out
    assert "Cannot connect to the Docker daemon" in captured.out
    assert captured.err == ""


def test_resolve_container_data_settings_uses_existing_container_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        openwebui,
        "_container_inspect",
        lambda rt_path, name: {
            "Mounts": [
                {
                    "Destination": "/app/backend/data",
                    "Type": "volume",
                    "Name": "open-webui_open-webui",
                },
            ],
        },
    )
    monkeypatch.setattr(openwebui, "volume_exists", lambda rt_path, name: False)

    resolved_dir, resolved_volume = openwebui.resolve_container_data_settings(
        rt_path="docker",
        name="open-webui",
        data_dir=None,
        data_volume=None,
        default_data_dir=Path("/tmp/open-webui"),
    )

    assert resolved_dir is None
    assert resolved_volume == "open-webui_open-webui"


def test_resolve_container_data_settings_uses_orbstack_volume_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openwebui, "_container_inspect", lambda rt_path, name: None)
    monkeypatch.setattr(
        openwebui,
        "volume_exists",
        lambda rt_path, name: name == openwebui.OPEN_WEBUI_ORBSTACK_DATA_VOLUME_DEFAULT,
    )

    resolved_dir, resolved_volume = openwebui.resolve_container_data_settings(
        rt_path="docker",
        name="open-webui",
        data_dir=None,
        data_volume=None,
        default_data_dir=Path("/tmp/open-webui"),
    )

    assert resolved_dir is None
    assert resolved_volume == "open-webui_open-webui"


def test_main_uses_default_data_dir_to_auto_select_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(openwebui, "detect_runtime", lambda explicit=None: ("docker", "/usr/local/bin/docker"))
    monkeypatch.setattr(openwebui, "runtime_available", lambda rt_name, rt_path: (True, ""))
    monkeypatch.setattr(openwebui, "container_exists", lambda rt_path, name: False)
    monkeypatch.setattr(openwebui, "_container_inspect", lambda rt_path, name: None)
    monkeypatch.setattr(
        openwebui,
        "volume_exists",
        lambda rt_path, name: name == openwebui.OPEN_WEBUI_ORBSTACK_DATA_VOLUME_DEFAULT,
    )

    def fake_create_container(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(openwebui, "create_container", fake_create_container)

    openwebui.main([
        "--name", "open-webui",
        "--port", "3000",
        "--default-data-dir", str(tmp_path),
    ])

    assert captured["data_dir"] is None
    assert captured["data_volume"] == "open-webui_open-webui"


def test_needs_recreate_when_switching_bind_to_volume() -> None:
    info = {
        "Mounts": [
            {"Destination": "/app/backend/data", "Type": "bind", "Source": "/some/path"},
        ],
        "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "3000"}]}},
    }

    needs, reason = openwebui._needs_recreate_for_settings(  # noqa: SLF001
        info,
        desired_host_port=3000,
        desired_data_volume="open-webui",
        container_port=8080,
    )
    assert needs is True
    assert "data mount kind" in reason


def test_needs_recreate_when_volume_name_differs() -> None:
    info = {
        "Mounts": [
            {"Destination": "/app/backend/data", "Type": "volume", "Name": "old-volume", "Source": "/var/lib/docker/volumes/old-volume/_data"},
        ],
        "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "3000"}]}},
    }

    needs, reason = openwebui._needs_recreate_for_settings(  # noqa: SLF001
        info,
        desired_host_port=3000,
        desired_data_volume="open-webui",
        container_port=8080,
    )
    assert needs is True
    assert "data volume" in reason


def test_needs_recreate_when_port_differs() -> None:
    info = {
        "Mounts": [
            {"Destination": "/app/backend/data", "Type": "volume", "Name": "open-webui"},
        ],
        "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "3999"}]}},
    }

    needs, reason = openwebui._needs_recreate_for_settings(  # noqa: SLF001
        info,
        desired_host_port=3000,
        desired_data_volume="open-webui",
        container_port=8080,
    )
    assert needs is True
    assert "host port" in reason


def test_no_recreate_when_volume_and_port_match() -> None:
    info = {
        "Mounts": [
            {"Destination": "/app/backend/data", "Type": "volume", "Name": "open-webui"},
        ],
        "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "3000"}]}},
    }

    needs, reason = openwebui._needs_recreate_for_settings(  # noqa: SLF001
        info,
        desired_host_port=3000,
        desired_data_volume="open-webui",
        container_port=8080,
    )
    assert needs is False
    assert reason == ""


def test_refresh_openwebui_skips_when_runtime_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    update_script = load_update_script_module()
    calls: list[object] = []

    monkeypatch.setattr(update_script, "detect_runtime", lambda explicit=None: ("docker", "/usr/local/bin/docker"))
    monkeypatch.setattr(
        update_script,
        "runtime_available",
        lambda rt_name, rt_path: (False, "Cannot connect to the Docker daemon"),
    )
    monkeypatch.setattr(update_script, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    caplog.set_level("WARNING")
    update_script.refresh_openwebui(
        repo=tmp_path,
        venv_python=tmp_path / ".venv" / "bin" / "python",
        container_name="open-webui",
        port=3000,
        image="ghcr.io/open-webui/open-webui:main",
        runtime=None,
        data_volume="open-webui",
    )

    assert calls == []
    assert "Skipping Open WebUI refresh" in caplog.text
    assert "Cannot connect to the Docker daemon" in caplog.text


def test_install_uses_default_data_dir_when_no_explicit_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_script = load_install_script_module()
    captured: dict[str, object] = {}

    monkeypatch.setattr(install_script, "run", lambda cmd, **kwargs: captured.setdefault("cmd", list(cmd)))

    install_script.ensure_openwebui_container(
        venv_python=tmp_path / ".venv" / "bin" / "python",
        data_dir=tmp_path / "var" / "open-webui" / "data",
        data_volume=None,
        name="open-webui",
        port=3000,
        image="ghcr.io/open-webui/open-webui:main",
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--default-data-dir" in cmd
    assert "--data-dir" not in cmd


def test_resolve_openwebui_data_volume_prefers_requested_value() -> None:
    update_script = load_update_script_module()

    resolved = update_script.resolve_openwebui_data_volume(
        rt_path="/usr/local/bin/docker",
        container_name="open-webui",
        requested_data_volume="custom-volume",
    )

    assert resolved == "custom-volume"


def test_resolve_openwebui_data_volume_reuses_existing_container_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    update_script = load_update_script_module()

    monkeypatch.setattr(
        update_script,
        "container_inspect",
        lambda rt_path, name: {
            "Mounts": [
                {
                    "Destination": "/app/backend/data",
                    "Type": "volume",
                    "Name": "open-webui_open-webui",
                },
            ],
        },
    )
    monkeypatch.setattr(update_script, "volume_exists", lambda rt_path, name: False)

    resolved = update_script.resolve_openwebui_data_volume(
        rt_path="/usr/local/bin/docker",
        container_name="open-webui",
        requested_data_volume=None,
    )

    assert resolved == "open-webui_open-webui"


def test_resolve_openwebui_data_volume_uses_orbstack_volume_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    update_script = load_update_script_module()

    monkeypatch.setattr(update_script, "container_inspect", lambda rt_path, name: None)
    monkeypatch.setattr(
        update_script,
        "volume_exists",
        lambda rt_path, name: name == update_script.OPEN_WEBUI_ORBSTACK_DATA_VOLUME_DEFAULT,
    )

    resolved = update_script.resolve_openwebui_data_volume(
        rt_path="/usr/local/bin/docker",
        container_name="open-webui",
        requested_data_volume=None,
    )

    assert resolved == "open-webui_open-webui"


def test_refresh_openwebui_uses_named_volume_when_orbstack_volume_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    update_script = load_update_script_module()
    calls: list[list[str]] = []

    monkeypatch.setattr(update_script, "detect_runtime", lambda explicit=None: ("docker", "/usr/local/bin/docker"))
    monkeypatch.setattr(update_script, "runtime_available", lambda rt_name, rt_path: (True, ""))
    monkeypatch.setattr(update_script, "container_inspect", lambda rt_path, name: None)
    monkeypatch.setattr(
        update_script,
        "volume_exists",
        lambda rt_path, name: name == update_script.OPEN_WEBUI_ORBSTACK_DATA_VOLUME_DEFAULT,
    )
    monkeypatch.setattr(update_script, "run", lambda cmd, **kwargs: calls.append(list(cmd)))

    update_script.refresh_openwebui(
        repo=tmp_path,
        venv_python=tmp_path / ".venv" / "bin" / "python",
        container_name="open-webui",
        port=3000,
        image="ghcr.io/open-webui/open-webui:main",
        runtime=None,
        data_volume=None,
    )

    assert calls[-1][-2:] == ["--data-volume", "open-webui_open-webui"]
    assert "--data-dir" not in calls[-1]
