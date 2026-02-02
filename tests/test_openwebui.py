from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llama_suite.utils import openwebui


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
