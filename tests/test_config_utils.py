from __future__ import annotations

from llama_suite.utils.config_utils import build_llama_server_command_util


def test_build_llama_server_command_maps_flash_attn_bool_to_on() -> None:
    cmd = build_llama_server_command_util(
        {
            "_name_for_log": "m1",
            "cmd": {
                "bin": "llama-server",
                "port": 9001,
                "model": "models/m1.gguf",
                "ctx-size": 8192,
                "flash-attn": True,
            },
            "sampling": {"temp": 0.6},
        }
    )

    assert "--flash-attn on" in cmd
    assert "--flash-attn --temp" not in cmd


def test_build_llama_server_command_maps_flash_attn_bool_to_off() -> None:
    cmd = build_llama_server_command_util(
        {
            "_name_for_log": "m1",
            "cmd": {
                "bin": "llama-server",
                "port": 9001,
                "model": "models/m1.gguf",
                "ctx-size": 8192,
                "flash-attn": False,
            },
        }
    )

    assert "--flash-attn off" in cmd


def test_build_llama_server_command_quotes_json_kwargs() -> None:
    cmd = build_llama_server_command_util(
        {
            "_name_for_log": "m1",
            "cmd": {
                "bin": "llama-server",
                "port": 9001,
                "model": "models/m1.gguf",
                "ctx-size": 8192,
                "chat-template-kwargs": {"enable_thinking": False},
            },
        }
    )

    assert "--chat-template-kwargs '{\"enable_thinking\":false}'" in cmd


def test_build_llama_server_command_maps_legacy_draft_flag() -> None:
    cmd = build_llama_server_command_util(
        {
            "_name_for_log": "m1",
            "cmd": {
                "bin": "llama-server",
                "port": 9001,
                "model": "models/m1.gguf",
                "ctx-size": 8192,
                "spec-type": "draft-mtp",
                "draft": 4,
            },
        }
    )

    assert "--spec-type draft-mtp" in cmd
    assert "--spec-draft-n-max 4" in cmd
    assert "--draft 4" not in cmd
