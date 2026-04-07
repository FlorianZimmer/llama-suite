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
