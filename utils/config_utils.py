# config_utils.py
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml # Dependency: PyYAML
from colorama import Fore, Style, init as colorama_init # Renamed for clarity

# Initialize Colorama
try:
    colorama_init(autoreset=True)
except Exception:
    print("Warning: colorama init failed or not installed. Proceeding without colors.")
    class DummyStyle: RESET_ALL = "" # type: ignore
    class DummyFore: RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA = ("",)*6 # type: ignore
    Style = DummyStyle() # type: ignore
    Fore = DummyFore() # type: ignore

# --- Constants for build_llama_server_command ---
MANDATORY_CMD_KEYS = {"bin", "port", "model", "ctx-size"}
MODEL_META_KEYS = {"aliases", "sampling"} # Keys at the model's top-level, not CLI flags

# Keys in the 'cmd' dictionary that might contain relative paths to be resolved.
# 'bin' is the server executable.
# 'model' is the main model GGUF.
# 'md' is assumed to be the draft model GGUF path.
# Add other keys like 'mmproj', 'lora' if they are file paths and live in 'cmd'.
PATH_KEYS_IN_CMD = {"bin", "model", "model-draft"}

# --- Constants for cleaning up effective config ---
REDUNDANT_BLOCK_PATTERNS = {
    "suffixes": ["_SAMPLING", "_FLAGS", "_COMMON"],
    "prefixes": ["COMMON_"],
}
PRESERVED_EFFECTIVE_CONFIG_KEYS = {"models"}


# --- Utility Functions ---
def colour_util(text: str, ansi_color: str) -> str:
    return f"{ansi_color}{text}{Style.RESET_ALL}"

def deep_merge_dicts_util(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge_dicts_util(target[key], value)
        else:
            target[key] = value

# --- Custom YAML Loader ---
class ConfigEnvLoader(yaml.SafeLoader):
    pass

def _env_constructor_util(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    value = loader.construct_scalar(node)
    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        print(colour_util(f"Util: Warning - Expected ${{VAR}} format for !ENV tag, got: {value}", Fore.YELLOW))
        return str(value)
    raw_content = value.strip("${}")
    env_var_name, _, default_value = raw_content.partition(":-")
    return os.getenv(env_var_name, default_value or "")

ConfigEnvLoader.add_constructor("!ENV", _env_constructor_util)


# --- Helper Function to Resolve Paths ---
def resolve_path_relative_to_config(
    path_str: Union[str, Path],
    config_dir: Path,
    key_name_for_log: str = "path" # For more informative logging
) -> str:
    """Resolves a path string relative to the config directory if it's relative."""
    if not path_str:
        print(colour_util(f"Util: Warning - Empty path provided for '{key_name_for_log}'. Returning empty string.", Fore.YELLOW))
        return ""

    try:
        path_obj = Path(path_str)
        if not path_obj.is_absolute():
            resolved_path = (config_dir / path_obj).resolve()
            # No verbose logging here, moved to caller for better context
            return str(resolved_path)
        return str(path_obj) # Already absolute
    except Exception as e:
        print(colour_util(f"Util: Error - Could not process path '{path_str}' for '{key_name_for_log}': {e}", Fore.RED))
        # Depending on policy, might raise an error or return original path
        # For robustness in finding models, returning original might lead to "file not found" later, which is fine.
        return str(path_str)

# --- Core Command Building Logic ---
def build_llama_server_command_util(model_config: Dict[str, Any]) -> str:
    """
    Assembles the command-line string for llama-server based on model configuration.
    Expects 'cmd' sub-dictionary with RESOLVED paths (especially for 'bin', 'model', 'md').
    """
    cmd_map = model_config.get("cmd")
    model_name_for_log = model_config.get("_name_for_log", "Unknown Model")

    if not isinstance(cmd_map, dict):
        raise ValueError(f"Model '{model_name_for_log}': Config missing 'cmd' dictionary.")

    missing_keys = MANDATORY_CMD_KEYS - cmd_map.keys()
    if missing_keys:
        raise ValueError(
            f"Model '{model_name_for_log}': 'cmd' block missing required key(s): {', '.join(sorted(missing_keys))}"
        )

    # Ensure paths are strings and exist for critical components
    bin_path = str(cmd_map["bin"])
    model_path = str(cmd_map["model"])

    # Basic check for existence of bin and model paths (optional but good for early error)
    # if not Path(bin_path).exists():
    #     print(colour_util(f"Util: Warning - Server binary '{bin_path}' for model '{model_name_for_log}' does not exist.", Fore.YELLOW))
    # if not Path(model_path).exists():
    #     print(colour_util(f"Util: Warning - Main model file '{model_path}' for model '{model_name_for_log}' does not exist.", Fore.YELLOW))


    args = [
        bin_path,
        "--port", str(cmd_map["port"]),
        "--model", model_path,
        "--ctx-size", str(cmd_map["ctx-size"]),
    ]

    if str(cmd_map.get("gpu-layers", "auto")).lower() != "auto":
        args.extend(["--n-gpu-layers", str(cmd_map["gpu-layers"])])
    if str(cmd_map.get("threads", "auto")).lower() != "auto":
        args.extend(["--threads", str(cmd_map["threads"])])

    for key, value in cmd_map.items():
        if key in MANDATORY_CMD_KEYS | {"gpu-layers", "threads"}:
            continue
        cli_flag = f"--{key.replace('_', '-')}"
        
        # For 'md' (draft model), ensure value is a non-empty string (path)
        if key == "md": # Assuming 'md' is the draft model key in 'cmd'
            if value and isinstance(value, str): # Value should be the resolved path string
                # if not Path(value).exists(): # Optional check
                #    print(colour_util(f"Util: Warning - Draft model '{value}' for model '{model_name_for_log}' does not exist.", Fore.YELLOW))
                args.extend([cli_flag, value]) # value is already a string path
            elif value: # It's not a string or empty
                print(colour_util(f"Util: Warning - Invalid value for draft model 'md' ('{value}') for model '{model_name_for_log}'. Skipping.", Fore.YELLOW))
            continue # ensure 'md' is fully handled here

        # Handle other path keys if they are not 'bin', 'model', 'md' but are in PATH_KEYS_IN_CMD
        # This part might be redundant if all path keys are handled specifically or are 'bin'/'model'/'md'
        # if key in PATH_KEYS_IN_CMD and value and isinstance(value, str):
        #    args.extend([cli_flag, value]) # Value is already resolved path string
        #    continue

        if isinstance(value, bool) and value:
            args.append(cli_flag)
        elif value not in (None, False, "auto", "Auto", ""): # Also skip empty strings for non-flag args
            args.extend([cli_flag, str(value)])

    for key, value in model_config.items():
        if key == "cmd" or key in MODEL_META_KEYS or key == "_name_for_log":
            continue
        cli_flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value:
            args.append(cli_flag)
        elif value not in (None, False, "auto", "Auto", ""):
            args.extend([cli_flag, str(value)])

    sampling_config = model_config.get("sampling")
    if isinstance(sampling_config, dict):
        for key, s_value in sampling_config.items():
            cli_flag = f"--{key.replace('_', '-')}"
            args.extend([cli_flag, str(s_value)])
    elif sampling_config is not None:
         print(colour_util(f"[WARN] Util: 'sampling' for model '{model_name_for_log}' is not a dictionary, ignoring.", Fore.YELLOW))
         
    return " ".join(args)


# --- Main Configuration Processing Function ---
def generate_processed_config(
    base_config_path_arg: Path,
    override_config_path_arg: Optional[Path] = None,
    script_dir_for_overrides: Optional[Path] = None,
    verbose_logging: bool = True
) -> Dict[str, Any]:
    """
    Loads, merges, resolves relative paths, processes model configurations,
    and cleans up redundant top-level blocks.
    """
    base_config_path = Path(base_config_path_arg).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base configuration file not found or is not a file: {base_config_path}")

    config_root_dir = base_config_path.parent # Key for resolving relative paths
    if verbose_logging:
        print(colour_util(f"Util: Using config root for relative paths: {config_root_dir}", Fore.BLUE))

    try:
        text_content = base_config_path.read_text(encoding="utf-8")
        effective_config = yaml.load(text_content, Loader=ConfigEnvLoader) if text_content.strip() else {}
        if not isinstance(effective_config, dict):
            raise ValueError(f"Base config {base_config_path} did not parse into a dictionary.")
    except Exception as e:
        raise ValueError(f"Error loading base config {base_config_path}: {e}") from e

    override_to_load: Optional[Path] = None
    if override_config_path_arg:
        resolved_override = Path(override_config_path_arg).expanduser().resolve()
        if not resolved_override.is_file():
            raise FileNotFoundError(f"Specified override file not found: {resolved_override}")
        override_to_load = resolved_override
    elif script_dir_for_overrides: # script_dir_for_overrides is typically project root
        hostname = platform.node().split(".")[0].lower()
        default_override = Path(script_dir_for_overrides) / "overrides" / f"{hostname}.yaml"
        if default_override.is_file():
            override_to_load = default_override
            if verbose_logging: print(colour_util(f"Util: Found hostname override: {default_override}", Fore.BLUE))
        elif verbose_logging: print(colour_util(f"Util: No hostname override found for '{hostname}' in {Path(script_dir_for_overrides) / 'overrides'}.", Fore.BLUE))

    if override_to_load:
        if verbose_logging: print(colour_util(f"Util: Loading override: {override_to_load}", Fore.BLUE))
        try:
            override_text = override_to_load.read_text(encoding="utf-8")
            if override_text.strip():
                override_data = yaml.load(override_text, Loader=ConfigEnvLoader)
                if isinstance(override_data, dict):
                    deep_merge_dicts_util(effective_config, override_data)
                    if verbose_logging: print(colour_util(f"Util: Merged override: {override_to_load}", Fore.CYAN))
                elif verbose_logging: print(colour_util(f"Util: Override file {override_to_load} not a dictionary.", Fore.YELLOW))
            elif verbose_logging and override_config_path_arg and Path(override_config_path_arg).resolve() == override_to_load:
                 print(colour_util(f"Util: Specified override file {override_to_load} is empty.", Fore.YELLOW))
        except Exception as e:
            raise ValueError(f"Error loading override config {override_to_load}: {e}") from e
    elif verbose_logging and not override_config_path_arg : # No override arg and no hostname override found
        print(colour_util("Util: No override file specified or default found.", Fore.BLUE))


    models_section = effective_config.get("models", {})
    if not isinstance(models_section, dict):
        if verbose_logging: print(colour_util("Util: 'models' section is not a dictionary. No models processed.", Fore.YELLOW))
    else:
        for model_name, model_data in models_section.items():
            if not isinstance(model_data, dict):
                if verbose_logging: print(colour_util(f"Util: Config for model '{model_name}' is not a dict. Skipping.", Fore.YELLOW))
                continue
            
            original_cmd_dict = model_data.get("cmd")
            if not isinstance(original_cmd_dict, dict):
                raise ValueError(f"Model '{model_name}' must have a 'cmd' dictionary for processing.")

            resolved_cmd_dict = original_cmd_dict.copy()
            for key_to_resolve in PATH_KEYS_IN_CMD:
                if key_to_resolve in resolved_cmd_dict and isinstance(resolved_cmd_dict[key_to_resolve], str):
                    original_path = resolved_cmd_dict[key_to_resolve]
                    if original_path: # Only resolve non-empty paths
                        # Resolve relative to the directory of the BASE config file (config_root_dir)
                        resolved_path_str = resolve_path_relative_to_config(original_path, config_root_dir, key_name_for_log=f"{model_name}.cmd.{key_to_resolve}")
                        if verbose_logging and original_path != resolved_path_str:
                            print(colour_util(f"Util: Model '{model_name}', resolved 'cmd.{key_to_resolve}': '{original_path}' -> '{resolved_path_str}'", Fore.MAGENTA))
                        resolved_cmd_dict[key_to_resolve] = resolved_path_str
                    elif verbose_logging: # Path was empty string
                         print(colour_util(f"Util: Model '{model_name}', 'cmd.{key_to_resolve}' is an empty string. Not resolving.", Fore.BLUE))

            model_data["cmd"] = resolved_cmd_dict # Update model_data with resolved paths in 'cmd'

            try:
                build_input = model_data.copy()
                build_input["_name_for_log"] = model_name
                command_str = build_llama_server_command_util(build_input)
                model_data["generated_cmd_str"] = command_str
                    
            except ValueError as e:
                raise ValueError(f"Util: Error building command for model '{model_name}': {e}") from e

    keys_to_cull = []
    for key in list(effective_config.keys()):
        if key in PRESERVED_EFFECTIVE_CONFIG_KEYS: continue
        is_redundant = any(key.endswith(s) for s in REDUNDANT_BLOCK_PATTERNS["suffixes"]) or \
                        any(key.startswith(p) for p in REDUNDANT_BLOCK_PATTERNS["prefixes"])
        if is_redundant and isinstance(effective_config.get(key), dict):
            keys_to_cull.append(key)
    for key in keys_to_cull:
        if verbose_logging: print(colour_util(f"Util: Removing top-level redundant block: {key}", Fore.MAGENTA))
        del effective_config[key]
        
    return effective_config

# --- Self-test execution block ---
if __name__ == "__main__":
    print(f"{Style.BRIGHT}--- Running config_utils.py self-test ---{Style.RESET_ALL}")
    
    current_file_path = Path(__file__).resolve()
    project_root_for_test = current_file_path.parent.parent 

    test_base_path = project_root_for_test / "config.base.yaml"
    test_override_path_arg = None 

    print(f"Base config path for test: {test_base_path}")
    print(f"Explicit override path for test: {test_override_path_arg}")
    print(f"Script dir for hostname override lookup (project root): {project_root_for_test}")

    if not test_base_path.exists():
        print(colour_util(f"Test SKIPPED: Base config '{test_base_path}' not found.", Fore.RED))
    else:
        try:
            print(colour_util("\n--- Testing config processing (verbose_logging=True) ---", Fore.CYAN))
            processed_config_result = generate_processed_config(
                base_config_path_arg=test_base_path, # <<< CORRECTED ARGUMENT NAME
                override_config_path_arg=test_override_path_arg,
                script_dir_for_overrides=project_root_for_test,
                verbose_logging=True
            )
            # ... rest of the self-test block remains the same ...
            print(f"\n{Style.BRIGHT}--- Processed Configuration Output (YAML) ---{Style.RESET_ALL}")
            print(yaml.dump(processed_config_result, indent=2, sort_keys=False, allow_unicode=True, Dumper=yaml.Dumper))
            
            print(f"\n{Style.BRIGHT}--- Basic Validation of Processed Config ---{Style.RESET_ALL}")
            if "models" in processed_config_result and isinstance(processed_config_result["models"], dict):
                num_models = len(processed_config_result['models'])
                print(colour_util(f"Found 'models' section with {num_models} entries.", Fore.GREEN))
                
                for model_name, model_data in processed_config_result['models'].items():
                    print(f"  Validating model: '{model_name}'")
                    if "cmd" in model_data and isinstance(model_data["cmd"], dict):
                        cmd_dict = model_data["cmd"]
                        print(colour_util(f"    'cmd' dictionary found.", Fore.GREEN))
                        for path_key_check in PATH_KEYS_IN_CMD:
                            if path_key_check in cmd_dict:
                                path_val = cmd_dict[path_key_check]
                                if not path_val or Path(path_val).is_absolute(): # Empty or Absolute
                                    print(colour_util(f"      'cmd.{path_key_check}': '{path_val}' (appears absolute or empty, OK).", Fore.GREEN))
                                else:
                                    print(colour_util(f"      'cmd.{path_key_check}': '{path_val}' (WARNING: not absolute).", Fore.YELLOW))
                    else:
                         print(colour_util(f"    Model '{model_name}': 'cmd' key is NOT a dictionary or missing.", Fore.RED))
                         
                    if "generated_cmd_str" in model_data and isinstance(model_data["generated_cmd_str"], str):
                         print(colour_util(f"    'generated_cmd_str' found.", Fore.GREEN))
                    else:
                         print(colour_util(f"    Model '{model_name}': 'generated_cmd_str' missing or not a string.", Fore.RED))
                         
                    if "sampling" not in model_data: 
                        print(colour_util(f"    'sampling' key correctly removed from top level.", Fore.GREEN))
                    else:
                         print(colour_util(f"    Model '{model_name}': 'sampling' key was NOT removed.", Fore.YELLOW))
                    break 
            else:
                print(colour_util("No 'models' section found or not a dict in processed config.", Fore.RED))

            removed_test_key = "COMMON_FLAGS" 
            if removed_test_key not in processed_config_result:
                 print(colour_util(f"Top-level block '{removed_test_key}' (if existed) correctly removed.", Fore.GREEN))
            else:
                 print(colour_util(f"Top-level block '{removed_test_key}' might still be present.", Fore.YELLOW))

        except Exception as e:
            print(colour_util(f"\n--- ERROR during config_utils self-test ---", Fore.RED))
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{Style.BRIGHT}--- config_utils.py self-test finished ---{Style.RESET_ALL}")