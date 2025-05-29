#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Get the directory where the script is located
ScriptDir=$(cd "$(dirname "$0")" && pwd)

# --- !!! ADJUST THESE PATHS AS NEEDED FOR YOUR MACOS SETUP !!! ---
# Assume the InstallDir is the parent directory of the script's directory
# Or set it explicitly, e.g., InstallDir="$HOME/llama_suite"
InstallDir=$(cd "$ScriptDir/" && pwd)

# llama-swap paths
LlamaSwapDir="$InstallDir/llama-swap"
# Adjust executable name if needed (likely no .exe on macOS)
LlamaSwapExePath="$LlamaSwapDir/llama-swap"
# Assuming config is relative to the script dir, like the original
ConfigYamlPath="$ScriptDir/model-configs/config-mac.yaml"

# Listen address for clients (Open WebUI, etc.)
LlamaSwapListenAddress=":8080"
# --- End Configuration ---

# --- Helper Functions ---
error_message() {
  echo "Error: $1" >&2
}

wait_for_enter() {
  # Only prompt if running interactively
  if [[ -t 0 ]]; then
      read -p "Press Enter to exit"
  fi
}

# --- Validate paths ---
if [[ ! -d "$LlamaSwapDir" ]]; then
    error_message "llama-swap directory not found at: $LlamaSwapDir"
    wait_for_enter
    exit 1
fi
if [[ ! -f "$LlamaSwapExePath" ]]; then
    error_message "llama-swap executable not found at: $LlamaSwapExePath"
    error_message "Please check the LlamaSwapExePath variable in this script."
    wait_for_enter
    exit 1
fi
if [[ ! -x "$LlamaSwapExePath" ]]; then
    error_message "llama-swap is not executable at: $LlamaSwapExePath"
    error_message "You might need to run: chmod +x $LlamaSwapExePath"
    wait_for_enter
    exit 1
fi
# Adjust config path validation based on new variable
if [[ ! -f "$ConfigYamlPath" ]]; then
    error_message "Configuration file not found at: $ConfigYamlPath"
    error_message "Please check the ConfigYamlPath variable in this script."
    wait_for_enter
    exit 1
fi

# --- Check for fswatch (needed for auto-restart) ---
if ! command -v fswatch &> /dev/null; then
    error_message "'fswatch' command not found. Auto-restart requires fswatch."
    error_message "To install fswatch on macOS: brew install fswatch"
    wait_for_enter
    exit 1
fi
# Check for timeout command (gtimeout if coreutils installed, otherwise system timeout)
timeout_cmd="timeout"
if ! command -v timeout &> /dev/null; then
    if command -v gtimeout &> /dev/null; then
        timeout_cmd="gtimeout"
        echo "Info: Using 'gtimeout' (from brew coreutils) for wait timeout."
    else
        error_message "'timeout' or 'gtimeout' command not found. Cannot guarantee clean shutdown on hangs."
        error_message "Consider installing GNU coreutils: brew install coreutils"
        # Proceed without timeout? Risky. Let's exit.
        wait_for_enter
        exit 1
    fi
fi


# --- Build argument list ---
LlamaSwapArgs=(
    "--config" "$ConfigYamlPath"
    "-listen" "$LlamaSwapListenAddress"
    # Add '-vv' here for verbose logging if needed
    # "-vv"
)

# --- Colors ---
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Pretty banner ---
echo -e "${CYAN}-----------------------------------------------------${NC}"
echo -e "${CYAN}Starting llama-swap (auto-restart enabled using fswatch)${NC}"
echo -e "${CYAN}Executable: $LlamaSwapExePath${NC}"
echo -e "${CYAN}Arguments : ${LlamaSwapArgs[*]}${NC}" # Show arguments
echo -e "${CYAN}Config    : $ConfigYamlPath${NC}"
echo -e "${CYAN}Watching  : $ConfigYamlPath${NC}"
echo -e "${CYAN}Listening : $LlamaSwapListenAddress${NC}"
DisplayPort="${LlamaSwapListenAddress##*:}"
echo -e "${CYAN}Open WebUI → http://localhost:$DisplayPort/v1${NC}"
echo -e "${CYAN}-----------------------------------------------------${NC}"
echo -e "${YELLOW}--- llama-swap OUTPUT (live & restarts on save) ---${NC}"

# --- Process Management for Auto-Restart ---
child_pid=""

# Function to run llama-swap in the background
run_llama_swap() {
    echo "--- Starting llama-swap process ---"

    # --- STRATEGY: Launch directly, don't cd in subshell ---
    # This assumes llama-swap finds its resources correctly,
    # or that paths in config.yaml are absolute or relative to InstallDir/LlamaSwapDir.
    # Make sure LlamaSwapExePath is correct!
    echo "Executing from script dir: $LlamaSwapExePath ${LlamaSwapArgs[*]} &"
    # Run in the background directly from the script's current PWD
    # We rely on the executable or config handling paths correctly.
    "$LlamaSwapExePath" "${LlamaSwapArgs[@]}" &

    # ADDED: Tiny sleep again, just in case
    sleep 0.1

    _pid_raw=$! # Capture PID immediately after '&' and sleep
    echo "DEBUG: Raw value of \$! is '$_pid_raw'" # Added Debug

    child_pid="$_pid_raw" # Assign it

    # Check if PID was captured AND is a valid number > 0
    # Use standard integer comparison -gt
    if ! [[ "$child_pid" =~ ^[0-9]+$ ]] || ! [ "$child_pid" -gt 0 ]; then
         echo -e "${RED}--- Failed to capture valid PID after launch! PID='$child_pid' ---${NC}"
         child_pid="" # Clear invalid PID

         # --- Fallback: Use pgrep to find the process ---
         # This is less reliable as it depends on the process name.
         _exe_name=$(basename "$LlamaSwapExePath")
         echo "Attempting pgrep fallback for process name: $_exe_name"
         # -n gets the newest process matching the name
         _pgrep_pid=$(pgrep -n "$_exe_name")

         if [[ "$_pgrep_pid" =~ ^[0-9]+$ ]] && [ "$_pgrep_pid" -gt 0 ]; then
            echo "DEBUG: pgrep found potential PID: $_pgrep_pid"
            # Try to verify this PID is the correct command
            _cmd_check=$(ps -p "$_pgrep_pid" -o command=)
            echo "DEBUG: Command for PID $_pgrep_pid: $_cmd_check"
            # Basic check - does the command contain the exe path?
            if [[ "$_cmd_check" == *"$LlamaSwapExePath"* ]]; then
                 echo -e "${YELLOW}--- Fallback PID capture SUCCESS via pgrep: $_pgrep_pid ---${NC}"
                 child_pid="$_pgrep_pid" # Assign pgrep PID
            else
                 echo -e "${RED}--- Fallback PID check FAILED: command '$_cmd_check' doesn't match '$LlamaSwapExePath' ---${NC}"
                 child_pid=""
            fi
         else
            echo -e "${RED}--- Fallback PID capture FAILED: pgrep found nothing or invalid PID '$_pgrep_pid'. ---${NC}"
            child_pid=""
         fi

         # If child_pid is still empty after fallback, fail the function
         if [[ -z "$child_pid" ]]; then
            sleep 5 # Prevent rapid failure loop
            return 1
         fi
         # --- End Fallback ---

    else
         echo "--- PID capture successful via \$!: $child_pid ---"
    fi

    # Now we have a child_pid (either from $! or pgrep)
    echo "llama-swap process associated with PID: $child_pid"

    # Give it a moment to potentially fail early (like binding error)
    sleep 1.5

    # Check if the captured/found PID is still running
    if ! kill -0 "$child_pid" 2>/dev/null; then
         echo -e "${RED}--- llama-swap (PID: $child_pid) seems to have exited OR was never the correct process. Check logs above. ---${NC}"
         child_pid="" # Clear the potentially incorrect/dead PID
         sleep 5      # Prevent rapid failure loop
         return 1
    fi

    echo "--- llama-swap (PID: $child_pid) confirmed running. ---"
    return 0
}

# Function to gracefully stop llama-swap
stop_llama_swap() {
    # Check if PID exists and the process is likely running
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        echo "--- Stopping llama-swap process (PID: $child_pid) ---"
        # Send SIGTERM (graceful shutdown)
        kill -SIGTERM "$child_pid"

        echo "--- Waiting up to 10 seconds for PID $child_pid to exit gracefully... ---"
        # Wait for the process to terminate, with a timeout
        # Use the determined timeout command
        if $timeout_cmd 10s wait "$child_pid" 2>/dev/null; then
            echo "--- Process $child_pid terminated gracefully. ---"
        else
            # If wait timed out or process was already gone (wait exits non-zero)
            echo -e "${YELLOW}--- Process $child_pid did not terminate gracefully within 10s (or was already gone), checking if it needs SIGKILL... ---${NC}"
            # Double check if it's still running before sending SIGKILL
            if kill -0 "$child_pid" 2>/dev/null; then
                 echo -e "${RED}--- Sending SIGKILL to PID $child_pid. ---${NC}"
                 kill -SIGKILL "$child_pid"
                 sleep 1 # Give OS a moment after SIGKILL
            else
                 echo "--- Process $child_pid already gone. ---"
            fi
        fi
        # Give the OS a little more time to release the network port
        echo "--- Allowing 1 second for port release... ---"
        sleep 1
    elif [[ -n "$child_pid" ]]; then
        echo "--- Process $child_pid was not running or PID was invalid. ---"
    fi
    # Clear the PID regardless
    child_pid=""
}

# Trap signals (Ctrl+C, Terminate) for cleanup
# Use EXIT trap to ensure cleanup regardless of how script exits
trap 'echo; echo "--- Script exiting, cleaning up llama-swap ---"; stop_llama_swap; echo -e "${CYAN}llama-swap monitoring stopped.${NC}"; wait_for_enter' EXIT SIGINT SIGTERM

# Initial start
if ! run_llama_swap; then
    echo -e "${RED}Initial start failed. Exiting.${NC}"
    # Trap will run cleanup
    exit 1
fi

# Use fswatch to monitor the config file
# -o groups events, reducing restarts for rapid saves
# --event Updated might be too restrictive, try monitoring any write/rename/attribute change
echo "Watching $ConfigYamlPath for changes..."
fswatch -o "$ConfigYamlPath" | while read -r file event; do
    echo "--- Config file '$file' changed (event: $event) ---"
    stop_llama_swap
    # No need for extra sleep here, stop_llama_swap includes one
    run_llama_swap
done

echo "fswatch loop exited unexpectedly."
# Trap should handle cleanup