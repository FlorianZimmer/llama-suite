#!/usr/bin/env bash

# Stop on errors
# set -e # Be careful with this, as some checks rely on non-zero exit codes

#==============================================================================
# Script Configuration & Defaults
#==============================================================================
DEFAULT_CONFIG_PATH="../model-configs/config-mac.yaml"
DEFAULT_OUTPUT_FILE="benchmark_results_memory.csv" # Changed default to CSV
DEFAULT_QUESTION="Why is the sky blue?"
DEFAULT_HEALTH_POLL_INTERVAL=2
DEFAULT_HEALTH_TIMEOUT=60 # Default if not in config or overridden
OUTPUT_FORMAT="csv" # Hardcoded to CSV for Bash version

# Script parameters (will be set by parsing)
CONFIG_PATH=""
OUTPUT_FILE=""
QUESTION=""
HEALTH_CHECK_POLL_INTERVAL=""
OVERRIDE_HEALTH_CHECK_TIMEOUT=""
MODEL_TO_TEST=""
HEALTH_CHECK_TIMEOUT_SECONDS="" # Final calculated value

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TEMP_DIR=$(mktemp -d) # Create a temporary directory for logs

# Cleanup function to remove temp files and kill stray processes
cleanup() {
    echo "Cleaning up..."
    rm -rf "$TEMP_DIR"
    # Add any specific process killing logic if needed, though stopping them individually is better
    echo "Cleanup finished."
}
trap cleanup EXIT INT TERM

# ANSI Color Codes
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
COLOR_RED="\033[0;31m"
COLOR_CYAN="\033[0;36m"

#==============================================================================
# Helper Functions
#==============================================================================

# --- Log messages ---
log_info() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $1"; }
log_warn() { echo -e "${COLOR_YELLOW}$(date '+%Y-%m-%d %H:%M:%S') [WARN] $1${COLOR_RESET}" >&2; }
log_error() { echo -e "${COLOR_RED}$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1${COLOR_RESET}" >&2; }
log_success() { echo -e "${COLOR_GREEN}$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] $1${COLOR_RESET}"; }
log_verbose() { [[ "$VERBOSE" == "true" ]] && echo -e "$(date '+%Y-%m-%d %H:%M:%S') [VERBOSE] $1"; }
log_cyan() { echo -e "${COLOR_CYAN}$1${COLOR_RESET}"; }
log_hdr() { echo -e "${COLOR_CYAN}============================================================${COLOR_RESET}"; }
log_subhdr() { echo -e "${COLOR_CYAN}------------------------------------------------------------${COLOR_RESET}"; }

# --- Check for required commands ---
check_dependencies() {
    local missing=0
    for cmd in yq jq curl bc pgrep pkill date mktemp dirname basename grep sed awk; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' not found. Please install it."
            if [[ "$cmd" == "yq" || "$cmd" == "jq" || "$cmd" == "bc" ]]; then
                log_error "  Try: brew install $cmd"
            fi
            missing=1
        fi
    done
    if [[ $missing -eq 1 ]]; then
        exit 1
    fi
    # Check yq version (v4+ is recommended for consistent syntax)
    if yq --version 2>&1 | grep -q 'version [1-3]'; then
       log_warn "yq version might be old (v1-v3). Syntax used here assumes v4+. Consider 'brew upgrade yq'."
    fi
}

# --- Parse Memory String (Returns GB) ---
# Input: String like "1234.5 MiB" or "2.1 GiB" etc.
# Output: Value in GB (float) or "null" on failure
parse_memory_string() {
    local mem_line="$1"
    local value=""
    local unit=""
    local value_mb=0

    # Regex to capture value and unit (handles comma or period decimal separator)
    if [[ "$mem_line" =~ ([0-9]+([.,][0-9]+)?)[[:space:]]*([KMGT]i?B) ]]; then
        value=$(echo "${BASH_REMATCH[1]}" | sed 's/,/./') # Normalize decimal point
        unit=$(echo "${BASH_REMATCH[3]}" | tr '[:lower:]' '[:upper:]') # Uppercase unit

        # Use bc for floating point calculations
        case "$unit" in
            'KB')  value_mb=$(echo "scale=10; $value / 1000" | bc) ;; # Using 1000 for KB/MB/GB
            'KIB') value_mb=$(echo "scale=10; $value / 1024" | bc) ;; # Using 1024 for KiB/MiB/GiB
            'MB')  value_mb=$value ;;
            'MIB') value_mb=$value ;;
            'GB')  value_mb=$(echo "scale=10; $value * 1000" | bc) ;;
            'GIB') value_mb=$(echo "scale=10; $value * 1024" | bc) ;;
            'TB')  value_mb=$(echo "scale=10; $value * 1000 * 1000" | bc) ;;
            'TIB') value_mb=$(echo "scale=10; $value * 1024 * 1024" | bc) ;;
            *)     echo "null"; return ;;
        esac

        if (( $(echo "$value_mb != 0" | bc -l) )); then
            # Calculate GB (using 1024 definition for GB from MiB) and format to 2 decimal places
            local value_gb=$(echo "scale=10; $value_mb / 1024" | bc)
            printf "%.2f\n" "$value_gb"
        else
            echo "0.00"
        fi
    else
        echo "null"
    fi
}

# --- Parse Llama Server Command String ---
# Input: Full command string, Model Name (for logging)
# Output: Echoes "EXECUTABLE|ARGUMENTS" or "" on failure
parse_llama_server_command() {
    local full_command="$1"
    local model_name="$2"
    local executable=""
    local arguments=""

    # Simple approach: Assume first word is executable, rest are args.
    # This might fail if the executable path contains spaces NOT enclosed in quotes.
    # PowerShell's regex was more robust for Windows paths.
    read -r executable arguments <<< "$full_command"

    if [[ -z "$executable" ]]; then
        log_warn "Parse-LlamaServerCommand: Failed to extract server executable path for '$model_name' from '$full_command'."
        echo ""
        return 1
    fi

    # Basic check if executable seems plausible (e.g., exists)
    # This check is done later during start, so just return parsed parts here.
    echo "${executable}|${arguments}"
    return 0
}

# --- Start Llama Server Process ---
# Input: Executable path, Arguments string, Model Name
# Output: Echoes "PID|STDOUT_LOG|STDERR_LOG" or "" on failure
start_llama_server() {
    local executable="$1"
    local arguments_str="$2"
    local model_name="$3"
    local result_file="$4" # New argument for the output file path

    if [[ -z "$result_file" ]]; then
        log_error "Start-LlamaServer: Error - Result file path argument is empty for model '$model_name'." >&2
        return 1 # Indicate failure
    fi

    local -a arguments_array
    read -ra arguments_array <<< "$arguments_str"

    local stdout_log="${TEMP_DIR}/${model_name}_stdout.log"
    local stderr_log="${TEMP_DIR}/${model_name}_stderr.log"
    local pid=""

    # Ensure result file starts empty or non-existent
    rm -f "$result_file"

    # Send logging to STDERR
    log_verbose "Start-LlamaServer: Starting '$model_name'" >&2
    log_verbose "  Executable: '$executable'" >&2
    log_verbose "  Arguments: '${arguments_array[*]}'" >&2
    log_verbose "  Stdout Log: $stdout_log" >&2
    log_verbose "  Stderr Log: $stderr_log" >&2

    # Execute in background, redirecting stdout and stderr
    "$executable" "${arguments_array[@]}" > "$stdout_log" 2> "$stderr_log" &
    pid=$!

    # Brief sleep to allow immediate failure detection
    sleep 0.5
    if ! kill -0 "$pid" 2>/dev/null; then
        # Send logging to STDERR
        log_error "Start-LlamaServer: Failed to start server process for '$model_name'. Check logs:" >&2
        log_error "  Stderr: $stderr_log" >&2
        # Show stderr content on STDERR
        cat "$stderr_log" >&2
        # Write empty string to result file indicating failure
        echo "" > "$result_file"
        return 1 # Indicate failure with exit code
    fi

    # Send logging to STDERR
    log_verbose "Start-LlamaServer: Process '$model_name' started with PID $pid." >&2
    # Write ONLY the result string to the specified result file
    echo "${pid}|${stdout_log}|${stderr_log}" > "$result_file"
    return 0 # Indicate success with exit code
}

# --- Wait for Server Health ---
# Input: PID, Health Check URL, Timeout (s), Poll Interval (s), Model Name
# Output: 0 on success, 1 on failure/timeout
wait_llama_server_healthy() {
    local pid="$1"
    local health_check_url="$2"
    local timeout_seconds="$3"
    local poll_interval_seconds="$4"
    local model_name="$5"

    log_info "  Waiting for server '$model_name' (PID: $pid) to become healthy at $health_check_url ..."
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout_seconds))
    local current_time=$(date +%s)
    local healthy=1 # 1 means not healthy yet (like exit code)

    while [[ $current_time -lt $end_time ]]; do
        # Check if process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            log_warn "Wait-LlamaServerHealthy: Server process '$model_name' (PID: $pid) exited prematurely."
            return 1 # Failure
        fi

        # Perform health check
        local http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$poll_interval_seconds" "$health_check_url")

        if [[ "$http_code" -eq 200 ]]; then
            log_success "  Server '$model_name' is healthy."
            healthy=0 # Success
            break
        else
            log_verbose "Wait-LlamaServerHealthy: Health check for '$model_name' failed (HTTP code: $http_code). Retrying..."
        fi

        sleep "$poll_interval_seconds"
        current_time=$(date +%s)
    done

    if [[ $healthy -ne 0 ]]; then
        log_warn "Wait-LlamaServerHealthy: Server '$model_name' did not become healthy within $timeout_seconds seconds."
    fi

    return $healthy
}

# --- Stop Llama Server Process ---
# Input: PID, Model Name
stop_llama_server() {
    local pid="$1"
    local model_name="$2"

    if [[ -z "$pid" ]]; then
        log_verbose "Stop-LlamaServer: No PID provided for '$model_name'."
        return
    fi

    # Check if the process exists
    if ps -p "$pid" > /dev/null; then
        log_info "  Stopping server process '$model_name' (PID: $pid)..."
        # Try graceful termination first
        kill "$pid"
        sleep 2 # Give it time to shut down

        # Force kill if still running
        if ps -p "$pid" > /dev/null; then
            log_warn "  Process '$model_name' (PID: $pid) did not stop gracefully, forcing kill..."
            kill -9 "$pid"
            sleep 1
        fi

        # Verify it's stopped
        if ps -p "$pid" > /dev/null; then
             log_error "  Failed to stop process '$model_name' (PID: $pid) even with kill -9."
        else
             log_verbose "  Process '$model_name' (PID: $pid) stopped."
        fi
    else
        log_verbose "Stop-LlamaServer: Server process '$model_name' (PID: $pid) already stopped."
    fi
}

# --- Parse Memory From Log File ---
# Input: Stderr Log Path, Model Name
# Output: Echoes "GPU_GB|CPU_GB|STATUS" to STDOUT (STATUS=Success/Failed/Parse Error) values can be "null". Logs to STDERR.
parse_memory_from_log() {
    local stderr_log_path="$1"
    local model_name="$2"
    local gpu_mem_gb="null"
    local cpu_mem_gb="null"
    local scan_status="Failed"

    if [[ ! -f "$stderr_log_path" ]]; then
        log_warn "Parse-MemoryFromLog: Stderr log file not found for '$model_name': '$stderr_log_path'" >&2
        echo "null|null|Failed"
        return 1
    fi

    log_verbose "Parse-MemoryFromLog: Reading stderr log for '$model_name': $stderr_log_path" >&2
    sleep 1 # Give logs time to flush

    local gpu_capture cpu_capture

    # *** ADD LC_ALL=C before sed commands ***
    gpu_capture=$(LC_ALL=C sed -n -E -e '/load_tensors:[[:space:]]+(Metal_Mapped|CUDA[0-9]*)[[:space:]]+model buffer size[[:space:]]*=[[:space:]]*/I { s/.*=[[:space:]]*//p; q; }' "$stderr_log_path")
    cpu_capture=$(LC_ALL=C sed -n -E -e '/load_tensors:[[:space:]]+CPU_Mapped[[:space:]]+model buffer size[[:space:]]*=[[:space:]]*/I { s/.*=[[:space:]]*//p; q; }' "$stderr_log_path")
    # *** END CHANGE ***

    # Check if sed commands themselves failed (e.g., returned non-zero exit code)
    # This might not be strictly necessary if the capture variable check below is sufficient
    # local sed_rc=$?
    # if [[ $sed_rc -ne 0 ]]; then
    #    log_warn "Parse-MemoryFromLog: Sed command failed with exit code $sed_rc for '$model_name'." >&2
    # fi

    # Trim potential leading/trailing whitespace
    gpu_capture=$(echo "$gpu_capture" | awk '{$1=$1};1')
    cpu_capture=$(echo "$cpu_capture" | awk '{$1=$1};1')

    if [[ -n "$gpu_capture" ]]; then
        gpu_mem_gb=$(parse_memory_string "$gpu_capture")
        if [[ "$gpu_mem_gb" != "null" ]]; then
            scan_status="Success"
        else
            log_warn "Parse-MemoryFromLog: Failed to parse GPU value for '$model_name' from capture: [$gpu_capture]" >&2
            scan_status="Parse Error"
        fi
    else
        log_verbose "Parse-MemoryFromLog: Did not find GPU (Metal/CUDA) memory pattern for '$model_name'." >&2
    fi

    if [[ -n "$cpu_capture" ]]; then
        cpu_mem_gb=$(parse_memory_string "$cpu_capture")
        if [[ "$cpu_mem_gb" != "null" ]]; then
            [[ "$scan_status" != "Parse Error" ]] && scan_status="Success"
        else
            log_warn "Parse-MemoryFromLog: Failed to parse CPU value for '$model_name' from capture: [$cpu_capture]" >&2
             scan_status="Parse Error"
        fi
    else
        log_verbose "Parse-MemoryFromLog: Did not find CPU memory pattern for '$model_name'." >&2
    fi

    # Log result summary to STDERR
    if [[ "$scan_status" == "Success" ]]; then
        local gpu_display=${gpu_mem_gb:-?}
        local cpu_display=${cpu_mem_gb:-?}
        log_cyan "  Parsed Memory for '$model_name': GPU=${gpu_display} GB, CPU=${cpu_display} GB" >&2
    elif [[ "$scan_status" == "Parse Error" ]]; then
         log_warn "Parse-MemoryFromLog: Failed to parse one or more memory values correctly for '$model_name'." >&2
    else # Failed
        log_warn "Parse-MemoryFromLog: Failed to find memory values for '$model_name'." >&2
    fi

    # Echo final result string to STDOUT ONLY
    echo "${gpu_mem_gb}|${cpu_mem_gb}|${scan_status}"
    return 0
}

#==============================================================================
# Argument Parsing
#==============================================================================

usage() {
    echo "Usage: $0 [-c <config_path>] [-o <output_file>] [-q <question>] [-p <poll_interval>] [-t <health_timeout>] [-m <model_name>] [-h] [-v]"
    echo "  -c : Path to the llama-swap configuration YAML file (default: $DEFAULT_CONFIG_PATH)"
    echo "  -o : Path for the output CSV file (default: $DEFAULT_OUTPUT_FILE)"
    echo "  -q : Question/prompt for benchmarking (default: '$DEFAULT_QUESTION')"
    echo "  -p : Health check poll interval in seconds (default: $DEFAULT_HEALTH_POLL_INTERVAL)"
    echo "  -t : Override health check timeout in seconds (default: from config or $DEFAULT_HEALTH_TIMEOUT)"
    echo "  -m : Benchmark only this specific model name (must match config key)"
    echo "  -v : Verbose output"
    echo "  -h : Display this help message"
    exit 1
}

# Initialize parameters to defaults
CONFIG_PATH="$DEFAULT_CONFIG_PATH"
OUTPUT_FILE="$DEFAULT_OUTPUT_FILE"
QUESTION="$DEFAULT_QUESTION"
HEALTH_CHECK_POLL_INTERVAL="$DEFAULT_HEALTH_POLL_INTERVAL"
OVERRIDE_HEALTH_CHECK_TIMEOUT=""
MODEL_TO_TEST=""
VERBOSE="false"

# Parse options
while getopts ":c:o:q:p:t:m:vh" opt; do
    case ${opt} in
        c ) CONFIG_PATH=$OPTARG;;
        o ) OUTPUT_FILE=$OPTARG;;
        q ) QUESTION=$OPTARG;;
        p ) HEALTH_CHECK_POLL_INTERVAL=$OPTARG;;
        t ) OVERRIDE_HEALTH_CHECK_TIMEOUT=$OPTARG;;
        m ) MODEL_TO_TEST=$OPTARG;;
        v ) VERBOSE="true";;
        h ) usage;;
        \? ) log_error "Invalid option: -$OPTARG"; usage;;
        : ) log_error "Invalid option: -$OPTARG requires an argument"; usage;;
    esac
done
shift $((OPTIND -1))

#==============================================================================
# Main Script Logic
#==============================================================================

SCRIPT_START_TIME_S=$(date +%s)
log_info "Starting LLM Benchmark Script..."

# --- Check Dependencies ---
check_dependencies

# --- Calculate Full Paths ---
# Resolve config path relative to script dir if it's not absolute
if [[ ! "$CONFIG_PATH" = /* ]]; then
    CONFIG_FULL_PATH="$SCRIPT_DIR/$CONFIG_PATH"
else
    CONFIG_FULL_PATH="$CONFIG_PATH"
fi
# Resolve output path relative to script dir if it's not absolute
if [[ ! "$OUTPUT_FILE" = /* ]]; then
    OUTPUT_FULL_PATH="$SCRIPT_DIR/$OUTPUT_FILE"
else
    OUTPUT_FULL_PATH="$OUTPUT_FILE"
fi
# Ensure output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT_FULL_PATH")
mkdir -p "$OUTPUT_DIR" || { log_error "Failed to create output directory: $OUTPUT_DIR"; exit 1; }

log_info "Config File: $CONFIG_FULL_PATH"
log_info "Output File: $OUTPUT_FULL_PATH (Format: CSV)"
log_info "Test Question: '$QUESTION'"
log_warn "The llama-server process will run in the background. Ensure you have permissions."

# --- Load Config ---
log_info "Loading configuration..."
if [[ ! -f "$CONFIG_FULL_PATH" ]]; then
    log_error "Configuration file not found at '$CONFIG_FULL_PATH'."
    exit 1
fi

# Check if yq can parse the file (basic validation)
if ! yq -e '.' "$CONFIG_FULL_PATH" > /dev/null; then
    log_error "Failed to parse configuration file '$CONFIG_FULL_PATH' with yq."
    exit 1
fi

# Extract model keys and potentially the default timeout
ALL_MODEL_KEYS=$(yq -e '.models | keys | .[]' "$CONFIG_FULL_PATH")
if [[ -z "$ALL_MODEL_KEYS" ]]; then
     log_error "No models found under 'models:' section in the configuration file."
     exit 1
fi
CONFIG_HEALTH_TIMEOUT=$(yq -e '.healthCheckTimeout // null' "$CONFIG_FULL_PATH") # Use // for default null if not present

log_info "Configuration loaded successfully."

# --- Determine Health Check Timeout ---
if [[ -n "$OVERRIDE_HEALTH_CHECK_TIMEOUT" ]]; then
    HEALTH_CHECK_TIMEOUT_SECONDS="$OVERRIDE_HEALTH_CHECK_TIMEOUT"
elif [[ "$CONFIG_HEALTH_TIMEOUT" != "null" && "$CONFIG_HEALTH_TIMEOUT" -gt 0 ]]; then
    HEALTH_CHECK_TIMEOUT_SECONDS="$CONFIG_HEALTH_TIMEOUT"
else
    HEALTH_CHECK_TIMEOUT_SECONDS="$DEFAULT_HEALTH_TIMEOUT"
fi
# Validate timeout is a positive integer
if ! [[ "$HEALTH_CHECK_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    log_warn "Invalid health check timeout value '$HEALTH_CHECK_TIMEOUT_SECONDS'. Using default: $DEFAULT_HEALTH_TIMEOUT seconds."
    HEALTH_CHECK_TIMEOUT_SECONDS=$DEFAULT_HEALTH_TIMEOUT
fi
# Validate poll interval
if ! [[ "$HEALTH_CHECK_POLL_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
     log_warn "Invalid health check poll interval '$HEALTH_CHECK_POLL_INTERVAL'. Using default: $DEFAULT_HEALTH_POLL_INTERVAL seconds."
     HEALTH_CHECK_POLL_INTERVAL=$DEFAULT_HEALTH_POLL_INTERVAL
fi

log_info "Health Check Timeout: $HEALTH_CHECK_TIMEOUT_SECONDS seconds"
log_info "Health Check Poll Interval: $HEALTH_CHECK_POLL_INTERVAL seconds"

# --- Initialize Result Storage ---
declare -A MEMORY_SCAN_RESULTS # Associative array: Key=ModelName, Value="GPU_GB|CPU_GB|Status"
# CSV Header will be written later

# --- Determine Models to Benchmark ---
MODELS_TO_TEST_LIST=()
if [[ -n "$MODEL_TO_TEST" ]]; then
    # Check if the specified model exists in the config
    if yq -e ".models[\"$MODEL_TO_TEST\"]" "$CONFIG_FULL_PATH" > /dev/null 2>&1; then
        MODELS_TO_TEST_LIST=("$MODEL_TO_TEST")
        log_info "Specific model selected: $MODEL_TO_TEST (Memory scan will be skipped)"
    else
        log_error "The specified model '$MODEL_TO_TEST' was not found in the configuration file."
        log_info "Available models:"
        echo "$ALL_MODEL_KEYS" # Print each key on a new line
        exit 1
    fi
else
    # Convert multi-line output of yq to bash array
    mapfile -t MODELS_TO_TEST_LIST < <(echo "$ALL_MODEL_KEYS")
    log_info "All models selected for memory scan and benchmarking."
fi
log_info "Models to process: ${MODELS_TO_TEST_LIST[*]}"

#==============================================================================
# PASS 1: Memory Scan
#==============================================================================
PERFORM_MEMORY_SCAN=true

if [[ "$PERFORM_MEMORY_SCAN" == "true" ]]; then
    log_hdr
    log_info "Starting PASS 1: Memory Scan for ${#MODELS_TO_TEST_LIST[@]} models..."
    log_hdr

    for model_name in "${MODELS_TO_TEST_LIST[@]}"; do
        log_subhdr
        log_info "Memory Scan: Processing '$model_name'"

        # Extract model config using yq - could be slow if config is huge and called repeatedly
        model_proxy=$(yq -e ".models[\"$model_name\"].proxy // \"\"" "$CONFIG_FULL_PATH")
        model_cmd=$(yq -e ".models[\"$model_name\"].cmd // \"\"" "$CONFIG_FULL_PATH")

        if [[ -z "$model_proxy" || -z "$model_cmd" ]]; then
            log_warn "Memory Scan: Skipping '$model_name' - Invalid or incomplete config (missing proxy or cmd)."
            MEMORY_SCAN_RESULTS["$model_name"]="null|null|Skipped"
            continue
        fi

        parsed_command_output=$(parse_llama_server_command "$model_cmd" "$model_name")
        if [[ -z "$parsed_command_output" ]]; then
             log_warn "Memory Scan: Skipping '$model_name' - Cannot parse command."
             MEMORY_SCAN_RESULTS["$model_name"]="null|null|Cmd Error"
             continue
        fi
        IFS='|' read -r server_executable server_arguments <<< "$parsed_command_output"

        # Check if executable exists and is executable
         if [[ ! -x "$server_executable" ]]; then
              log_warn "Memory Scan: Skipping '$model_name' - Executable not found or not executable at '$server_executable'."
              MEMORY_SCAN_RESULTS["$model_name"]="null|null|Exe Error"
              continue
         fi

        server_pid=""      # Reset variables for this iteration
        stdout_log=""
        stderr_log=""
        memory_result_str=""
        current_scan_status="Failed"

        # Create a temporary file path for the start result
        start_result_file="${TEMP_DIR}/${model_name}_start_result_p1.txt"

        # Call start_llama_server, passing the result file path. Check its exit code.
        if start_llama_server "$server_executable" "$server_arguments" "$model_name" "$start_result_file"; then
            # start_llama_server succeeded (returned 0), read the result file
            start_output=""
            if [[ -f "$start_result_file" ]]; then
                start_output=$(<"$start_result_file") # Read file content
            fi

            if [[ -n "$start_output" ]]; then
                # Parse PID from the file content
                IFS='|' read -r server_pid stdout_log stderr_log <<< "$start_output"

                # *** PID VALIDATION (Crucial) ***
                if ! [[ "$server_pid" =~ ^[0-9]+$ ]]; then
                    log_error "Pass 1: Invalid PID parsed from result file for '$model_name'. Content: [$start_output]" >&2
                    current_scan_status="Start Error (Bad PID)"
                    MEMORY_SCAN_RESULTS["$model_name"]="null|null|$current_scan_status"
                    server_pid="" # Ensure PID is invalid
                else
                    # --- PID is valid, proceed with health check ---
                    log_verbose "Pass 1: Parsed Valid PID: $server_pid" >&2

                    if [[ ! "$model_proxy" =~ ^https?:// ]]; then health_url="http://${model_proxy}/health"; else health_url="${model_proxy}/health"; fi

                    if wait_llama_server_healthy "$server_pid" "$health_url" "$HEALTH_CHECK_TIMEOUT_SECONDS" "$HEALTH_CHECK_POLL_INTERVAL" "$model_name"; then
                        # Server healthy, parse memory
                        memory_result_str=$(parse_memory_from_log "$stderr_log" "$model_name")
                        log_verbose "parse_memory_from_log output captured: [$memory_result_str]" >&2 # Debug

                        # Use read directly, it will create the variables
                        # Initialize to empty first to handle cases where read might partially fail
                        tmp_gpu="" tmp_cpu="" tmp_status=""
                        IFS='|' read -r tmp_gpu tmp_cpu tmp_status <<< "$memory_result_str"
                        # Check if status was read correctly (basic check)
                        if [[ -z "$tmp_status" ]]; then
                            log_warn "Pass 1: Failed to correctly parse memory results string: [$memory_result_str]" >&2
                            # Store failure status
                            MEMORY_SCAN_RESULTS["$model_name"]="null|null|Parse Error (Read)"
                            log_verbose "Stored in MEMORY_SCAN_RESULTS for $model_name (Parse Error): [null|null|Parse Error (Read)]" >&2
                        else
                            # Store the parsed values
                            MEMORY_SCAN_RESULTS["$model_name"]="$tmp_gpu|$tmp_cpu|$tmp_status"
                            log_verbose "Stored in MEMORY_SCAN_RESULTS for $model_name ($tmp_status): [$tmp_gpu|$tmp_cpu|$tmp_status]" >&2
                        fi
                    fi # End health check

                fi # End PID validation
            else
                # start_llama_server succeeded but result file was empty? Should not happen with current logic.
                log_error "Pass 1: start_llama_server succeeded but result file was empty for '$model_name'." >&2
                current_scan_status="Start Error (Empty Result)"
                MEMORY_SCAN_RESULTS["$model_name"]="null|null|$current_scan_status"
                server_pid="" # No valid PID
            fi # End checking start_output
        else
            # start_llama_server failed (returned non-zero exit code)
            # Error message should have been printed by the function itself to stderr
            current_scan_status="Start Error"
            MEMORY_SCAN_RESULTS["$model_name"]="null|null|$current_scan_status"
            log_verbose "Stored in MEMORY_SCAN_RESULTS for $model_name ($current_scan_status): [null|null|$current_scan_status]" >&2
            server_pid="" # No PID to stop
        fi # End checking start_llama_server exit code

        # Always try to stop the server if PID is valid
        if [[ -n "$server_pid" && "$server_pid" =~ ^[0-9]+$ ]]; then # Check validity again
            stop_llama_server "$server_pid" "$model_name"
        elif [[ -n "$server_pid" ]]; then # Log if PID was set but invalid
            log_warn "Pass 1: Attempted to stop server for $model_name, but PID was invalid: [$server_pid]" >&2
        fi
        # Clean up the start result file
        rm -f "$start_result_file"

    done # End foreach model in Pass 1

    log_hdr
    log_info "PASS 1: Memory Scan Complete."
    log_hdr
    sleep 1
else
    log_info "Skipping PASS 1: Memory Scan because a specific model was requested via -m."
fi


#==============================================================================
# PASS 2: Benchmark
#==============================================================================
log_hdr
log_info "Starting PASS 2: Benchmarking Selected Models..."
log_hdr

# --- Prepare Output CSV File ---
CSV_HEADER="ModelName,Timestamp,Status,GpuMemoryGB,CpuMemoryGB,DurationSeconds,TokensPerSecond,PromptTokens,CompletionTokens,TotalTokens,ProxyUrl,Error"
echo "$CSV_HEADER" > "$OUTPUT_FULL_PATH" || { log_error "Failed to write header to output file: $OUTPUT_FULL_PATH"; exit 1; }
log_info "Results will be appended to: $OUTPUT_FULL_PATH"

# --- Benchmark Loop ---
for model_name in "${MODELS_TO_TEST_LIST[@]}"; do
    log_subhdr
    log_info "Benchmark: Processing '$model_name'"

    model_proxy=$(yq -e ".models[\"$model_name\"].proxy // \"\"" "$CONFIG_FULL_PATH")
    model_cmd=$(yq -e ".models[\"$model_name\"].cmd // \"\"" "$CONFIG_FULL_PATH")

    # Basic validation
    if [[ -z "$model_proxy" || -z "$model_cmd" ]]; then
        log_warn "Benchmark: Skipping '$model_name' - Invalid or incomplete config."
        error_msg="Invalid/incomplete config"
        echo "\"$model_name\",\"$(date '+%Y-%m-%d %H:%M:%S')\",\"Config Error\",,,,,\",\",\"\",\"\",\"\",\"$model_proxy\",\"$error_msg\"" >> "$OUTPUT_FULL_PATH"
        continue
    fi

    parsed_command_output=$(parse_llama_server_command "$model_cmd" "$model_name")
     if [[ -z "$parsed_command_output" ]]; then
         log_warn "Benchmark: Skipping '$model_name' - Cannot parse command."
         error_msg="Cannot parse cmd"
         echo "\"$model_name\",\"$(date '+%Y-%m-%d %H:%M:%S')\",\"Cmd Error\",,,,,\",\",\"\",\"\",\"\",\"$model_proxy\",\"$error_msg\"" >> "$OUTPUT_FULL_PATH"
         continue
     fi
    IFS='|' read -r server_executable server_arguments <<< "$parsed_command_output"

    # Check if executable exists and is executable
    if [[ ! -x "$server_executable" ]]; then
        log_warn "Benchmark: Skipping '$model_name' - Executable not found or not executable at '$server_executable'."
        error_msg="Executable not found or not executable"
        echo "\"$model_name\",\"$(date '+%Y-%m-%d %H:%M:%S')\",\"Exe Error\",,,,,\",\",\"\",\"\",\"\",\"$model_proxy\",\"$error_msg\"" >> "$OUTPUT_FULL_PATH"
        continue
    fi

    gpu_gb_from_scan=""
    cpu_gb_from_scan=""
    retrieved_status="Not Scanned" # REMOVED local - Default status if not found

    if [[ -v MEMORY_SCAN_RESULTS["$model_name"] ]]; then
        scan_data="${MEMORY_SCAN_RESULTS[$model_name]}"
        log_verbose "Retrieved scan data from MEMORY_SCAN_RESULTS for ${model_name}: [$scan_data]" # Debug log

        # Use temporary variables for reading to avoid conflicts
        # REMOVED local, initialize instead
        temp_gpu="" temp_cpu="" temp_status=""
        IFS='|' read -r temp_gpu temp_cpu temp_status <<< "$scan_data"
        retrieved_status="${temp_status:-"Read Error"}" # Use default if read resulted in empty status
        log_verbose "Parsed status after IFS read in Pass 2: [$retrieved_status]" # Debug log

        if [[ "$retrieved_status" == "Success" ]]; then
             gpu_gb_from_scan=$temp_gpu
             cpu_gb_from_scan=$temp_cpu
             log_verbose "Scan status Success. Using values: GPU=[$gpu_gb_from_scan] CPU=[$cpu_gb_from_scan]"
        else
            # Use the retrieved_status variable in the warning message
            log_warn "Benchmark: Memory scan data for '$model_name' indicates failure/skip (Status: $retrieved_status). Memory fields will be empty."
        fi
    else
         log_info "Benchmark: No pre-scanned memory data available for '$model_name' (expected if -m was used)."
         # retrieved_status remains "Not Scanned"
    fi
    # Use placeholder if null/empty (remains the same)
    [[ "$gpu_gb_from_scan" == "null" || -z "$gpu_gb_from_scan" ]] && gpu_gb_from_scan=""
    [[ "$cpu_gb_from_scan" == "null" || -z "$cpu_gb_from_scan" ]] && cpu_gb_from_scan=""


    # --- Initialize Result Variables ---
    current_status="Not Run"
    duration_sec=""
    prompt_tokens=""
    completion_tokens=""
    total_tokens=""
    tps=""
    error_msg=""
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # --- Start Server and Benchmark ---
    server_pid=""       # Reset variables
    stdout_log=""
    stderr_log=""

    # Create a temporary file path for the start result
    start_result_file="${TEMP_DIR}/${model_name}_start_result_p2.txt"

    # Call start_llama_server, passing the result file path. Check its exit code.
    if start_llama_server "$server_executable" "$server_arguments" "$model_name" "$start_result_file"; then
        # start_llama_server succeeded (returned 0), read the result file
        start_output=""
        if [[ -f "$start_result_file" ]]; then
            start_output=$(<"$start_result_file") # Read file content
        fi
        log_verbose "start_llama_server (Pass 2) output from file: [$start_output]" >&2

        if [[ -n "$start_output" ]]; then
            # Parse PID from the file content
            IFS='|' read -r server_pid stdout_log stderr_log <<< "$start_output"

            # *** PID VALIDATION (Crucial) ***
            if ! [[ "$server_pid" =~ ^[0-9]+$ ]]; then
                log_error "Pass 2: Invalid PID parsed from result file for '$model_name'. Content: [$start_output]" >&2
                current_status="Start Error (Bad PID)"
                error_msg="Failed to parse PID from server start output file: $start_output"
                server_pid="" # Ensure PID is invalid
                # Skip health check and API call - jump to cleanup/recording
            else
# === Inside Pass 2 Loop, after PID validation succeeds ===

                # --- PID is valid, proceed with health check ---
                log_verbose "Pass 2: Parsed Valid PID: $server_pid" >&2

                if [[ ! "$model_proxy" =~ ^https?:// ]]; then base_url="http://${model_proxy}"; else base_url="${model_proxy}"; fi
                health_url="${base_url}/health"
                api_endpoint="${base_url}/v1/chat/completions"

                if wait_llama_server_healthy "$server_pid" "$health_url" "$HEALTH_CHECK_TIMEOUT_SECONDS" "$HEALTH_CHECK_POLL_INTERVAL" "$model_name"; then
                    # --- Server Healthy: Perform Benchmark Request ---
                    log_info "  Server healthy. Sending benchmark request to $api_endpoint..." >&2

                    # Construct JSON payload using jq
                    json_payload=$(jq -n \
                        --arg model "$model_name" \
                        --arg question "$QUESTION" \
                        '{model: $model, messages: [{role: "user", content: $question}], temperature: 0.7, stream: false}')

                    response_file="${TEMP_DIR}/${model_name}_response.json"
                    error_file="${TEMP_DIR}/${model_name}_curl_error.txt"

                    # Time measurement vars
                    start_req_ns="" end_req_ns=""

                    if date +%s.%N >/dev/null 2>&1; then start_req_ns=$(date +%s.%N); else start_req_ns=$(date +%s); fi

                    curl_response_code=$(curl -sS --fail-with-body \
                         -X POST \
                         -H "Content-Type: application/json" \
                         -d "$json_payload" \
                         --max-time 600 \
                         -o "$response_file" \
                         -w "%{http_code}" \
                         "$api_endpoint" 2> "$error_file")
                    curl_rc=$?

                    if date +%s.%N >/dev/null 2>&1; then end_req_ns=$(date +%s.%N); else end_req_ns=$(date +%s); fi

                    duration_sec=$(echo "$end_req_ns - $start_req_ns" | bc -l)
                    duration_sec=$(printf "%.3f" "$duration_sec") # Format

                    # --- Process API Results ---
                    if [[ $curl_rc -eq 0 && "$curl_response_code" -eq 200 ]]; then
                         log_info "  Received response from $model_name." >&2

                         # Parse response using jq
                         prompt_tokens=$(jq -e '.usage.prompt_tokens // 0' "$response_file")
                         completion_tokens=$(jq -e '.usage.completion_tokens // 0' "$response_file")
                         total_tokens=$(jq -e '.usage.total_tokens // 0' "$response_file")
                         jq_rc=$?

                         if [[ $jq_rc -eq 0 ]]; then
                             current_status="Success"
                             if (( $(echo "$duration_sec > 0 && $completion_tokens > 0" | bc -l) )); then
                                 tps=$(echo "scale=2; $completion_tokens / $duration_sec" | bc -l)
                                 tps=$(printf "%.2f" "$tps") # Format
                                 log_success "  Result: ${duration_sec}s, ${completion_tokens} completion tokens, ${tps} TPS" # Use log_success
                             else
                                 log_warn "  Could not calculate TokensPerSecond (Duration or Completion Tokens were zero or invalid)." >&2
                                 tps="0.00"
                             fi
                         else # JQ parsing failed
                            current_status="API Response Error"
                            error_msg="Failed to parse JSON response from API. Check $response_file"
                            log_error "  $error_msg" >&2
                            prompt_tokens=""; completion_tokens=""; total_tokens=""; tps=""
                         fi # End JQ parsing if

                    else # Curl failed or non-200 response
                        current_status="API Request Failed"
                        if [[ -s "$error_file" ]]; then error_msg=$(head -n 1 "$error_file" | tr -d '\r\n' | cut -c 1-200);
                        elif [[ -s "$response_file" ]]; then error_msg=$(head -n 1 "$response_file" | tr -d '\r\n' | cut -c 1-200);
                        else error_msg="Curl failed (rc=$curl_rc, http=$curl_response_code). No error message captured."; fi
                        log_error "  API request failed for '$model_name': $error_msg" >&2
                        prompt_tokens=""; completion_tokens=""; total_tokens=""; tps=""
                    fi # End Curl success check if/else

                # *** THIS IS WHERE THE SYNTAX ERROR WAS LIKELY INTRODUCED ***
                # The 'else' corresponding to the 'if wait_llama_server_healthy...'
                else # Health check failed
                    current_status="Health Timeout"
                    error_msg="Server (PID: $server_pid) did not become healthy for benchmark."
                    log_warn "  Benchmark: Server '$model_name' (PID: $server_pid) failed health check." >&2
                    # Check if process exited during wait
                    if ! kill -0 "$server_pid" 2>/dev/null; then
                        error_msg="$error_msg | Server process $server_pid exited prematurely."
                    fi
                fi
            fi # End PID validation check
        else
            # start_llama_server succeeded but result file was empty?
            log_error "Pass 2: start_llama_server succeeded but result file was empty for '$model_name'." >&2
            current_status="Start Error (Empty Result)"
            error_msg="start_llama_server gave empty result file"
            server_pid="" # No valid PID
        fi # End checking start_output from file
    else
        # start_llama_server failed (returned non-zero exit code)
        current_status="Start Error"
        error_msg="Failed to start server process for benchmark. See logs."
        # Error message printed by function to stderr
        log_error "  Benchmark: Failed to start server for '$model_name'." >&2
        server_pid="" # No PID to stop
    fi # End checking start_llama_server exit code

    # --- Stop Server (if PID is valid) and Record Result ---
    if [[ -n "$server_pid" && "$server_pid" =~ ^[0-9]+$ ]]; then # Check validity again
        stop_llama_server "$server_pid" "$model_name"
    elif [[ -n "$server_pid" ]]; then # Log if PID was set but invalid
        log_warn "Pass 2: Attempted to stop server for $model_name, but PID was invalid: [$server_pid]" >&2
    fi

    # Append result to CSV
    echo "\"$model_name\",\"$timestamp\",\"$current_status\",\"$gpu_gb_from_scan\",\"$cpu_gb_from_scan\",\"$duration_sec\",\"$tps\",\"$prompt_tokens\",\"$completion_tokens\",\"$total_tokens\",\"$model_proxy\",\"${error_msg//\"/\\\"}\"" >> "$OUTPUT_FULL_PATH"

    # Clean up the start result file
    rm -f "$start_result_file"

done # End foreach model for Benchmark

log_hdr
log_info "PASS 2: Benchmark Complete."
log_hdr

#==============================================================================
# Final Summary
#==============================================================================
log_subhdr
if [[ -f "$OUTPUT_FULL_PATH" ]]; then
    log_success "Benchmark results saved to: $OUTPUT_FULL_PATH"
    # Optionally show results summary? 'column -t -s,' might work if installed.
    # column -t -s, "$OUTPUT_FULL_PATH" | head -n 10 # Example
else
    log_warn "No output file was generated."
fi

SCRIPT_END_TIME_S=$(date +%s)
SCRIPT_DURATION=$((SCRIPT_END_TIME_S - SCRIPT_START_TIME_S))
log_info "Benchmark script finished."
log_info "Total duration: $(date -u -r "${SCRIPT_DURATION}" +'%H:%M:%S')"

# Cleanup is handled by the trap

exit 0