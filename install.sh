#!/bin/bash
set -euo pipefail # Exit on error, exit on unset variable, check pipeline errors

# --- Configuration ---
# Adjust INSTALL_DIR if you want to install somewhere else
INSTALL_DIR="${HOME}/llama_suite" # Default: /Users/your_username/llama_suite
LLAMA_CPP_DIR="${INSTALL_DIR}/llama.cpp"
LLAMA_SWAP_SOURCE_DIR="${INSTALL_DIR}/llama-swap-source"
LLAMA_SWAP_EXE_DIR="${INSTALL_DIR}/llama-swap" # Directory where the compiled binary will be placed
LLAMA_SWAP_EXE="${LLAMA_SWAP_EXE_DIR}/llama-swap" # Executable name without .exe
MODEL_DIR="${INSTALL_DIR}/models"
WEBUI_DIR="${INSTALL_DIR}/open-webui" # Data directory for Open WebUI persistent data
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_SWAP_REPO="https://github.com/mostlygeek/llama-swap.git"
OPEN_WEBUI_IMAGE="ghcr.io/open-webui/open-webui:main"

# --- Prerequisite Check ---
echo "Checking prerequisites..."
command -v git >/dev/null 2>&1 || { echo >&2 "ERROR: Git not found. Install Git (e.g., brew install git or via Xcode Command Line Tools)."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo >&2 "ERROR: CMake not found. Install CMake (e.g., brew install cmake)."; exit 1; }
command -v make >/dev/null 2>&1 || { echo >&2 "ERROR: Make not found. Install Make (usually included with Xcode Command Line Tools)."; exit 1; }
command -v c++ >/dev/null 2>&1 || { echo >&2 "ERROR: C++ Compiler not found. Install Xcode Command Line Tools (run 'xcode-select --install')."; exit 1; }
command -v go >/dev/null 2>&1 || { echo >&2 "ERROR: Go compiler not found. Install Go (e.g., brew install go)."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo >&2 "ERROR: Docker not found. Install Docker Desktop for Mac and ensure it's running."; exit 1; }

# Check for CUDA (less common on Mac, but check anyway)
if ! command -v nvcc >/dev/null 2>&1; then
    echo "WARNING: NVCC (CUDA Compiler) not found. GPU acceleration will use Metal (recommended on macOS)."
fi
# Check if Xcode Command Line Tools are installed (needed for Metal)
if ! xcode-select -p > /dev/null 2>&1; then
     echo >&2 "WARNING: Xcode Command Line Tools not found or configured. Metal build might fail. Run 'xcode-select --install'."
fi

echo "Prerequisites check passed."

# --- Create Directories ---
echo "Creating directories in ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
mkdir -p "${MODEL_DIR}"
mkdir -p "${WEBUI_DIR}"
mkdir -p "${LLAMA_SWAP_EXE_DIR}" # Create the target directory for the llama-swap binary

# --- Define llama.cpp server executable path ---
# Adjust if the build process outputs the server binary elsewhere
LLAMA_SERVER_EXE="server" # Common name for the server binary in llama.cpp builds
LLAMA_SERVER_FULL_PATH="${LLAMA_CPP_DIR}/build/bin/${LLAMA_SERVER_EXE}"

# Function for Yes/No prompts
ask_yes_no() {
    while true; do
        read -p "$1 [y/N]: " yn
        case $yn in
            [Yy]* ) return 0;; # Yes returns 0 (true in shell)
            [Nn]* ) return 1;; # No returns 1 (false in shell)
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# --- Handle llama.cpp Installation/Compilation ---
COMPILE_LLAMA_CPP="N"
cd "${INSTALL_DIR}" # Ensure we are in the main installation directory

if [ -d "${LLAMA_CPP_DIR}" ]; then
    echo "llama.cpp source found at ${LLAMA_CPP_DIR}."
    if ask_yes_no "Recompile llama.cpp? (Choosing N assumes existing binary ${LLAMA_SERVER_EXE} is usable)"; then
        COMPILE_LLAMA_CPP="Y"
        echo "Updating and recompiling llama.cpp..."
        cd "${LLAMA_CPP_DIR}"
        git pull
    else
        echo "Skipping llama.cpp recompilation."
    fi
else
    echo "llama.cpp source NOT found. Cloning and compiling..."
    git clone "${LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
    COMPILE_LLAMA_CPP="Y"
fi

# --- Build llama.cpp block (only if COMPILE_LLAMA_CPP is Y) ---
if [ "${COMPILE_LLAMA_CPP}" = "Y" ]; then
    echo "Configuring and building llama.cpp..."
    cd "${LLAMA_CPP_DIR}"
    # Clean previous build artifacts (optional but recommended)
    rm -rf build
    mkdir build
    cd build
    # Configure using CMake. Enable Metal for GPU acceleration on macOS.
    # Disable CURL if not needed (as per original script), enable if you need URL model loading.
    echo "Running CMake with Metal support..."
    cmake .. -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    if [ $? -ne 0 ]; then
        echo "ERROR: llama.cpp CMake configuration failed." >&2
        exit 1
    fi
    echo "Building llama.cpp project (using 'make' or Ninja)..."
    # Use cmake --build which abstracts the underlying build system (make, ninja)
    # The number of jobs (-j) speeds up compilation, adjust based on your CPU cores
    cmake --build . --config Release -j $(sysctl -n hw.logicalcpu)
    if [ $? -ne 0 ]; then
        echo "ERROR: llama.cpp Build failed." >&2
        exit 1
    fi
    echo "llama.cpp build complete."
fi
cd "${INSTALL_DIR}" # Go back to base install dir

# --- Check llama.cpp server executable ---
echo "Checking for llama.cpp server binary..."
if [ ! -f "${LLAMA_SERVER_FULL_PATH}" ]; then
    echo "ERROR: ${LLAMA_SERVER_EXE} not found at ${LLAMA_SERVER_FULL_PATH} after installation steps." >&2
    echo "Possible issues: Build failed, or the binary is in a different location." >&2
     # Let's try a common alternative location
    ALT_LLAMA_SERVER_FULL_PATH="${LLAMA_CPP_DIR}/server"
    if [ -f "${ALT_LLAMA_SERVER_FULL_PATH}" ]; then
         echo "Found server binary at alternative location: ${ALT_LLAMA_SERVER_FULL_PATH}"
         LLAMA_SERVER_FULL_PATH="${ALT_LLAMA_SERVER_FULL_PATH}" # Update the path
    else
        exit 1
    fi
else
    echo "${LLAMA_SERVER_EXE} check PASSED: Found at ${LLAMA_SERVER_FULL_PATH}"
fi
# Ensure the server is executable
chmod +x "${LLAMA_SERVER_FULL_PATH}"

# --- Handle llama-swap Installation/Compilation ---
COMPILE_LLAMA_SWAP="N"
cd "${INSTALL_DIR}" # Ensure we are in the main installation directory

if [ -d "${LLAMA_SWAP_SOURCE_DIR}" ]; then
     echo "llama-swap source found at ${LLAMA_SWAP_SOURCE_DIR}."
     if ask_yes_no "Recompile llama-swap? (Choosing N assumes existing binary ${LLAMA_SWAP_EXE} is usable)"; then
        COMPILE_LLAMA_SWAP="Y"
        echo "Updating and recompiling llama-swap..."
        cd "${LLAMA_SWAP_SOURCE_DIR}"
        git pull
     else
        echo "Skipping llama-swap recompilation."
     fi
else
    echo "llama-swap source NOT found. Cloning and compiling..."
    git clone "${LLAMA_SWAP_REPO}" "${LLAMA_SWAP_SOURCE_DIR}"
    COMPILE_LLAMA_SWAP="Y"
fi

# --- Build llama-swap block (only if COMPILE_LLAMA_SWAP is Y) ---
if [ "${COMPILE_LLAMA_SWAP}" = "Y" ]; then
    echo "Building llama-swap using 'go build'..."
    cd "${LLAMA_SWAP_SOURCE_DIR}"
    # Build the binary and place it in the designated EXE directory
    go build -o "${LLAMA_SWAP_EXE}" .
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build llama-swap." >&2
        exit 1
    fi
    echo "llama-swap build complete."
fi
cd "${INSTALL_DIR}" # Go back to base install dir

# --- Check llama-swap executable ---
echo "Checking for llama-swap binary..."
if [ ! -f "${LLAMA_SWAP_EXE}" ]; then
     echo "ERROR: llama-swap not found at ${LLAMA_SWAP_EXE} after installation steps." >&2
     exit 1
else
     echo "llama-swap check PASSED: Found at ${LLAMA_SWAP_EXE}"
fi
# Ensure the swap binary is executable
chmod +x "${LLAMA_SWAP_EXE}"

# --- Pull Open WebUI Docker Image ---
echo "Pulling Open WebUI Docker image (${OPEN_WEBUI_IMAGE})..."
if ! docker pull "${OPEN_WEBUI_IMAGE}"; then
    echo "WARNING: Failed to pull Open WebUI image. Check Docker Hub/GHCR status and your internet connection."
fi

echo
echo "--- Installation Summary ---"
echo "Installation Directory: ${INSTALL_DIR}"
echo "llama.cpp source:     ${LLAMA_CPP_DIR}"
echo "llama.cpp server:     ${LLAMA_SERVER_FULL_PATH} (Using Metal for GPU: ${COMPILE_LLAMA_CPP})"
echo "llama-swap source:    ${LLAMA_SWAP_SOURCE_DIR}"
echo "llama-swap binary:    ${LLAMA_SWAP_EXE}"
echo "Model directory:      ${MODEL_DIR} (Download models here)"
echo "Open WebUI data dir:  ${WEBUI_DIR}"
echo "Open WebUI Docker img: ${OPEN_WEBUI_IMAGE} (pulled)"
echo

echo "--- Next Steps ---"
echo "1. Download desired GGUF models into your model directory: ${MODEL_DIR}"
echo "   (e.g., from Hugging Face: https://huggingface.co/models?search=gguf)"
echo "2. Create/Edit 'config.yaml' in the script directory (${INSTALL_DIR}) to define models for llama-swap."
echo "   - Example config: https://github.com/mostlygeek/llama-swap/blob/main/config.yaml.example"
echo "   - Ensure paths to '${LLAMA_SERVER_FULL_PATH}' and models (in '${MODEL_DIR}') within the 'cmd:' section are correct."
echo "   - Make sure the '--model' path in 'cmd:' points to the actual model file within ${MODEL_DIR}."
echo "   - Use --gpu-layers (-ngl) > 0 in the 'cmd:' for llama.cpp server to enable Metal acceleration (e.g., -ngl 35)."
echo "   - Each model needs a UNIQUE proxy port (e.g., 9901, 9902...)."
echo "3. Ensure Docker Desktop is running."
echo "4. Create a 'run_server.sh' script (or similar) in ${INSTALL_DIR} to start llama-swap:"
echo "   #!/bin/bash"
echo "   cd \"${INSTALL_DIR}\" # Or wherever config.yaml is"
echo "   \"${LLAMA_SWAP_EXE}\" -config config.yaml"
echo "   chmod +x run_server.sh"
echo "5. Run './run_server.sh' - This will start llama-swap (which then starts llama.cpp instances)."
echo "6. Start Open WebUI Docker container. Example command:"
echo "   docker run -d -p 3000:8080 \\"
echo "     -v \"${WEBUI_DIR}:/app/backend/data\" \\"
echo "     --add-host=host.docker.internal:host-gateway \\" # Important for connecting to server on host
echo "     --name open-webui --restart always \\"
echo "     ${OPEN_WEBUI_IMAGE}"
echo "   (Access WebUI at http://localhost:3000)"
echo "7. In Open WebUI (Settings -> Connections), add your models by connecting to llama-swap's proxy ports"
echo "   (e.g., http://host.docker.internal:9901, http://host.docker.internal:9902 etc., using the ports defined in config.yaml)."
echo "   Use 'host.docker.internal' instead of 'localhost' inside the Docker container."

echo
echo "--- Installation Complete ---"

exit 0