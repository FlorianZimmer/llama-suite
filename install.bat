@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set INSTALL_DIR=C:\Users\Florian\OneDrive\Dokumente\Privat\Programmieren\llama_suite
set LLAMA_CPP_DIR=%INSTALL_DIR%\llama.cpp
set LLAMA_SWAP_SOURCE_DIR=%INSTALL_DIR%\llama-swap-source
set LLAMA_SWAP_EXE_DIR=%INSTALL_DIR%\llama-swap
set LLAMA_SWAP_EXE=%LLAMA_SWAP_EXE_DIR%\llama-swap.exe
set MODEL_DIR=%INSTALL_DIR%\models
set WEBUI_DIR=%INSTALL_DIR%\open-webui
set LLAMA_CPP_REPO=https://github.com/ggerganov/llama.cpp.git
set LLAMA_SWAP_REPO=https://github.com/mostlygeek/llama-swap.git

REM --- Prerequisite Check (Basic) ---
echo "Checking prerequisites (basic)..."
where git >nul 2>nul || (echo "ERROR: Git not found in PATH." & exit /b 1)
where cmake >nul 2>nul || (echo "ERROR: CMake not found in PATH." & exit /b 1)
where cl >nul 2>nul || (echo "ERROR: C++ Compiler (cl.exe) not found." & exit /b 1)
where docker >nul 2>nul || (echo "ERROR: Docker not found in PATH." & exit /b 1)
where go >nul 2>nul || (echo "ERROR: Go compiler (go.exe) not found in PATH." & exit /b 1)
where nvcc >nul 2>nul || (echo "WARNING: NVCC (CUDA Compiler) not found in PATH.")

echo "Prerequisites check passed (basic)."

REM --- Create Directories ---
echo "Creating directories..."
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%WEBUI_DIR%" mkdir "%WEBUI_DIR%"
if not exist "%LLAMA_SWAP_EXE_DIR%" mkdir "%LLAMA_SWAP_EXE_DIR%"
cd /d "%INSTALL_DIR%"

REM --- Set llama.cpp server executable name ---
set LLAMA_SERVER_EXE=llama-server.exe
REM set LLAMA_SERVER_EXE=llama-server.exe # <-- Adjust if needed
set LLAMA_SERVER_FULL_PATH=%LLAMA_CPP_DIR%\build\bin\Release\%LLAMA_SERVER_EXE%

REM --- Handle llama.cpp Installation/Compilation ---
set COMPILE_LLAMA_CPP=N
if exist "%LLAMA_CPP_DIR%" (
    echo "llama.cpp source found."
    choice /c YN /m "Recompile llama.cpp? (Choosing N assumes existing binary %LLAMA_SERVER_EXE% is usable)"
    if errorlevel 1 if not errorlevel 2 (
        set COMPILE_LLAMA_CPP=Y
        echo "Updating and recompiling llama.cpp..."
        cd /d "%LLAMA_CPP_DIR%"
        git pull
    ) else (
        echo "Skipping llama.cpp recompilation."
    )
) else (
    echo "llama.cpp source NOT found. Cloning and compiling..."
    git clone %LLAMA_CPP_REPO% "%LLAMA_CPP_DIR%"
    if errorlevel 1 (echo "ERROR: Failed to clone llama.cpp." & exit /b 1)
    set COMPILE_LLAMA_CPP=Y
    cd /d "%LLAMA_CPP_DIR%"
)

REM --- Build llama.cpp block (only if COMPILE_LLAMA_CPP is Y) ---
if "!COMPILE_LLAMA_CPP!"=="Y" (
    if exist build rmdir /s /q build
    mkdir build
    cd build
    echo "Configuring CMake for llama.cpp..."
    cmake .. -DGGML_CUDA=ON -DLLAMA_CURL=OFF
    if errorlevel 1 (echo "ERROR: llama.cpp CMake configuration failed." & exit /b 1)
    echo "Building llama.cpp project..."
    cmake --build . --config Release
    if errorlevel 1 (echo "ERROR: llama.cpp Build failed." & exit /b 1)
)
cd /d "%INSTALL_DIR%" REM Go back to base install dir

REM --- Check llama.cpp server executable ---
if not exist "%LLAMA_SERVER_FULL_PATH%" (
    echo "ERROR: %LLAMA_SERVER_EXE% not found at %LLAMA_SERVER_FULL_PATH% after installation steps."
    exit /b 1
) else (
    echo "%LLAMA_SERVER_EXE% check PASSED."
)

REM --- Handle llama-swap Installation/Compilation ---
set COMPILE_LLAMA_SWAP=N
if exist "%LLAMA_SWAP_SOURCE_DIR%" (
     echo "llama-swap source found."
     choice /c YN /m "Recompile llama-swap? (Choosing N assumes existing binary %LLAMA_SWAP_EXE% is usable)"
     if errorlevel 1 if not errorlevel 2 (
        set COMPILE_LLAMA_SWAP=Y
        echo "Updating and recompiling llama-swap..."
        cd /d "%LLAMA_SWAP_SOURCE_DIR%"
        git pull
     ) else (
        echo "Skipping llama-swap recompilation."
     )
) else (
    echo "llama-swap source NOT found. Cloning and compiling..."
    git clone %LLAMA_SWAP_REPO% "%LLAMA_SWAP_SOURCE_DIR%"
    if errorlevel 1 (echo "ERROR: Failed to clone llama-swap." & exit /b 1)
    set COMPILE_LLAMA_SWAP=Y
    cd /d "%LLAMA_SWAP_SOURCE_DIR%"
)

REM --- Build llama-swap block (only if COMPILE_LLAMA_SWAP is Y) ---
if "!COMPILE_LLAMA_SWAP!"=="Y" (
    echo "Building llama-swap using 'go build'..."
    go build -o "%LLAMA_SWAP_EXE%" .
    if errorlevel 1 (
        echo "ERROR: Failed to build llama-swap."
        exit /b 1
    )
)
cd /d "%INSTALL_DIR%" REM Go back to base install dir

REM --- Check llama-swap executable ---
if not exist "%LLAMA_SWAP_EXE%" (
     echo "ERROR: llama-swap.exe not found at %LLAMA_SWAP_EXE% after installation steps."
     exit /b 1
) else (
     echo "llama-swap.exe check PASSED."
)

REM --- Pull Open WebUI Docker Image ---
echo "Pulling Open WebUI Docker image..."
docker pull ghcr.io/open-webui/open-webui:main
if errorlevel 1 (echo "WARNING: Failed to pull Open WebUI image.")

echo "--- Installation Summary ---"
echo "llama.cpp source: %LLAMA_CPP_DIR%"
echo "llama.cpp server: %LLAMA_SERVER_FULL_PATH%"
echo "llama-swap source: %LLAMA_SWAP_SOURCE_DIR%"
echo "llama-swap binary: %LLAMA_SWAP_EXE%"
echo "Model directory: %MODEL_DIR% (Download models here, e.g., D:\LLMs\GGUF)"
echo "Open WebUI data directory: %WEBUI_DIR%"
echo "Open WebUI Docker image pulled (hopefully)."

echo "--- Next Steps ---"
echo "1. Download desired GGUF models into your model directory (e.g., D:\LLMs\GGUF)."
echo "2. Create/Edit 'config.yaml' in the script directory (e.g., %INSTALL_DIR%) to define models for llama-swap."
echo "   - Ensure paths to %LLAMA_SERVER_EXE% and models in 'cmd:' section are correct."
echo "   - Each model needs a UNIQUE proxy port (e.g., 9901, 9902...)."
echo "3. Run Docker Desktop."
echo "4. Run 'run_server.ps1' - This will start llama-swap."
echo "5. Start Open WebUI Docker container."
echo "6. Configure Open WebUI to connect to llama-swap's frontend port (default 8080)."

echo "--- Installation Complete ---"

endlocal
pause
exit /b 0