@echo off
setlocal

REM --- Configuration ---
set INSTALL_DIR=C:\Users\Florian\OneDrive\Dokumente\Privat\Programmieren\llama_suite
set LLAMA_SWAP_SOURCE_DIR=%INSTALL_DIR%\llama-swap-source
set CONFIG_YAML=%~dp0config.yaml
set WEBUI_CONTAINER_NAME=open-webui
set WEBUI_IMAGE_NAME=ghcr.io/open-webui/open-webui:main

echo "WARNING: This script will permanently delete the following:"
echo "- %INSTALL_DIR% (including llama.cpp source, builds, ALL downloaded models, llama-swap binary/source)"
echo "- %WEBUI_DIR% (Open WebUI persistent data)"
echo "- %CONFIG_YAML% (llama-swap config file, if it's next to this script)"
echo "It will also attempt to stop and remove the Open WebUI Docker container."

choice /c YN /m "Are you sure you want to continue?"
if errorlevel 2 (
    echo "Uninstallation cancelled."
    exit /b 1
)
if errorlevel 1 (
    echo "Proceeding with uninstallation..."
)

REM --- Stop and Remove Open WebUI Docker Container ---
echo "Stopping Open WebUI container (%WEBUI_CONTAINER_NAME%)..."
docker stop %WEBUI_CONTAINER_NAME% >nul 2>&1

echo "Removing Open WebUI container (%WEBUI_CONTAINER_NAME%)..."
docker rm %WEBUI_CONTAINER_NAME% >nul 2>&1

REM --- Optional: Remove Open WebUI Docker Image ---
choice /c YN /m "Do you also want to remove the Open WebUI Docker image (%WEBUI_IMAGE_NAME%)?"
if errorlevel 1 (
    if not errorlevel 2 (
        echo "Removing Open WebUI Docker image..."
        docker rmi %WEBUI_IMAGE_NAME% >nul 2>&1
    )
)

REM --- Delete Installation Directory & Files ---
echo "Deleting installation directory and related files..."

REM Delete config file (if next to script)
if exist "%CONFIG_YAML%" (
    echo "Removing %CONFIG_YAML%..."
    del "%CONFIG_YAML%"
)

REM Delete main install directory (contains llama.cpp, models, built llama-swap, webui data, source)
if exist "%INSTALL_DIR%" (
    echo "Removing %INSTALL_DIR%..."
    rmdir /s /q "%INSTALL_DIR%"
    if errorlevel 1 (
        echo "ERROR: Failed to delete %INSTALL_DIR%. It might be in use. Please close related programs/terminals."
        exit /b 1
    ) else (
        echo "Successfully removed %INSTALL_DIR%."
    )
) else (
    echo "Directory %INSTALL_DIR% not found. Skipping deletion."
)


echo "--- Uninstallation Complete ---"
echo "NOTE: Prerequisites (Git, CMake, Visual Studio, CUDA Toolkit, Docker Desktop, Go) are NOT uninstalled by this script."

pause
exit /b 0