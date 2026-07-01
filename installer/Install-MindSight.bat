@echo off
REM ============================================================================
REM  MindSight -- Windows installer (SP4.0 interim)
REM
REM  Double-click this file to install MindSight into your user profile.
REM  It installs the uv package manager, a managed Python 3.12, the MindSight
REM  source tree, all locked dependencies, downloads the required model
REM  weights, and creates Desktop / Start Menu shortcuts.
REM
REM  Re-running this file updates an existing install (it is safe to run again).
REM  ASCII output only. No PowerShell-only control flow in this script.
REM ============================================================================

setlocal EnableExtensions

REM ---- Locations ------------------------------------------------------------
set "SRC_DIR=%~dp0"
if "%SRC_DIR:~-1%"=="\" set "SRC_DIR=%SRC_DIR:~0,-1%"
set "INSTALL_DIR=%LOCALAPPDATA%\MindSight"
set "APP_DIR=%INSTALL_DIR%\app"
set "VENV_DIR=%INSTALL_DIR%\venv"
set "UV_BIN=%USERPROFILE%\.local\bin"
set "FAILSTEP="

echo(
echo ============================================================
echo   MindSight installer
echo   Target folder: "%INSTALL_DIR%"
echo ============================================================
echo(

REM ==========================================================================
REM  [1/7] Locate or install uv
REM ==========================================================================
set "FAILSTEP=1 (install uv)"
where uv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [1/7] uv already installed ... OK
) else (
    if exist "%UV_BIN%\uv.exe" (
        echo [1/7] uv found in "%UV_BIN%" ... OK
    ) else (
        echo [1/7] Installing uv package manager ...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        if not %ERRORLEVEL% EQU 0 (
            echo [1/7] Installing uv ... FAILED
            goto fail
        )
        echo [1/7] Installing uv ... OK
    )
)
REM Make sure uv is on PATH for the rest of this session.
set "PATH=%UV_BIN%;%PATH%"
where uv >nul 2>&1
if not %ERRORLEVEL% EQU 0 (
    echo [1/7] uv is not on PATH after install ... FAILED
    goto fail
)

REM ==========================================================================
REM  [2/7] Install a managed Python 3.12
REM ==========================================================================
set "FAILSTEP=2 (install Python 3.12)"
echo [2/7] Ensuring managed Python 3.12 ...
uv python install 3.12
if not %ERRORLEVEL% EQU 0 (
    echo [2/7] Installing Python 3.12 ... FAILED
    goto fail
)
echo [2/7] Python 3.12 ready ... OK

REM ==========================================================================
REM  [3/7] Deploy the MindSight source tree
REM ==========================================================================
set "FAILSTEP=3 (copy application files)"
echo [3/7] Copying application files to "%APP_DIR%" ...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
REM robocopy exit codes 0-7 are success; 8 and above are real errors.
robocopy "%SRC_DIR%\app" "%APP_DIR%" /E /NFL /NDL /NJH /NJS /NP >nul
if %ERRORLEVEL% GEQ 8 (
    echo [3/7] Copying application files ... FAILED
    goto fail
)
echo [3/7] Application files in place ... OK

REM ==========================================================================
REM  [4/7] Create the virtual environment and install locked dependencies
REM ==========================================================================
set "FAILSTEP=4 (install dependencies)"
echo [4/7] Installing dependencies from the locked manifest (this can take a while) ...
set "UV_PROJECT_ENVIRONMENT=%VENV_DIR%"
uv sync --frozen --project "%APP_DIR%"
if not %ERRORLEVEL% EQU 0 (
    echo [4/7] Installing dependencies ... FAILED
    goto fail
)
echo [4/7] Dependencies installed (editable) ... OK

REM ==========================================================================
REM  [5/7] GPU: install the CUDA build of PyTorch when an NVIDIA GPU is present
REM ==========================================================================
REM  (Kept flat with labels: %ERRORLEVEL% is stale inside nested () blocks.)
set "FAILSTEP=5 (GPU torch selection)"
where nvidia-smi >nul 2>&1
if not %ERRORLEVEL% EQU 0 goto cpu_torch
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 goto cuda_torch
echo [5/7] No usable NVIDIA GPU -- keeping CPU PyTorch ... OK
goto after_torch

:cuda_torch
echo [5/7] NVIDIA GPU detected -- installing CUDA build of PyTorch ...
uv pip install --python "%VENV_DIR%\Scripts\python.exe" torch torchvision --index-url https://download.pytorch.org/whl/cu124
if not %ERRORLEVEL% EQU 0 (
    echo [5/7] Installing CUDA PyTorch ... FAILED
    echo       You can still run MindSight on CPU. Re-run to retry the GPU step.
    goto fail
)
echo [5/7] CUDA PyTorch installed ... OK
goto after_torch

:cpu_torch
echo [5/7] No NVIDIA GPU detected -- keeping CPU PyTorch ... OK

:after_torch

REM ==========================================================================
REM  [6/7] Download the required model weights (headless)
REM ==========================================================================
set "FAILSTEP=6 (download weights)"
echo [6/7] Downloading required model weights (Gaze-LLE, MobileGaze, YOLO) ...
"%VENV_DIR%\Scripts\mindsight-weights.exe" --required
if not %ERRORLEVEL% EQU 0 (
    echo [6/7] Downloading weights ... FAILED
    echo       Check your internet connection, then re-run this installer.
    goto fail
)
echo [6/7] Required weights present and verified ... OK

REM ==========================================================================
REM  [7/7] Create Desktop and Start Menu shortcuts
REM ==========================================================================
set "FAILSTEP=7 (create shortcuts)"
echo [7/7] Creating Desktop and Start Menu shortcuts ...
set "GUI_EXE=%VENV_DIR%\Scripts\mindsight-gui.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = New-Object -ComObject WScript.Shell; $desktop = $s.SpecialFolders('Desktop'); $programs = $s.SpecialFolders('Programs'); foreach ($dir in @($desktop, $programs)) { $lnk = $s.CreateShortcut((Join-Path $dir 'MindSight.lnk')); $lnk.TargetPath = '%GUI_EXE%'; $lnk.WorkingDirectory = '%APP_DIR%'; $lnk.Description = 'MindSight eye-tracking analysis'; $lnk.Save() }"
if not %ERRORLEVEL% EQU 0 (
    echo [7/7] Creating shortcuts ... FAILED
    echo       MindSight is installed; you can launch it with:
    echo         "%GUI_EXE%"
    goto fail
)
echo [7/7] Shortcuts created ... OK

echo(
echo ============================================================
echo   MindSight install: PASS
echo(
echo   Launch it from the "MindSight" Desktop shortcut or the
echo   Start Menu, or run directly:
echo     "%GUI_EXE%"
echo(
echo   Installed in: "%INSTALL_DIR%"
echo   Your projects, weights and outputs live under "%APP_DIR%".
echo   To uninstall: delete that folder and the two MindSight shortcuts.
echo ============================================================
echo(
pause
endlocal
exit /b 0

:fail
echo(
echo ============================================================
echo   MindSight install: FAILED at step %FAILSTEP%
echo(
echo   Nothing was pushed to your system beyond the partial folder
echo   at "%INSTALL_DIR%". You can delete that folder and run this
echo   installer again. See INSTALL-WINDOWS.md for troubleshooting.
echo ============================================================
echo(
pause
endlocal
exit /b 1
