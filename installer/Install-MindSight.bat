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

REM ---- Install mode ---------------------------------------------------------
REM  Local-zip mode (default): the zip bundles an app\ source tree next to this
REM  script; MindSight installs editable and PROJECT_ROOT resolves to that tree
REM  (MINDSIGHT_HOME stays unset). Release mode: no app\ tree is bundled, so
REM  MindSight installs non-editable from a wheel asset on the GitHub Release
REM  and MINDSIGHT_HOME points every data path at "%APP_DIR%". The wheel URL is
REM  supplied via MINDSIGHT_RELEASE_WHEEL_URL. Local-zip mode stays the default
REM  until v1.0 assets ship.
REM  Release-mode assets (wheel, weights manifest, pipeline preset) live on the
REM  GitHub Release; RELEASE_BASE_URL is that tag's download base. All three URLs
REM  are overridable via env so a future release can point at a new tag.
if not defined MINDSIGHT_RELEASE_BASE_URL set "MINDSIGHT_RELEASE_BASE_URL=https://github.com/kylen-d/mindsight/releases/download/v1.0.0-indev"
set "RELEASE_BASE_URL=%MINDSIGHT_RELEASE_BASE_URL%"
if not defined MINDSIGHT_RELEASE_WHEEL_URL set "MINDSIGHT_RELEASE_WHEEL_URL=%RELEASE_BASE_URL%/mindsight-1.0.0.dev3-py3-none-any.whl"
set "RELEASE_WHEEL_URL=%MINDSIGHT_RELEASE_WHEEL_URL%"
if not defined MINDSIGHT_RELEASE_MANIFEST_URL set "MINDSIGHT_RELEASE_MANIFEST_URL=%RELEASE_BASE_URL%/weights_manifest.json"
set "RELEASE_MANIFEST_URL=%MINDSIGHT_RELEASE_MANIFEST_URL%"
if not defined MINDSIGHT_RELEASE_PRESET_URL set "MINDSIGHT_RELEASE_PRESET_URL=%RELEASE_BASE_URL%/pipeline_known_good.yaml"
if not defined MINDSIGHT_RELEASE_LOWPOWER_URL set "MINDSIGHT_RELEASE_LOWPOWER_URL=%RELEASE_BASE_URL%/pipeline_low_power.yaml"
set "RELEASE_LOWPOWER_URL=%MINDSIGHT_RELEASE_LOWPOWER_URL%"
set "RELEASE_PRESET_URL=%MINDSIGHT_RELEASE_PRESET_URL%"
if exist "%SRC_DIR%\app" (
    set "INSTALL_MODE=local"
) else (
    set "INSTALL_MODE=release"
)

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
        REM  Dynamic 'errorlevel' form: %ERRORLEVEL% would expand stale here
        REM  (parse-time) because this check sits inside a parenthesized block.
        if errorlevel 1 (
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
echo [3/7] Preparing "%APP_DIR%" (mode: %INSTALL_MODE%) ...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if "%INSTALL_MODE%"=="release" goto deploy_release
REM robocopy exit codes 0-7 are success; 8 and above are real errors.
robocopy "%SRC_DIR%\app" "%APP_DIR%" /E /NFL /NDL /NJH /NJS /NP >nul
if %ERRORLEVEL% GEQ 8 (
    echo [3/7] Copying application files ... FAILED
    goto fail
)
echo [3/7] Application files in place ... OK
goto after_deploy

:deploy_release
REM  Release mode: "%APP_DIR%" is the data home (Weights/Outputs/Projects),
REM  not a source tree; MINDSIGHT_HOME points here for the rest of the run. The
REM  wheel ships only the Python packages, so the weights manifest and the
REM  pipeline_known_good.yaml preset are fetched from the release into "%APP_DIR%".
if not exist "%APP_DIR%" mkdir "%APP_DIR%"
set "MINDSIGHT_HOME=%APP_DIR%"
curl -LsSf "%RELEASE_MANIFEST_URL%" -o "%APP_DIR%\weights_manifest.json"
if not %ERRORLEVEL% EQU 0 (
    echo [3/7] Downloading weights manifest ... FAILED
    goto fail
)
if not exist "%APP_DIR%\configs" mkdir "%APP_DIR%\configs"
REM  Low-power variant is best-effort: absence must not fail the install.
curl -LsSf "%RELEASE_LOWPOWER_URL%" -o "%APP_DIR%\configs\pipeline_low_power.yaml"
curl -LsSf "%RELEASE_PRESET_URL%" -o "%APP_DIR%\configs\pipeline_known_good.yaml"
if not %ERRORLEVEL% EQU 0 (
    echo [3/7] Downloading pipeline preset ... FAILED
    goto fail
)
echo [3/7] Data home ready at "%APP_DIR%" ... OK

:after_deploy

REM ==========================================================================
REM  [4/7] Create the virtual environment and install locked dependencies
REM ==========================================================================
set "FAILSTEP=4 (install dependencies)"
echo [4/7] Installing dependencies from the locked manifest (this can take a while) ...
set "UV_PROJECT_ENVIRONMENT=%VENV_DIR%"
if "%INSTALL_MODE%"=="release" goto deps_release
uv sync --frozen --python 3.12 --project "%APP_DIR%"
if not %ERRORLEVEL% EQU 0 (
    echo [4/7] Installing dependencies ... FAILED
    goto fail
)
echo [4/7] Dependencies installed (editable) ... OK
goto after_deps

:deps_release
if not defined RELEASE_WHEEL_URL goto deps_release_nourl
if "%RELEASE_WHEEL_URL%"=="" goto deps_release_nourl
uv venv --python 3.12 "%VENV_DIR%"
if not %ERRORLEVEL% EQU 0 (
    echo [4/7] Creating virtual environment ... FAILED
    goto fail
)
uv pip install --python "%VENV_DIR%\Scripts\python.exe" "%RELEASE_WHEEL_URL%"
if not %ERRORLEVEL% EQU 0 (
    echo [4/7] Installing from release wheel ... FAILED
    echo       Usually a dropped connection. Re-run this installer to retry.
    goto fail
)
echo [4/7] Dependencies installed (from release wheel) ... OK
goto after_deps

:deps_release_nourl
echo [4/7] Installing dependencies ... FAILED
echo       Release mode needs a wheel URL, but MINDSIGHT_RELEASE_WHEEL_URL
echo       is not set and this zip has no bundled app\ tree. Use a local-zip
echo       installer, or set MINDSIGHT_RELEASE_WHEEL_URL to the wheel asset
echo       on the GitHub Release.
goto fail

:after_deps

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
REM  Local mode points the shortcut straight at the GUI exe (unchanged). Release
REM  mode points it at a tiny launcher that exports MINDSIGHT_HOME first, since a
REM  .lnk cannot carry an environment variable and the wheel lives in
REM  site-packages (PROJECT_ROOT would otherwise resolve there).
REM  NOTE: a windowless launch (pythonw, no console flash) arrives separately via
REM  a [project.gui-scripts] move in pyproject -- do NOT change the target here.
set "SHORTCUT_TARGET=%GUI_EXE%"
if not "%INSTALL_MODE%"=="release" goto shortcut_icon
set "LAUNCH_BAT=%APP_DIR%\MindSight-Launch.bat"
> "%LAUNCH_BAT%" echo @echo off
>> "%LAUNCH_BAT%" echo set "MINDSIGHT_HOME=%APP_DIR%"
>> "%LAUNCH_BAT%" echo start "" "%GUI_EXE%"
set "SHORTCUT_TARGET=%LAUNCH_BAT%"

:shortcut_icon
REM  Give the shortcut the MindSight icon (fail-soft: a missing icon just leaves
REM  the default). Local mode uses the .ico already copied into the app tree;
REM  release mode fetches it from the GitHub Release into "%APP_DIR%".
set "ICON_PATH="
if "%INSTALL_MODE%"=="release" goto icon_release
if exist "%APP_DIR%\assets\mindsight_icon.ico" set "ICON_PATH=%APP_DIR%\assets\mindsight_icon.ico"
goto make_shortcut
:icon_release
if not defined MINDSIGHT_RELEASE_ICON_URL set "MINDSIGHT_RELEASE_ICON_URL=%RELEASE_BASE_URL%/mindsight_icon.ico"
curl -LsSf "%MINDSIGHT_RELEASE_ICON_URL%" -o "%APP_DIR%\mindsight_icon.ico"
if exist "%APP_DIR%\mindsight_icon.ico" set "ICON_PATH=%APP_DIR%\mindsight_icon.ico"

:make_shortcut
set "ICON_PS="
if defined ICON_PATH set "ICON_PS=$lnk.IconLocation = '%ICON_PATH%,0';"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = New-Object -ComObject WScript.Shell; $desktop = $s.SpecialFolders('Desktop'); $programs = $s.SpecialFolders('Programs'); foreach ($dir in @($desktop, $programs)) { $lnk = $s.CreateShortcut((Join-Path $dir 'MindSight.lnk')); $lnk.TargetPath = '%SHORTCUT_TARGET%'; $lnk.WorkingDirectory = '%APP_DIR%'; $lnk.Description = 'MindSight eye-tracking analysis'; %ICON_PS% $lnk.Save() }"
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
