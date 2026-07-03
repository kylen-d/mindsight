#!/bin/bash
# ============================================================================
#  MindSight -- macOS installer (SP4.0 interim)
#
#  Double-click this file to install MindSight into your home folder.
#  It installs the uv package manager, a managed Python 3.12, the MindSight
#  source tree, all locked dependencies, downloads the required model weights,
#  and puts a MindSight launcher on your Desktop.
#
#  Re-running this file updates an existing install (it is safe to run again).
#  ASCII output only. Explicit per-step error checks; no dev virtualenvs.
# ============================================================================

set -uo pipefail

# ---- Locations ------------------------------------------------------------
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/MindSight"
APP_DIR="$INSTALL_DIR/app"
VENV_DIR="$INSTALL_DIR/venv"
UV_BIN="$HOME/.local/bin"
LAUNCHER="$HOME/Desktop/MindSight.command"

# ---- Install mode ---------------------------------------------------------
# Local-zip mode (the default): the release zip bundles an app/ source tree
# next to this script; MindSight is installed editable from it and PROJECT_ROOT
# resolves to that tree (MINDSIGHT_HOME stays unset). Release mode: when no
# app/ tree is bundled, MindSight is installed non-editable from a wheel asset
# published on the GitHub Release, and MINDSIGHT_HOME points every data path
# (Weights/Outputs/Projects) at "$APP_DIR" so nothing lands under site-packages.
# The wheel URL is supplied via MINDSIGHT_RELEASE_WHEEL_URL (a real release-mode
# zip will carry it). Local-zip mode remains the default until v1.0 assets ship.
RELEASE_WHEEL_URL="${MINDSIGHT_RELEASE_WHEEL_URL:-}"
if [ -d "$SRC_DIR/app" ]; then
    INSTALL_MODE="local"
else
    INSTALL_MODE="release"
fi

pause_and_exit() {
    echo
    read -r -p "Press Return to close this window... " _ || true
    exit "$1"
}

fail() {
    # fail "<step label>" "<plain-English cause>"
    echo
    echo "============================================================"
    echo "  MindSight install: FAILED at step $1"
    echo
    echo "  $2"
    echo
    echo "  Nothing was pushed to your system beyond the partial folder"
    echo "  at \"$INSTALL_DIR\". You can delete that folder and run this"
    echo "  installer again. See INSTALL-MACOS.md for troubleshooting."
    echo "============================================================"
    pause_and_exit 1
}

echo
echo "============================================================"
echo "  MindSight installer"
echo "  Target folder: \"$INSTALL_DIR\""
echo "============================================================"
echo

# ==========================================================================
#  [1/6] Locate or install uv
# ==========================================================================
if command -v uv >/dev/null 2>&1; then
    echo "[1/6] uv already installed ... OK"
elif [ -x "$UV_BIN/uv" ]; then
    echo "[1/6] uv found in \"$UV_BIN\" ... OK"
else
    echo "[1/6] Installing uv package manager ..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        fail "1 (install uv)" \
             "Could not download or install uv. Check your internet connection and that your network allows downloads from astral.sh."
    fi
    echo "[1/6] Installing uv ... OK"
fi
# Make sure uv is on PATH for the rest of this session.
export PATH="$UV_BIN:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    fail "1 (install uv)" "uv is not on PATH after install."
fi

# ==========================================================================
#  [2/6] Install a managed Python 3.12
# ==========================================================================
echo "[2/6] Ensuring managed Python 3.12 ..."
if ! uv python install 3.12; then
    fail "2 (install Python 3.12)" "Could not install a managed Python 3.12."
fi
echo "[2/6] Python 3.12 ready ... OK"

# ==========================================================================
#  [3/6] Deploy the MindSight source tree
# ==========================================================================
echo "[3/6] Preparing \"$APP_DIR\" (mode: $INSTALL_MODE) ..."
if ! mkdir -p "$APP_DIR"; then
    fail "3 (copy application files)" "Could not create the install folder \"$APP_DIR\"."
fi
if [ "$INSTALL_MODE" = "local" ]; then
    # rsync without --delete: update files in place but never purge a returning
    # user's Outputs/ or Projects/ (mirrors the Windows robocopy /E behavior).
    if ! rsync -a "$SRC_DIR/app/" "$APP_DIR/"; then
        fail "3 (copy application files)" "Could not copy the application files into \"$APP_DIR\"."
    fi
    echo "[3/6] Application files in place ... OK"
else
    # Release mode: "$APP_DIR" is the data home (Weights/Outputs/Projects),
    # not a source tree; MINDSIGHT_HOME will point here.
    export MINDSIGHT_HOME="$APP_DIR"
    echo "[3/6] Data home ready at \"$APP_DIR\" ... OK"
fi

# ==========================================================================
#  [4/6] Create the virtual environment and install locked dependencies
# ==========================================================================
echo "[4/6] Installing dependencies from the locked manifest (this can take a while) ..."
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
# Default torch on macOS is the CPU/MPS build from the committed lock -- no
# separate GPU step (Apple Silicon uses MPS through that same wheel).
if [ "$INSTALL_MODE" = "local" ]; then
    if ! uv sync --frozen --python 3.12 --project "$APP_DIR"; then
        fail "4 (install dependencies)" \
             "Dependency install failed -- usually a dropped connection. Re-run this installer to retry."
    fi
    echo "[4/6] Dependencies installed (editable) ... OK"
else
    if [ -z "$RELEASE_WHEEL_URL" ]; then
        fail "4 (install dependencies)" \
             "Release mode needs a wheel URL, but MINDSIGHT_RELEASE_WHEEL_URL is not set. This zip has no bundled app/ tree. Use a local-zip installer, or set MINDSIGHT_RELEASE_WHEEL_URL to the wheel asset on the GitHub Release."
    fi
    if ! uv venv --python 3.12 "$VENV_DIR"; then
        fail "4 (install dependencies)" "Could not create the virtual environment at \"$VENV_DIR\"."
    fi
    if ! uv pip install --python "$VENV_DIR/bin/python" "$RELEASE_WHEEL_URL"; then
        fail "4 (install dependencies)" \
             "Could not install MindSight from the release wheel -- usually a dropped connection. Re-run this installer to retry."
    fi
    echo "[4/6] Dependencies installed (from release wheel) ... OK"
fi

# ==========================================================================
#  [5/6] Download the required model weights (headless)
# ==========================================================================
echo "[5/6] Downloading required model weights (Gaze-LLE, MobileGaze, YOLO) ..."
if ! "$VENV_DIR/bin/mindsight-weights" --required; then
    fail "5 (download weights)" \
         "A network hiccup during the model download. Check your internet connection, then re-run this installer."
fi
echo "[5/6] Required weights present and verified ... OK"

# ==========================================================================
#  [6/6] Create the MindSight launcher on the Desktop
# ==========================================================================
echo "[6/6] Creating the MindSight launcher on your Desktop ..."
# In release mode the package lives under site-packages, so MINDSIGHT_HOME must
# be exported for the launcher to keep data under "$APP_DIR". In local mode the
# var stays unset and PROJECT_ROOT resolves to the source tree (unchanged).
if [ "$INSTALL_MODE" = "release" ]; then
    LAUNCHER_HOME_LINE="export MINDSIGHT_HOME=\"$APP_DIR\""
else
    LAUNCHER_HOME_LINE="# local-zip install: MINDSIGHT_HOME unset (PROJECT_ROOT = app tree)"
fi
if ! cat > "$LAUNCHER" <<EOF
#!/bin/bash
# MindSight launcher -- opens the graphical user interface.
$LAUNCHER_HOME_LINE
cd "$APP_DIR"
exec "$VENV_DIR/bin/mindsight-gui"
EOF
then
    fail "6 (create launcher)" "Could not write the launcher to \"$LAUNCHER\"."
fi
if ! chmod +x "$LAUNCHER"; then
    fail "6 (create launcher)" "Could not make the launcher executable at \"$LAUNCHER\"."
fi
echo "[6/6] Launcher created ... OK"

echo
echo "============================================================"
echo "  MindSight install: PASS"
echo
echo "  Launch it by double-clicking the \"MindSight\" launcher on"
echo "  your Desktop, or run directly:"
echo "    \"$VENV_DIR/bin/mindsight-gui\""
echo
echo "  Installed in: \"$INSTALL_DIR\""
echo "  Your projects, weights and outputs live under \"$APP_DIR\"."
echo "  To uninstall: delete that folder and the Desktop launcher."
echo "============================================================"
pause_and_exit 0
