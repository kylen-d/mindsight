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
echo "[3/6] Copying application files to \"$APP_DIR\" ..."
if ! mkdir -p "$APP_DIR"; then
    fail "3 (copy application files)" "Could not create the install folder \"$APP_DIR\"."
fi
# rsync without --delete: update files in place but never purge a returning
# user's Outputs/ or Projects/ (mirrors the Windows robocopy /E behavior).
if ! rsync -a "$SRC_DIR/app/" "$APP_DIR/"; then
    fail "3 (copy application files)" "Could not copy the application files into \"$APP_DIR\"."
fi
echo "[3/6] Application files in place ... OK"

# ==========================================================================
#  [4/6] Create the virtual environment and install locked dependencies
# ==========================================================================
echo "[4/6] Installing dependencies from the locked manifest (this can take a while) ..."
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
# Default torch on macOS is the CPU/MPS build from the committed lock -- no
# separate GPU step (Apple Silicon uses MPS through that same wheel).
if ! uv sync --frozen --python 3.12 --project "$APP_DIR"; then
    fail "4 (install dependencies)" \
         "Dependency install failed -- usually a dropped connection. Re-run this installer to retry."
fi
echo "[4/6] Dependencies installed (editable) ... OK"

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
if ! cat > "$LAUNCHER" <<EOF
#!/bin/bash
# MindSight launcher -- opens the graphical user interface.
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
