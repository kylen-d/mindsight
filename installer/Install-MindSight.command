#!/bin/bash
# ============================================================================
#  MindSight -- macOS installer (SP4.0 interim)
#
#  Double-click this file to install MindSight into your home folder.
#  It installs the uv package manager, a managed Python 3.12, the MindSight
#  source tree, all locked dependencies, downloads the required model weights,
#  and installs a MindSight.app into /Applications (with a Desktop link).
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
# The installed app is a real .app bundle so the Dock shows the MindSight name
# and icon (not a generic "Python"). Preferred in /Applications; falls back to
# ~/Applications when /Applications is not writable without sudo.
APP_BUNDLE_ID="io.github.kylen-d.mindsight"
OLD_DESKTOP_LAUNCHER="$HOME/Desktop/MindSight.command"
DESKTOP_LINK="$HOME/Desktop/MindSight.app"

# ---- Install mode ---------------------------------------------------------
# Local-zip mode (the default): the release zip bundles an app/ source tree
# next to this script; MindSight is installed editable from it and PROJECT_ROOT
# resolves to that tree (MINDSIGHT_HOME stays unset). Release mode: when no
# app/ tree is bundled, MindSight is installed non-editable from a wheel asset
# published on the GitHub Release, and MINDSIGHT_HOME points every data path
# (Weights/Outputs/Projects) at "$APP_DIR" so nothing lands under site-packages.
# Release-mode assets (wheel, weights manifest, and the pipeline preset) live on
# the GitHub Release; RELEASE_BASE_URL is the download base for that tag and the
# wheel/manifest/preset URLs derive from it. All three are overridable via env so
# a future release can point at a new tag without editing this script.
RELEASE_BASE_URL="${MINDSIGHT_RELEASE_BASE_URL:-https://github.com/kylen-d/mindsight/releases/download/v1.0.0}"
RELEASE_WHEEL_URL="${MINDSIGHT_RELEASE_WHEEL_URL:-$RELEASE_BASE_URL/mindsight-1.0.0-py3-none-any.whl}"
RELEASE_MANIFEST_URL="${MINDSIGHT_RELEASE_MANIFEST_URL:-$RELEASE_BASE_URL/weights_manifest.json}"
RELEASE_PRESET_URL="${MINDSIGHT_RELEASE_PRESET_URL:-$RELEASE_BASE_URL/pipeline_known_good.yaml}"
RELEASE_LOWPOWER_URL="${MINDSIGHT_RELEASE_LOWPOWER_URL:-$RELEASE_BASE_URL/pipeline_low_power.yaml}"
RELEASE_ICON_URL="${MINDSIGHT_RELEASE_ICON_URL:-$RELEASE_BASE_URL/mindsight_icon.png}"
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

# --------------------------------------------------------------------------
#  build_app_bundle <bundle-path>
#
#  Assemble a minimal MindSight.app at the given path. The MacOS/MindSight
#  launcher keeps the same semantics as the former Desktop .command (release
#  mode exports MINDSIGHT_HOME; cd "$APP_DIR"; exec the GUI). exec keeps the
#  PID so the Dock adopts the bundle's name and icon. Icon assembly is
#  fail-soft: a missing PNG (e.g. a skipped release download) just omits the
#  icon and never fails the install. Uses globals INSTALL_MODE, APP_DIR,
#  VENV_DIR, SRC_DIR, RELEASE_ICON_URL, APP_BUNDLE_ID.
# --------------------------------------------------------------------------
build_app_bundle() {
    local bundle="$1"
    local contents="$bundle/Contents"
    local macos_dir="$contents/MacOS"
    local resources="$contents/Resources"
    mkdir -p "$macos_dir" "$resources" || return 1

    # ---- Executable launcher (identical semantics to the old launcher) ----
    local home_line
    if [ "$INSTALL_MODE" = "release" ]; then
        home_line="export MINDSIGHT_HOME=\"$APP_DIR\""
    else
        home_line="# local-zip install: MINDSIGHT_HOME unset (PROJECT_ROOT = app tree)"
    fi
    cat > "$macos_dir/MindSight" <<EOF
#!/bin/bash
# MindSight launcher -- opens the graphical user interface.
$home_line
cd "$APP_DIR"
exec "$VENV_DIR/bin/mindsight-gui"
EOF
    chmod +x "$macos_dir/MindSight" || return 1

    # ---- Version strings (fail-soft to 0) ----
    local version
    version="$("$VENV_DIR/bin/python" -c "import mindsight; print(mindsight.__version__)" 2>/dev/null)"
    if [ -z "$version" ]; then
        version="0"
    fi

    # ---- Icon: assemble MindSight.icns from the PNG master (fail-soft) ----
    # Local mode reads the bundled app tree; release mode fetches the PNG from
    # the release. sips + iconutil both ship with macOS.
    local icon_png=""
    local icon_tmp=""
    if [ "$INSTALL_MODE" = "release" ]; then
        icon_tmp="$(mktemp -t mindsight_icon).png"
        if curl -LsSf "$RELEASE_ICON_URL" -o "$icon_tmp" 2>/dev/null && [ -s "$icon_tmp" ]; then
            icon_png="$icon_tmp"
        fi
    else
        if [ -f "$SRC_DIR/app/assets/mindsight_icon.png" ]; then
            icon_png="$SRC_DIR/app/assets/mindsight_icon.png"
        fi
    fi
    if [ -n "$icon_png" ]; then
        local iconset
        iconset="$(mktemp -d -t MindSight_iconset)/MindSight.iconset"
        if mkdir -p "$iconset"; then
            sips -z 16 16     "$icon_png" --out "$iconset/icon_16x16.png"      >/dev/null 2>&1
            sips -z 32 32     "$icon_png" --out "$iconset/icon_16x16@2x.png"   >/dev/null 2>&1
            sips -z 32 32     "$icon_png" --out "$iconset/icon_32x32.png"      >/dev/null 2>&1
            sips -z 64 64     "$icon_png" --out "$iconset/icon_32x32@2x.png"   >/dev/null 2>&1
            sips -z 128 128   "$icon_png" --out "$iconset/icon_128x128.png"    >/dev/null 2>&1
            sips -z 256 256   "$icon_png" --out "$iconset/icon_128x128@2x.png" >/dev/null 2>&1
            sips -z 256 256   "$icon_png" --out "$iconset/icon_256x256.png"    >/dev/null 2>&1
            sips -z 512 512   "$icon_png" --out "$iconset/icon_256x256@2x.png" >/dev/null 2>&1
            sips -z 512 512   "$icon_png" --out "$iconset/icon_512x512.png"    >/dev/null 2>&1
            iconutil -c icns "$iconset" -o "$resources/MindSight.icns" >/dev/null 2>&1 || true
        fi
        rm -rf "$(dirname "$iconset")" 2>/dev/null || true
    fi
    [ -n "$icon_tmp" ] && rm -f "$icon_tmp" 2>/dev/null

    # ---- Info.plist ----
    # NSCameraUsageDescription is REQUIRED: the GUI records live study sessions
    # from the camera, and without this key macOS denies camera access to the
    # bundled app. CFBundleIconFile names the .icns above (extension implied).
    cat > "$contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleName</key>
    <string>MindSight</string>
    <key>CFBundleDisplayName</key>
    <string>MindSight</string>
    <key>CFBundleExecutable</key>
    <string>MindSight</string>
    <key>CFBundleIdentifier</key>
    <string>$APP_BUNDLE_ID</string>
    <key>CFBundleIconFile</key>
    <string>MindSight</string>
    <key>CFBundleShortVersionString</key>
    <string>$version</string>
    <key>CFBundleVersion</key>
    <string>$version</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>MindSight uses the camera to record live study sessions.</string>
</dict>
</plist>
EOF
    return 0
}

# Test hook: a harness may source this file with MINDSIGHT_INSTALLER_SOURCE_ONLY
# set to exercise build_app_bundle() in isolation without running the install.
# Never set in normal use (double-click / right-click > Open).
if [ -n "${MINDSIGHT_INSTALLER_SOURCE_ONLY:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

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
    # not a source tree; MINDSIGHT_HOME will point here. The wheel ships only the
    # Python packages, so the weights manifest (mindsight-weights reads it from
    # PROJECT_ROOT) and the pipeline_known_good.yaml preset are fetched here.
    export MINDSIGHT_HOME="$APP_DIR"
    if ! curl -LsSf "$RELEASE_MANIFEST_URL" -o "$APP_DIR/weights_manifest.json"; then
        fail "3 (copy application files)" "Could not download the weights manifest from the release."
    fi
    if ! mkdir -p "$APP_DIR/configs"; then
        fail "3 (copy application files)" "Could not create the configs folder in \"$APP_DIR\"."
    fi
    # Low-power variant is best-effort: absence must not fail the install.
    curl -LsSf "$RELEASE_LOWPOWER_URL" -o "$APP_DIR/configs/pipeline_low_power.yaml" || true
    if ! curl -LsSf "$RELEASE_PRESET_URL" -o "$APP_DIR/configs/pipeline_known_good.yaml"; then
        fail "3 (copy application files)" "Could not download the pipeline preset from the release."
    fi
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
    if ! uv venv --clear --python 3.12 "$VENV_DIR"; then
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
#  [6/6] Install the MindSight.app bundle (Dock name + icon)
# ==========================================================================
echo "[6/6] Installing the MindSight app ..."
# A real .app bundle makes the Dock show "MindSight" with the MindSight icon
# instead of a generic "Python". Prefer /Applications; fall back to
# ~/Applications when /Applications is not writable without sudo, or when a
# foreign MindSight.app (different bundle identifier) already sits there.
APP_TARGET=""
for candidate in "/Applications/MindSight.app" "$HOME/Applications/MindSight.app"; do
    parent="$(dirname "$candidate")"
    if [ -d "$candidate" ]; then
        # Re-run safety: only reclaim a bundle that is unmistakably ours.
        existing_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' \
            "$candidate/Contents/Info.plist" 2>/dev/null)"
        if [ "$existing_id" = "$APP_BUNDLE_ID" ]; then
            if rm -rf "$candidate" 2>/dev/null; then
                APP_TARGET="$candidate"
                break
            fi
        else
            echo "      Note: \"$candidate\" exists with a different identifier;"
            echo "      leaving it untouched and trying the next location."
            continue
        fi
    fi
    # No existing bundle here (or we just removed ours): can we create it?
    if [ ! -d "$parent" ]; then
        mkdir -p "$parent" 2>/dev/null || continue
    fi
    if [ -w "$parent" ]; then
        APP_TARGET="$candidate"
        break
    fi
done
if [ -z "$APP_TARGET" ]; then
    fail "6 (install app)" \
         "Could not find a writable location for MindSight.app (tried /Applications and ~/Applications)."
fi
if ! build_app_bundle "$APP_TARGET"; then
    fail "6 (install app)" "Could not assemble the MindSight app bundle at \"$APP_TARGET\"."
fi
echo "[6/6] Installed \"$APP_TARGET\" ... OK"

# Desktop: retire the old MindSight.command launcher and drop a link to the app.
if [ -e "$OLD_DESKTOP_LAUNCHER" ]; then
    rm -f "$OLD_DESKTOP_LAUNCHER" 2>/dev/null || true
fi
ln -sfn "$APP_TARGET" "$DESKTOP_LINK" 2>/dev/null || true

echo
echo "============================================================"
echo "  MindSight install: PASS"
echo
echo "  Launch it from the \"MindSight\" app on your Desktop or in"
echo "  Finder, or run directly:"
echo "    \"$VENV_DIR/bin/mindsight-gui\""
echo
echo "  Installed in: \"$INSTALL_DIR\""
echo "  App bundle:   \"$APP_TARGET\""
echo "  Your projects, weights and outputs live under \"$APP_DIR\"."
echo "  To uninstall: delete that folder, the app bundle, and the"
echo "  Desktop link."
echo "============================================================"
pause_and_exit 0
