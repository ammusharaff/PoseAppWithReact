#!/bin/bash
set -e # Exit on error

# 1. CLEANUP (Critical for Size)
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ PoseApp.AppDir *.AppImage
rm -rf backend/sessions/* # Remove recorded videos
rm -rf backend/__pycache__
rm -rf backend/src/__pycache__

# 2. BUILD FRONTEND
echo "⚛️  Building React Frontend..."
cd frontend
npm install
npm run build
cd ..

# 3. BUILD BACKEND (PyInstaller)
echo "📦 Freezing Backend..."
# Ensure we are using the local venv's pyinstaller
./backend/.venv/bin/pyinstaller --clean --noconfirm poseapp.spec

# 4. PREPARE APPIMAGE STRUCTURE
echo "🏗️  Structuring AppImage..."
mkdir -p PoseApp.AppDir/usr/bin
mkdir -p PoseApp.AppDir/usr/share/icons
mkdir -p PoseApp.AppDir/usr/share/applications

# Copy ONLY the compiled application
cp -r dist/poseapp/* PoseApp.AppDir/usr/bin/

# Create Launcher Script
cat > PoseApp.AppDir/AppRun <<EOF
#!/bin/bash
# Get the directory where the script is located (inside the mounted AppImage)
HERE="\$(dirname "\$(readlink -f "\${0}")")"

# Set library path to look inside the AppImage's lib folder (PyInstaller puts them in /usr/bin or _internal)
# For one-dir mode, PyInstaller puts libs in the same folder as the executable or in _internal
export LD_LIBRARY_PATH="\${HERE}/usr/bin:\${HERE}/usr/bin/_internal:\${LD_LIBRARY_PATH}"

# Execute the binary
exec "\${HERE}/usr/bin/poseapp" "\$@"
EOF
chmod +x PoseApp.AppDir/AppRun

# Create Desktop Entry
cat > PoseApp.AppDir/poseapp.desktop <<EOF
[Desktop Entry]
Name=PoseApp
Exec=AppRun
Icon=poseapp
Type=Application
Categories=Utility;
EOF

# Create Dummy Icon
touch PoseApp.AppDir/poseapp.png

# 5. GENERATE APPIMAGE
echo "💿 Generating AppImage..."
if [ ! -f appimagetool-x86_64.AppImage ]; then
    wget -q https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x appimagetool-x86_64.AppImage
fi

# Run tool
ARCH=x86_64 ./appimagetool-x86_64.AppImage PoseApp.AppDir PoseApp-x86_64.AppImage

echo "✅ DONE! Created: PoseApp-x86_64.AppImage"