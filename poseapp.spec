# poseapp.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# 1. Collect minimal dependencies for backend
datas = []
binaries = []
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',
    'python_multipart',
    'mediapipe',
    'src', # Explicitly hint src package
    'src.poseapp',
    'src.poseapp.pose_engine'
]

# 2. Bundle MediaPipe (Essential)
tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# 3. Add Project Assets (Only what is needed)
datas += [
    ('backend/src/models', 'src/models'),       # AI Models
    ('backend/src/assets', 'src/assets'),       # GIFs
    ('frontend/dist', 'static'),                # React Frontend
]

# 4. Safe Excludes
excludes = [
    # GUI & Science
    'tkinter', 'notebook', 'ipython', 'pandas', 'unittest', 'scipy', 'matplotlib.tests', 'numpy.tests',
    
    # Heavy AI Frameworks (We only need tflite_runtime)
    'tensorflow', 
    'tensorboard', 
    'torch', 
    'torchvision', 
    'keras', 
    'jax', 
    'jaxlib',
    
    # NVIDIA CUDA Bloat (The most likely culprit)
    'nvidia', 
    'nvidia-cublas-cu11', 
    'nvidia-cudnn-cu11', 
    'nvidia-cuda-runtime-cu11', 
    'nvidia-cuda-cupti-cu11'
]

a = Analysis(
    ['backend/app/main.py'],
    pathex=['backend'],  # <--- THIS IS THE FIX
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='poseapp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='poseapp',
)