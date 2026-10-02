from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("pydantic-ai-slim", recursive=True)
datas.extend(copy_metadata("fora"))
webui = Path(SPECPATH, "..", "webui", "dist")
if webui.is_dir():
    datas.append((str(webui), "webui"))


a = Analysis(
    ["src/fora/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fora",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="fora",
)
