import os, sys

GUIDE_DIRS= []

_meipass = getattr(sys, "_MEIPASS", None)
if _meipass:
    GUIDE_DIRS.append(os.path.join(_meipass, "assets", "guides"))
if getattr(sys, "frozen", False):
    exe_dir = os.path.dirname(sys.executable)
    GUIDE_DIRS.append(os.path.join(exe_dir, "assets", "guides"))
GUIDE_DIRS.append(os.path.join(os.getcwd(), "assets", "guides"))  # dev run

def resource_path(rel_path: str) -> str:
    """
    Return an absolute path for data files both in dev and in PyInstaller.
    Search order:
      1) _MEIPASS (PyInstaller temp unpack dir)
      2) Current Working Directory
      3) Project root (two levels up from this utils/ folder)
    """
    # normalize to forward slashes and strip leading slashes
    rel_path = rel_path.replace("\\", "/").lstrip("/")

    # 1) PyInstaller extraction dir
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = os.path.join(meipass, rel_path)
        if os.path.exists(p):
            return p

    # 2) CWD
    p = os.path.join(os.getcwd(), rel_path)
    if os.path.exists(p):
        return p

    # 3) project root (utils/../../)
    here = os.path.abspath(os.path.dirname(__file__))
    proj_root = os.path.abspath(os.path.join(here, "..", ".."))
    p3 = os.path.join(proj_root, rel_path)
    if os.path.exists(p3):
        return p3

    # last resort: return project-root guess (helps error messages)
    return p3
