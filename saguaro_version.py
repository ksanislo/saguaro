"""What this checkout calls its own version: saguaro.py's banner and the dashboard's About > Updates both ask here,
so neither says something different.  Saguaro has no releases of its own yet, so a git clone is named by its commit;
a folder without git by the VERSION line it was built from."""
from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def saguaro_version(version: str, root: Path | None = None) -> str:
    """What the banner calls this version: the git commit of a clone (Saguaro has no releases of its own yet), else
    the VERSION line it was built from."""
    root = Path(root) if root is not None else HERE
    r = ""
    if (root / ".git").exists():
        try:
            r = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %cs"], capture_output=True,
                               text=True, timeout=10, stdin=subprocess.DEVNULL).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            r = ""
    return f"{r.split()[0]} ({r.split()[1]})" if len(r.split()) == 2 else f"v{version}"
