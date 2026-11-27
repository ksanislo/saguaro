"""What this checkout calls its own version: saguaro.py's banner and the dashboard's About > Updates both ask here,
so neither says something different, and --bench's header and the server's /health ask here too (installed_version:
the VERSION file's line, never a stale or an upstream string).  Saguaro has no releases of its own yet, so a git
clone is named by its commit; a folder without git by the VERSION line it was built from."""
from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def installed_version(root: Path | None = None) -> str | None:
    """The installed checkout's own version: the VERSION file's line (1.2.3).  --bench's header and the server's
    /health print this, asked here rather than reading VERSION themselves, so neither can show a stale or an
    upstream string.  None: this folder has no VERSION file."""
    root = Path(root) if root is not None else HERE
    try:
        return (root / "VERSION").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


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
