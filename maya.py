#!/usr/bin/env python3
"""Old name: the project is Saguaro now.  python maya.py runs saguaro.py, kept so old instructions keep working.

This is only a launcher; the installer is saguaro.py (start it with ./saguaro.sh, which makes the .venv)."""
import os
import sys
from pathlib import Path

if __name__ != "__main__":
    raise RuntimeError("maya.py is the old name: import saguaro (or run ./saguaro.sh)")
os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve().with_name("saguaro.py")), *sys.argv[1:]])
