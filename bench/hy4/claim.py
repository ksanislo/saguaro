#!/usr/bin/env python3
"""bench/hy4/claim.py LABEL -- command ...: the command run under a /tmp/gpu-status claim (waits for others')."""
import subprocess, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import run
label, cmd = sys.argv[1], sys.argv[3:]
run.claim(label)
try:
    sys.exit(subprocess.call(cmd))
finally:
    run.release()
