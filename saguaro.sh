#!/bin/sh
# Saguaro for Linux and macOS: the first run checks the PC, compiles the engine, asks before it downloads the
# model, builds the pack and starts the dashboard; later runs just start it.  ./saguaro.sh --help lists the options.
# Nothing is installed system-wide: a missing Python is reported with the command that installs it.
cd "$(dirname "$0")" || exit 1
# P125q: the QFN engine page-locks its expert RAM tier (mlock), and the account's memory-lock limit decides how
# much it may: past it the tier's cold tail stays pageable and the kernel can swap cold experts (the engine says
# "mlock failed (raise ulimit -l)"). Raise the SOFT limit to the hard one - free, and what llama-swap profiles do
# with prlimit --memlock=unlimited. A HARD limit under 8 GiB (qfn_fit's smallest tier: max(8, RAM - spare),
# tools/qfn_fit.py) cannot cover even the smallest pinned tier and the script cannot raise it: say so once, with
# the one-line fix.  Linux only: ulimit -l's unit differs elsewhere, and mlock's limit is a Linux thing.
raise_memlock() {
  [ "$(uname -s)" = Linux ] || return 0
  soft=$(ulimit -Sl 2>/dev/null) || return 0
  hard=$(ulimit -Hl 2>/dev/null) || return 0
  [ -n "$hard" ] || return 0
  if [ "$soft" != "$hard" ]; then ulimit -Sl "$hard" 2>/dev/null; fi
  case "$hard" in unlimited) return 0 ;; *[!0-9]*) return 0 ;; esac   # unlimited or unreadable: never too low
  if [ "$hard" -lt 8388608 ]; then            # 8 GiB, in KiB: what `ulimit -l` shows
    echo "the hard memory-lock limit (\`ulimit -l\`) is $hard KiB, under the 8 GiB the smallest pinned expert tier"
    echo "needs: the engine's cold experts can be swapped out. One-line fix: a \"<user> - memlock unlimited\" line"
    echo "in /etc/security/limits.conf, then log out and in (a systemd unit: LimitMEMLOCK=infinity in [Service])."
  fi
}
raise_memlock
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3 python; do
    if command -v $c >/dev/null 2>&1 && ok_py $c; then
      PY=$c; break
    fi
  done
  # macOS ships Python 3.9: a standalone Python 3.12 goes into .python (no sudo, no Homebrew; pinned by SHA-256,
  # the build Strata's setup.sh uses)
  if [ -z "$PY" ] && [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    if [ ! -x .python/python/bin/python3 ]; then
      PBS="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-aarch64-apple-darwin-install_only.tar.gz"
      echo "Python 3.10+ is needed; downloading a standalone Python 3.12 into .python (25 MB) ..."
      mkdir -p .python && curl -fL --retry 3 -o .python/python.tar.gz "$PBS" || { echo "could not download Python"; exit 1; }
      if [ "$(shasum -a 256 .python/python.tar.gz | cut -d" " -f1)" != "83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8" ]; then
        rm -f .python/python.tar.gz; echo "the Python download does not match its pinned SHA-256"; exit 1
      fi
      tar xzf .python/python.tar.gz -C .python && rm -f .python/python.tar.gz
    fi
    PY=.python/python/bin/python3
  fi
  if [ -z "$PY" ]; then
    echo "Python 3.10 or newer with venv is needed. Install it, then run ./saguaro.sh again:"
    echo "  Ubuntu/Debian: sudo apt-get install -y python3 python3-venv python3-pip"
    echo "  Fedora:        sudo dnf install -y python3 python3-pip"
    echo "  Arch:          sudo pacman -S python python-pip"
    echo "  macOS:         brew install python  (or python.org's installer)"
    exit 1
  fi
  # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
  $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt-get install -y python3-venv"; exit 1; }
fi
exec .venv/bin/python saguaro.py "$@"
