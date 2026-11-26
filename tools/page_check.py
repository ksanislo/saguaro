"""tools/page_check.py <base-url> [--key-file F] - P124d: a real browser page load in the release checks.
1.2.2 shipped a page that died on load (startingView(null) threw on the first paint) while every API smoke passed -
none of them opened the page. This one does: it loads <base-url> in headless chromium (chromium, chromium-browser
or google-chrome, whichever this PC has; a clear SKIP line and exit 0 when it has none), waits for the app to
render - the shell (.shell), or the loading screen (.start) while the model is still loading - and FAILS (exit 1)
on any uncaught JS error or console error (the CONSOLE lines chromium writes to stderr with --enable-logging=stderr)
or when neither element renders within 30 s. One PASS/FAIL line with the reason; exit 0 on PASS or SKIP, 1 on FAIL.

A page behind an API key renders its shell without the key (the app asks for it inside), so the browser never sees
the key: --key-file F holds the install's key and only proves it against /v1/models (a refused key is a FAIL; while
the model is still loading the key cannot be checked, and the line says so).

    python3 tools/page_check.py http://127.0.0.1:8095
    python3 tools/page_check.py http://127.0.0.1:8095 --key-file saguaro-api-key.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

RENDER_S = 30                       # the wait for .shell / .start to show up (the plan's 30 s)
BROWSERS = ("chromium", "chromium-browser", "google-chrome")
BROWSER_PATHS = (                   # the usual installs when the name is not on the PATH (macOS, Windows)
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)

# one chromium stderr line that carries a console message, in both formats it has written: the new
# "[pid:tid:stamp:INFO:CONSOLE:1] "msg", source: url (1)" and the old "[stamp:INFO:CONSOLE(34)] "msg", source: url".
CONSOLE_LINE = re.compile(r':[A-Z]+:CONSOLE[(:]\d*\]\s+"((?:[^"\\]|\\.)*)"')


def find_browser() -> str | None:
    """the chromium this PC has, or None"""
    for name in BROWSERS:
        p = shutil.which(name)
        if p:
            return p
    for p in BROWSER_PATHS:
        if os.path.exists(p):
            return p
    return None


def console_errors(log: str) -> list[str]:
    """the message of every CONSOLE line in a chromium stderr log - a page that logs nothing answers []
    (a clean page's log still carries chromium's own WARNING lines: those are not console messages)"""
    out = []
    for line in log.splitlines():
        m = CONSOLE_LINE.search(line)
        if m:
            try:
                out.append(json.loads(m.group(0)[m.group(0).index('"'):]))   # the quoted part is JSON-escaped
            except ValueError:
                out.append(m.group(1))
    return out


def rendered(dom: str) -> str:
    """what the dumped DOM shows: "shell" (the app), "start" (the loading screen) or "" (neither). The class tokens
    are matched whole: "shell--closed" is the shell, "start__card" is the card INSIDE .start and never .start."""
    if re.search(r'class="[^"]*\bshell\b', dom):
        return "shell"
    if re.search(r'class="[^"]*\bstart\b', dom):
        return "start"
    return ""


def decide(errors: list[str], dom: str) -> tuple[bool, str]:
    """the verdict on fixed inputs (the tests' subject): any console error fails, a page that drew neither
    element in the wait fails, the shell or a waiting loading screen passes"""
    if errors:
        return False, f"{len(errors)} console error(s), first: {errors[0]}"
    what = rendered(dom)
    if what == "shell":
        return True, "the page drew its shell"
    if what == "start":
        return True, "the page drew the loading screen (the model is still loading)"
    return False, f"neither .shell nor .start rendered within {RENDER_S} s"


def load_page(base: str, browser: str) -> tuple[str, str, str]:
    """one headless load: (the dumped DOM, the stderr log, a problem - "" when chromium ran to the end).
    The virtual-time budget runs the page's timers and waits for its fetches; --timeout is chromium's own wall
    clock, the subprocess one the safety net."""
    profile = tempfile.mkdtemp(prefix="page-check-")
    try:
        url = base if urllib.parse.urlsplit(base).path else base + "/"   # (an http root URL needs its "/")
        argv = [browser, "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                "--no-default-browser-check", f"--user-data-dir={profile}",
                "--enable-logging=stderr", "--v=0",
                f"--virtual-time-budget={RENDER_S * 1000}", f"--timeout={RENDER_S * 1000}",
                "--dump-dom", url]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=RENDER_S + 15)
        except subprocess.TimeoutExpired:
            return "", "", f"chromium did not finish within {RENDER_S + 15} s"
        problem = "" if r.returncode == 0 else f"chromium exited {r.returncode}"
        return r.stdout, r.stderr, problem
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def key_check(base: str, key: str) -> tuple[bool, str]:
    """the --key-file key against /v1/models: the page needs no key, so this only proves the file holds the
    install's own; a 503 means the model is still loading (the page then shows the loading screen, not a shell)"""
    req = urllib.request.Request(base + "/v1/models", headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return (r.status < 400, f"API key OK ({r.status})" if r.status < 400 else f"API key check got HTTP {r.status}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"the key file's key was refused (HTTP {e.code})"
        if e.code == 503:
            return True, "the key was not checked (the model is still loading)"
        return False, f"the key check got HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return False, f"the key check could not reach {base}: {e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="the page's address, e.g. http://127.0.0.1:8095")
    ap.add_argument("--key-file", help="a file holding the install's API key; it is checked against /v1/models "
                                       "(the page renders its shell without it - the app asks for the key inside)")
    a = ap.parse_args(argv)
    base = a.base.rstrip("/")
    browser = find_browser()
    if not browser:
        print(f"SKIP {base}: no chromium, chromium-browser or google-chrome on this PC - the page was not loaded")
        return 0
    key_note = ""
    if a.key_file:
        try:
            key = open(a.key_file, encoding="utf-8-sig").read().strip()
        except OSError as e:
            print(f"FAIL {base}: the key file {a.key_file} cannot be read: {e}")
            return 1
        if not key:
            key_note = " (the key file is empty, the key was not checked)"
        else:
            ok, key_note = key_check(base, key)
            if not ok:
                print(f"FAIL {base}: {key_note}")
                return 1
            key_note = f" ({key_note})"
    dom, log, problem = load_page(base, browser)
    ok, reason = decide(console_errors(log), dom)
    if problem:
        reason += f" ({problem})"
    print(f"{'PASS' if ok else 'FAIL'} {base}: {reason}{key_note}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
