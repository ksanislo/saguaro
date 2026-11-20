#!/usr/bin/env python3
"""tools/saguaro/migrate_configs.py - migrate OUR configs to the SAGUARO_* setting names (rename_env.py's rule for
config files, plus dropping the gone --engine maya / --engine qfn from llama-swap command lines).

    python3 tools/saguaro/migrate_configs.py [--write] <config.json | llama-swap.yaml>...

Default prints the diff; --write rewrites each file, keeping the old copy as <file>.bak-<date>-pre-env-rename.
What it changes:
  * a JSON config's "env" entries, at any depth: key OLD_X -> SAGUARO_X; a MAYA_ and a STRATA_ key of the same
    suffix collapse into the one SAGUARO_ key, the MAYA_ value winning (what the old alias code said);
  * a llama-swap model list (YAML or the old .properties style is not read - YAML only): `--env OLD_X=v` and
    `--env OLD_X v` inside a `command:` string, `env:` list items (`- OLD_X=v`) and `env:` mapping keys;
  * drops `--engine maya` / `--engine maya`-with-`=` / `--engine qfn` from those command lines - the shared server
    picks the driver from the config itself now (serve/server.py: engine_kind).
Exe names, other args, model names and everything else stay word for word.  Idempotent; run it again on regenerated
configs.  REPORT-env-rename.md carries the diffs for the ref-configs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import re
import sys
from pathlib import Path

NEW = "SAGUARO_"
SUF = r"(MAYA|STRATA)_([A-Z0-9_]+)"
ENV_IN_ARG = re.compile(rf"--env {SUF}=")
ENV_SEP_ARG = re.compile(rf"--env {SUF}(\s+)(\S)")
LIST_ITEM = re.compile(rf"^(\s*-\s+)({SUF})(=)", re.M)
MAP_KEY = re.compile(rf"^(\s*)({SUF})(\s*:)", re.M)
ENGINE_DROP = re.compile(r"\s--engine(?:=|\s+)(?:maya|qfn)\b")
OLD_FULL = re.compile(rf"\b{SUF}\b")


def rename_key(key: str) -> str:
    m = re.fullmatch(SUF, key)
    return NEW + m.group(2) if m else key


def migrate_json_obj(obj):
    """"env" dicts' keys renamed, recursively; MAYA_ wins a suffix clash.  Returns (migrated, n)."""
    if isinstance(obj, dict):
        out, n = {}, 0
        for k, v in obj.items():
            if k == "env" and isinstance(v, dict):
                n += 1 if any(isinstance(ek, str) and re.fullmatch(SUF, ek) for ek in v) else 0
                win, outenv = {}, {}
                for ek in v:
                    m = re.fullmatch(SUF, ek) if isinstance(ek, str) else None
                    if m and (m.group(2) not in win or win[m.group(2)] == "STRATA" and m.group(1) == "MAYA"):
                        win[m.group(2)] = m.group(1)                       # MAYA_ wins its suffix, as the alias code
                for ek, ev in v.items():
                    nv, c = migrate_json_obj(ev)
                    n += c
                    m = re.fullmatch(SUF, ek) if isinstance(ek, str) else None
                    if m:
                        if win[m.group(2)] != m.group(1):
                            continue                                       # (the loser of a suffix clash goes away)
                        ek = NEW + m.group(2)
                    if ek not in outenv:
                        outenv[ek] = nv
                out["env"] = outenv
            else:
                nv, c = migrate_json_obj(v)
                out[k] = nv
                n += c
        return out, n
    if isinstance(obj, list):
        vals, n = [], 0
        for v in obj:
            nv, c = migrate_json_obj(v)
            vals.append(nv)
            n += c
        return vals, n
    return obj, 0


def migrate_text(text: str):
    """llama-swap command lines / env blocks."""
    text, n1 = ENGINE_DROP.subn("", text)
    text, n2 = ENV_IN_ARG.subn(lambda m: "--env " + NEW + m.group(2) + "=", text)
    text, n3 = ENV_SEP_ARG.subn(lambda m: "--env " + NEW + m.group(2) + m.group(3) + m.group(4), text)
    text, n4 = LIST_ITEM.subn(lambda m: m.group(1) + NEW + m.group(4) + m.group(5), text)
    text, n5 = MAP_KEY.subn(lambda m: m.group(1) + NEW + m.group(4) + m.group(5), text)
    return text, n1 + n2 + n3 + n4 + n5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the migrated file (old copy: .bak-<date>-pre-env-rename)")
    ap.add_argument("paths", nargs="+", type=Path)
    a = ap.parse_args()
    stamp = dt.date.today().isoformat()
    changed = 0
    for p in a.paths:
        try:
            raw = p.read_bytes()
        except OSError as e:
            print(f"{p}: {e}")
            continue
        if b"\0" in raw[:8000]:
            print(f"{p}: binary, skipped")
            continue
        text = raw.decode("utf-8-sig")
        if p.suffix.lower() == ".json":
            try:
                migrated, n = migrate_json_obj(json.loads(text))
            except ValueError as e:
                print(f"{p}: not JSON ({e}); treating as text")
                after, n = migrate_text(text)
            else:
                after = (json.dumps(migrated, indent=1) + "\n") if n else text
        else:
            after, n = migrate_text(text)
        if after == text:
            print(f"{p}: unchanged")
            continue
        changed += 1
        print("".join(difflib.unified_diff(text.splitlines(keepends=True), after.splitlines(keepends=True),
                                           fromfile=str(p), tofile=str(p) + " (migrated)")))
        if a.write:
            bak = p.with_name(p.name + f".bak-{stamp}-pre-env-rename")
            bak.write_bytes(raw)
            p.write_text(after, encoding="utf-8")
            print(f"{p}: written (old copy: {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
