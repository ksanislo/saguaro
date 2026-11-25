"""tools/config_names.py - the run configs' name family (1.2.2): a setup writes saguaro-<model>.json beside
saguaro.py; the older maya-<model>.json name is still found.  Everything that lists, names or looks up a config
goes through this one module: serve/models.py's list, serve/model_api.py's find_config and match_name,
saguaro.py's configs() and its --config path.

The rules (C122):
  - finding: both families, the new name first; a model present under both names is the new one (the old file is
    shadowed on the list, and the rename at the next start sets it aside as <old name>.shadowed - it must never
    later be renamed back over a deleted model; it stops being a maya-*.json and nothing loads it);
  - the display name: the file stem without its leading family name;
  - a written config is always named with the new family;
  - a path given as the old name after the rename (llama-swap, systemd, a run script) resolves to the new file:
    resolve() is the fallback, so the old path keeps working without symlinks (os.symlink is not dependable on
    Windows, where the configs and their run scripts live too).

The sidecars that live beside a config (<config>.shared-settings.json, <config>.turbo.json) are listed with it
but are not configs; run-<stem>.sh / .bat, <config>.prev, <stem>.log and <config>.bak-* move with it (they are
the rename's, serve/models.py's and tools/saguaro/migrate_configs.py's own files)."""
from __future__ import annotations

import re
from pathlib import Path

NEW = "saguaro-"                       # 1.2.2 on: the family a setup writes
OLD = "maya-"                          # what setups wrote before: still found, never written
SIDECAR_SUFFIXES = (".shared-settings.json", ".turbo.json")
# a config file's own name, either family (serve/server.py checks a page's name with it)
CONFIG_NAME = re.compile(r"^(?:" + re.escape(NEW) + "|" + re.escape(OLD) + r")[^/\\]+\.json$")
GLOBS = (NEW + "*.json", OLD + "*.json")   # the new family first, everywhere a folder is searched


def is_config_name(name) -> bool:
    """A folder entry's name as a config's own (either family), and not one of the sidecars beside it."""
    n = str(name or "")
    return bool(CONFIG_NAME.match(n)) and not n.endswith(SIDECAR_SUFFIXES)


def bare(stem) -> str:
    """A config stem's display name: its leading family name taken off, of either family."""
    s = str(stem)
    return s[len(NEW):] if s.startswith(NEW) else s[len(OLD):] if s.startswith(OLD) else s


def write_stem(stem) -> str:
    """The stem a config of this name is written under now: always the new family."""
    return NEW + bare(stem)


def matches(stem, want) -> bool:
    """The name a request or a page gives (`want`, already trimmed of ":latest" / ".json") names this config stem?
    The whole stem, its display name, or the same name under the other family (an install that asks the API by the
    old name after the rename - the old name as the fallback)."""
    w, s = str(want or ""), str(stem)
    if not w:
        return False
    return w in (s, bare(s)) or bool(bare(w)) and bare(w) == bare(s)


def list_configs(root) -> list:
    """The config files in a folder: both families, the new name winning over the old one of the same model, the
    sidecars beside them left out.  Unsorted; an unreadable folder is []."""
    try:
        found = [p for pat in GLOBS for p in Path(root).glob(pat) if not p.name.endswith(SIDECAR_SUFFIXES)]
    except OSError:
        return []
    seen_new = {bare(p.name.removesuffix(".json")) for p in found if p.name.startswith(NEW)}
    out = []
    for p in found:                                    # (found: the new family first; an old name only when its
        if not p.name.startswith(OLD) or bare(p.name.removesuffix(".json")) not in seen_new:   #  model has no new)
            out.append(p)
    return out


def locate(root, stem) -> Path:
    """The config file of this stem: the new name when it is there, the old name when only that is, and the new
    path to write to when neither is.  `stem` may be given under either family."""
    root = Path(root)
    b = bare(stem)
    for name in (NEW + b, OLD + b):
        if (root / f"{name}.json").is_file():
            return root / f"{name}.json"
    return root / f"{NEW + b}.json"


def note_paths(root, name) -> list:
    """The server's notes to saguaro.py (serve/models.py's SWITCH_FILE / DELETE_FILE): the paths a note can be
    read from, the new name first - a server from before the 1.2.2 rename wrote .maya-switch.json /
    .maya-delete.json and can leave one behind mid-restart.  Only the new name is ever written."""
    n = str(name)
    names = [n, ".maya-" + n[len(".saguaro-"):] if n.startswith(".saguaro-") else n]
    return [Path(root) / f for f in dict.fromkeys(names) if (Path(root) / f).is_file()]


def resolve(path) -> Path:
    """A config path that no longer exists under its old family name, where the file did move to the new one:
    the new path (--config maya-<model>.json after the rename: the path llama-swap or systemd was given keeps
    working).  Anything else comes back unchanged."""
    p = Path(path)
    if p.is_file() or not p.name.startswith(OLD):
        return p
    twin = p.with_name(NEW + p.name[len(OLD):])
    return twin if twin.is_file() else p
