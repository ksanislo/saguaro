"""serve/turbo_live.py - the per-request Turbo (PLAN-turbo-live): lossy speed options carried by the request and
applied by the engine to that chat's tokens only; every other chat in the batch stays exact.  Absent or off is
today's exact request: the engine line is byte-for-byte what it was.

Options (request field `turbo`): {"drop": N, "skip": "off"|"ssd"|"ram"}
  drop  run top-k minus N experts per token (drop the N lowest router weights, renormalise the kept ones)
  skip  "ssd": drop an expert that would be read from SSD (VRAM and RAM still serve);
        "ram": drop an expert not resident in VRAM (VRAM only).  Engines without the tier treat it as off.
The wire form is the request line's keys `turbo_drop=N` and `turbo_skip=ssd|ram` (both only when they say
something): the same per-request path the calibration's tune keys take.  serve/web/turbo-core.js holds the page's
per-chat half of the same options."""

from __future__ import annotations

SKIP_LEVELS = ("off", "ssd", "ram")
DROP_MAX = 9   # the MoE routers pick 10 experts; dropping all 10 is off


def normalize_turbo(value):
    """The request's `turbo` field as {"drop": int, "skip": str} or None (off).  A wrong shape is a ValueError
    naming the field (the handler's 400).  drop: an integer 0..DROP_MAX; skip: "off", "ssd" or "ram"."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('turbo: {"drop": N, "skip": "off"|"ssd"|"ram"}')
    drop = value.get("drop", 0)
    if drop is None:
        drop = 0
    if isinstance(drop, str) and drop.strip().lstrip("+-").isdigit():
        drop = int(drop)
    if isinstance(drop, bool) or not isinstance(drop, int) or not 0 <= drop <= DROP_MAX:
        raise ValueError(f"turbo.drop: a whole number 0..{DROP_MAX}")
    skip = value.get("skip", "off")
    if skip is None:
        skip = "off"
    if not isinstance(skip, str) or skip.strip().lower() not in SKIP_LEVELS:
        raise ValueError('turbo.skip: "off", "ssd" or "ram"')
    skip = skip.strip().lower()
    return {"drop": drop, "skip": skip}   # off stays off: it says "not for me" over a model's live Turbo setting


def turbo_on(options):
    """A normalized options dict (or None) as it stays: the dict while any part is on, None when exact."""
    return options if isinstance(options, dict) and (options.get("drop") or options.get("skip", "off") != "off") else None


def turbo_of(sampling):
    """What this request runs Turbo with (already normalized by check_request_sampling), or None."""
    t = (sampling or {}).get("turbo")
    return t if isinstance(t, dict) and (t.get("drop") or t.get("skip", "off") != "off") else None


def turbo_keys(sampling):
    """The request line's Turbo keys: "" when the request is exact (absent, off or off-ish - the line is then
    today's), else ` turbo_drop=N` and/or ` turbo_skip=ssd|ram` for the parts that are on."""
    t = turbo_of(sampling)
    if not t:
        return ""
    keys = ""
    if t.get("drop"):
        keys += f" turbo_drop={int(t['drop'])}"
    if t.get("skip", "off") != "off":
        keys += f" turbo_skip={t['skip']}"
    return keys


def turbo_badge(turbo):
    """The Monitor's badge text for a request's Turbo options ("" when off)."""
    if not isinstance(turbo, dict):
        return ""
    parts = []
    if turbo.get("drop"):
        parts.append(f"-{turbo['drop']} experts")
    if turbo.get("skip", "off") != "off":
        parts.append({"ssd": "no SSD", "ram": "VRAM only"}.get(turbo["skip"], turbo["skip"]))
    return " · ".join(parts)
