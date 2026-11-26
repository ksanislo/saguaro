"""serve/savings.py - what the tokens Saguaro served would have cost on the model's own API: the web app's "Saved" line.

Every finished request adds its tokens to the running model's count: the prompt it read new, the prompt it reused from
its own cache (an API bills that as cached input, far cheaper) and what it wrote.  The count is kept per model (per run
config) in token-savings.json beside the configs, so it adds up across starts; "this start" is the server's own.

The prices are the model's own API's (the developer's, not a reseller's), per 1M tokens: the built-in table below and
data/api-prices.json are prefilled from OpenRouter's public model list at each release
(tools/refresh_prices.py), so nothing needs a hand-maintained price list.
DeepSeek charges twice its off-peak price at its peak hours (Monday to Friday, 01:00-04:00 and 06:00-10:00 UTC; its
Chinese public holidays are off-peak too, which this does not know - those days count as peak), so a request finished
then is counted at that price.  A model without a known API price is counted in tokens only.  Electricity is not
counted.

The page's Settings > Savings can change a model's prices (or give one to a model without them): they are kept in the
same file ("prices", by model name) and every total is counted again with them - the file keeps tokens, not dollars.

The default prices come from data/api-prices.json in this folder: {"as_of": "YYYY-MM", "prices": {model name: an entry}}
- the same entries the built-in table below holds.  A file that cannot be read or does not check out (check_price) is
ignored with one log line, and the built-ins are used; a model the file does not name keeps its built-in entry.

With live_prices on (Settings > Savings, the run config's "live_prices", or SAGUARO_LIVE_PRICES=1 - off by default,
nothing leaves the box unless the user turns it on), the server fetches OpenRouter's public model list at most once a
day, and at a start whose cache is older (openrouter.ai/api/v1/models, no key, no user data; a background thread, a
10 s timeout; the result cached with its fetch time beside the count file).  On any failure the cache or the shipped
table counts, with one log line.  The precedence: the page's own prices > the fetched OpenRouter prices >
data/api-prices.json > the built-ins; the Savings line says which of them counts."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

FILE = "token-savings.json"
ROOT = Path(__file__).resolve().parents[1]


def ledger_path(config: Path, env=None) -> Path | None:
    """Where the count is kept: beside the run config, or SAGUARO_SAVINGS_PATH (a file; its folder must be writable, the
    temporary file is written next to it).  "off" (or "none", "0") keeps no file: the count lives only until the stop."""
    v = ((os.environ if env is None else env).get("SAGUARO_SAVINGS_PATH") or "").strip()
    if v.lower() in ("off", "none", "0"):
        return None
    return Path(v).expanduser() if v else Path(config).parent / FILE
PRICES_AS_OF = "2026-11"
# per 1M tokens, in US dollars; "peak": the factor at DeepSeek's peak hours; "openrouter": the family's id in
# OpenRouter's public model list (serve/savings.py's live fetch, tools/refresh_prices.py).  The figures are the
# OpenRouter list's of November 2026; MiniMax's and Hy4-flash's are flat (no context tiers, no peak hours).
REFERENCES = [
    {"match": r"deepseek[-_.]?v?4\.1[-_.]?flash|deepseek[-_.]flash", "api": "DeepSeek",
     "openrouter": "deepseek/deepseek-v4.1-flash-1017",
     "input": 0.15, "cached": 0.003, "output": 0.60, "peak": 2.0},
    {"match": r"glm[-_.]?5\.3[-_.]?flash", "api": "Z.ai", "openrouter": "z-ai/glm-5.3-flash",
     "input": 0.15, "cached": 0.03, "output": 0.50},
    {"match": r"minimax[-_.]?m3\.1[-_.]?flash", "api": "MiniMax", "openrouter": "minimax/minimax-m3.1-flash",
     "input": 0.075, "cached": 0.0075, "output": 0.30},
    {"match": r"(?:^|[/_.-])hy[-_.]?4[-_.]?flash", "api": "Tencent", "openrouter": "tencent/hy4-flash",
     "input": 0.035, "cached": 0.0015, "output": 0.1405},
    # Qwen3.8-Flash-Next, every quant of it: the API's price is the family's
    {"match": r"qwen[-_.]?3\.?8[-_.]?flash", "api": "Qwen", "openrouter": "qwen/qwen3.8-flash",
     "input": 0.15, "cached": 0.016, "output": 0.47},
    {"match": r"inkling[-_.]?2[-_.]?mini", "api": "Thinking Machines", "openrouter": "thinkingmachines/inkling-2-mini",
     "input": 0.075, "cached": 0.015, "output": 0.30},
]
BANDS = ("base", "peak")


def reference(model: str | None) -> dict | None:
    for r in REFERENCES:
        if re.search(r["match"], str(model or ""), re.I):
            return {k: v for k, v in r.items() if k != "match"}
    return None


PRICE_KEYS = ("input", "cached", "output")


def check_price(model: str, p: dict) -> dict:
    """A price the page set for `model`: three numbers of dollars per 1M tokens (and DeepSeek's peak factor), checked."""
    if not isinstance(p, dict):
        raise ValueError("a price is input, cached and output, in dollars per 1M tokens")
    out = {}
    for k in PRICE_KEYS:
        v = p.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1000:
            raise ValueError(f"{k}: a number of dollars per 1M tokens, from 0 to 1000")
        out[k] = float(v)
    base = reference(model)
    if base and base.get("peak"):                      # (DeepSeek's peak hours: only where they are its own)
        f = p.get("peak", base["peak"])
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not 1 <= f <= 10:
            raise ValueError("peak hours: a factor from 1 to 10")
        out["peak"] = float(f)
    if p.get("api") is not None:
        out["api"] = str(p["api"]).strip()[:40]
    if p.get("openrouter") is not None:                   # the family's id in OpenRouter's public model list
        out["openrouter"] = str(p["openrouter"]).strip()[:100]
    return out


PRICE_FILE = "api-prices.json"                                  # the shipped fallback, beside this folder's code
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"          # the public model list: no key, nothing about the user
LIVE_FILE = "openrouter-prices.json"                            # the cached fetch, beside the ledger's file
AS_OF = re.compile(r"^\d{4}-\d{2}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def file_prices(prices: dict) -> dict:
    """A price file's "prices": every entry checked with check_price (plus its "match" regex, the model names it fits);
    raises ValueError on the first entry that is not a price."""
    out = {}
    for name, p in prices.items():
        v = check_price(str(name), p)
        pat = str(p.get("match")) if p.get("match") is not None else re.escape(str(name))
        try:
            re.compile(pat)
        except re.error:
            raise ValueError(f"{name}: its match is not a regex")
        v["match"] = pat[:200]
        out[str(name)] = v
    return out


def parse_price_file(text: str) -> tuple:
    """The file's text as {"as_of", "prices"} - or why it cannot be used: (None, why)."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        return None, f"not JSON ({e})"
    if not (isinstance(doc, dict) and isinstance(doc.get("as_of"), str) and AS_OF.match(doc["as_of"])
            and isinstance(doc.get("prices"), dict)):
        return None, "not an as_of month and a prices object"
    try:
        return {"as_of": doc["as_of"], "prices": file_prices(doc["prices"])}, None
    except (ValueError, TypeError) as e:
        return None, str(e)


def load_price_file(path: Path) -> dict | None:
    """data/api-prices.json (or a cached copy of it): the prices it holds, or None - a file that is missing is quiet,
    a file that is broken is ignored with one log line (the prices below it stay)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        print(f"[saguaro] price file {path.name} ignored ({e}): the prices below it stay", flush=True)
        return None
    doc, why = parse_price_file(text)
    if doc is None:
        print(f"[saguaro] price file {path.name} ignored ({why}): the prices below it stay", flush=True)
    return doc


def per_million(s) -> float:
    """One of OpenRouter's per-token USD strings ("0.000000075") as dollars per 1M tokens: exact, on the decimal
    string - never a float multiplied by a million.  ValueError: not a price."""
    try:
        return float(Decimal(str(s).strip()) * 1_000_000)
    except (InvalidOperation, AttributeError) as e:
        raise ValueError(f"not a per-token price ({str(s)[:20]!r})") from e


def price_families(disk: dict | None) -> dict:
    """The families a live fetch looks up, keyed by their OpenRouter id: the built-ins, overridden by the shipped
    file's entries (a release may add a family or retitle one there without touching the code)."""
    fam = {r["openrouter"]: {"name": r["api"], "api": r["api"], "match": r["match"],
                             **({"peak": r["peak"]} if r.get("peak") else {})}
           for r in REFERENCES if r.get("openrouter")}
    if disk:
        for name, p in disk["prices"].items():
            if p.get("openrouter"):
                fam[p["openrouter"]] = {"name": name, "api": p.get("api") or name, "match": p["match"],
                                        **({"peak": p["peak"]} if p.get("peak") else {})}
    return fam


def parse_openrouter(text: str, families: dict, now: float | None = None) -> tuple:
    """OpenRouter's model list as a price doc: every family whose id the list holds, its per-token strings (prompt,
    completion, input_cache_read) as dollars per 1M tokens - or (None, why this list cannot be used).  A family the
    list does not price simply keeps the price below it; only an answer that holds none of ours is refused."""
    try:
        listed = json.loads(text).get("data")
    except (ValueError, TypeError, AttributeError) as e:
        return None, f"not the model list ({e})"
    if not isinstance(listed, list):
        return None, "not the model list (no data array)"
    by_id = {}
    for m in listed:
        if isinstance(m, dict) and isinstance(m.get("id"), str):
            by_id.setdefault(m["id"].strip().lower(), m)
    found = {}
    for oid, f in families.items():
        entry = by_id.get(oid.lower())
        pr = entry.get("pricing") if isinstance(entry, dict) else None
        if not isinstance(pr, dict):
            continue                                        # (not listed: the prices below this family count)
        try:
            p = {"input": per_million(pr["prompt"]), "output": per_million(pr["completion"]),
                 "cached": per_million(pr["input_cache_read"])}
        except (KeyError, ValueError):
            continue                                        # (a free "-" or a malformed entry: same)
        if f.get("peak"):
            p["peak"] = f["peak"]                           # (peak hours are ours to know; the list shows the base)
        found[oid] = p
    if not found:
        return None, "none of our models in the list"
    prices = {}
    for oid, p in found.items():
        f = families[oid]
        prices[f["name"]] = file_prices({f["name"]: {**p, "api": f["api"], "match": f["match"], "openrouter": oid}}
                                        )[f["name"]]
    when = time.time() if now is None else now
    return {"as_of": time.strftime("%Y-%m-%d", time.gmtime(when)), "prices": prices}, None


def load_live_file(path: Path) -> dict | None:
    """The cached OpenRouter fetch (its prices and when they were fetched): a file that is missing is quiet, one that
    does not check out is ignored with one log line (the shipped table below it stays)."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        print(f"[saguaro] price cache {path.name} ignored ({e}): the shipped prices stay", flush=True)
        return None
    if not (isinstance(doc, dict) and isinstance(doc.get("prices"), dict)):
        print(f"[saguaro] price cache {path.name} ignored (not fetched prices): the shipped prices stay", flush=True)
        return None
    try:
        prices = file_prices(doc["prices"])
    except (ValueError, TypeError) as e:
        print(f"[saguaro] price cache {path.name} ignored ({e}): the shipped prices stay", flush=True)
        return None
    f = doc.get("fetched")
    fetched = float(f) if isinstance(f, (int, float)) and not isinstance(f, bool) else None
    as_of = str(doc.get("as_of") or "")
    if not DATE.match(as_of):
        as_of = time.strftime("%Y-%m-%d", time.gmtime(fetched if fetched is not None else time.time()))
    return {"as_of": as_of, "fetched": fetched, "prices": prices}


def live_enabled(cfg_value=None, env=None) -> bool:
    """Whether the OpenRouter prices are wanted: SAGUARO_LIVE_PRICES (1/0) beats the run config's "live_prices";
    both off by default - nothing leaves the box unless the user says so (docs/SETTINGS.md)."""
    v = ((os.environ if env is None else env).get("SAGUARO_LIVE_PRICES") or "").strip().lower()
    if v in ("1", "on", "true", "yes"):
        return True
    if v in ("0", "off", "false", "no"):
        return False
    return bool(cfg_value)


def release_name() -> str:
    """What the Savings line calls this Saguaro (its built-in prices were prefilled from OpenRouter at it)."""
    try:
        from saguaro_version import installed_version
    except ImportError:
        return ""
    return installed_version(ROOT) or ""


class PriceBook:
    """The default prices one Ledger counts with: the page's own (Settings > Savings) beat these, and within these:
    the fetched OpenRouter prices (live_prices, off by default) > data/api-prices.json > the built-ins.
    The fetch runs on a background thread at most once a day, and at a start whose cache is older; it never makes a
    request or the start wait.  On any failure the cache or the shipped table counts, with one log line."""

    FETCH_EVERY = 24 * 3600                   # at most one fetch a day ...
    RETRY_AFTER = 3600                        # ... and after one that did not work, ask again after this
    TIMEOUT = 10                              # seconds: the list is never waited on for longer

    def __init__(self, folder: Path | None = None, repo_file: Path = ROOT / "data" / PRICE_FILE, live: bool = False):
        self.folder = Path(folder) if folder is not None else None
        self.repo_file = Path(repo_file)
        self.live_file = self.folder / LIVE_FILE if self.folder is not None else None    # the cache, by the ledger
        self.disk = load_price_file(self.repo_file)
        self.live = bool(live)
        self.live_doc = load_live_file(self.live_file) if self.live_file is not None else None
        self.lock = threading.Lock()          # one fetch at a time; the network happens outside it
        self.next_try = 0.0
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        if self.live_file is not None:
            try:
                self.next_try = self.live_file.stat().st_mtime + self.FETCH_EVERY
            except OSError:
                pass
        self.start()                          # (at start, when the cache is older; never blocks)

    def stale(self) -> bool:
        return time.time() >= self.next_try

    def families(self) -> dict:
        return price_families(self.disk)

    def start(self):
        """The prices are live and the cache is a day old: fetch on a background thread (the start, and every time
        the Savings page asks, call this; nothing here waits for the network)."""
        if not self.live or self.live_file is None or not self.stale():
            return
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return                        # (one fetch at a time)
            self.next_try = time.time() + self.RETRY_AFTER      # (a success moves it a day ahead)
            self.thread = t = threading.Thread(target=self._fetch, daemon=True)
        t.start()

    def refresh(self):
        """The Savings page asked: nothing to wait for - a fetch on its thread when the cache says it is time."""
        self.start()

    def set_live(self, on: bool):
        """Settings > Savings: on starts the daily fetch right away; off means the shipped table only - no network,
        and the cached fetch is ignored while off."""
        self.live = bool(on)
        self.start()

    def _fetch(self):
        why = None
        doc = None
        try:
            req = urllib.request.Request(OPENROUTER_URL, headers={"User-Agent": "saguaro/savings"})
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                text = r.read().decode("utf-8", "replace")
        except (OSError, ValueError) as e:                    # offline, a proxy, OpenRouter's limit: the shipped ones count
            why = str(getattr(e, "reason", None) or e)
        else:
            doc, why = parse_openrouter(text, self.families(), time.time())
        if doc is None:                                       # one line for the log, then the prices below stay
            self.error = why
            print(f"[saguaro] the OpenRouter prices did not come ({why}): the shipped prices stay", flush=True)
            return
        try:                                                  # the cache is next to the ledger's file; unwritable: this run still counts
            payload = {"fetched": time.time(), "as_of": doc["as_of"], "prices": doc["prices"]}
            tmp = self.live_file.with_name(self.live_file.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            os.replace(tmp, self.live_file)
        except OSError:
            pass
        self.error = None
        with self.lock:
            self.next_try = time.time() + self.FETCH_EVERY
            self.live_doc = doc

    def source(self) -> dict:
        """Where the default prices come from right now (the Savings line says it): "OpenRouter, <date>" or
        "built-in prices, <release>"."""
        if self.live and self.live_doc is not None:
            return {"from": "openrouter", "date": self.live_doc["as_of"], "fetched": self.live_doc.get("fetched")}
        return {"from": "built-in", "as_of": PRICES_AS_OF, "release": release_name()}

    def as_of(self) -> str:
        if self.live and self.live_doc is not None:
            return self.live_doc["as_of"]
        if self.disk is not None:
            return self.disk["as_of"]
        return PRICES_AS_OF

    def reference(self, model: str | None) -> dict | None:
        """The model's default prices: the fetched list's entry whose "match" fits its name, then the file's,
        then the built-in one."""
        m = str(model or "")
        docs = ([self.live_doc] if (self.live and self.live_doc) else []) + ([self.disk] if self.disk else [])
        for doc in docs:
            for p in doc["prices"].values():
                if re.search(p["match"], m, re.I):
                    return {k: v for k, v in p.items() if k != "match"}
        return reference(model)


def deepseek_peak(t: float) -> bool:
    """DeepSeek's peak hours: Monday to Friday, 01:00-04:00 and 06:00-10:00 UTC."""
    g = time.gmtime(t)
    return g.tm_wday < 5 and (1 <= g.tm_hour < 4 or 6 <= g.tm_hour < 10)


def zero() -> dict:
    return {b: {"input": 0, "cached": 0, "output": 0, "requests": 0} for b in BANDS}


def cost(bands: dict, ref: dict | None) -> float | None:
    """The API's charge for these tokens, in dollars (None: no known price)."""
    if ref is None:
        return None
    usd = 0.0
    for b in BANDS:
        c = bands.get(b) or {}
        f = ref.get("peak", 1.0) if b == "peak" else 1.0
        usd += f * (c.get("input", 0) * ref["input"] + c.get("cached", 0) * ref["cached"]
                    + c.get("output", 0) * ref["output"]) / 1e6
    return usd


def tokens(bands: dict) -> dict:
    out = {"input": 0, "cached": 0, "output": 0, "requests": 0}
    for b in BANDS:
        for k in out:
            out[k] += int((bands.get(b) or {}).get(k, 0))
    return out


class Ledger:
    """The running model's count (`key`: its config's file name) and the others' from the file; one lock for both."""

    def __init__(self, path: Path | None, key: str, model: str, quant: str | None = None,
                 names: tuple | None = None, prices: PriceBook | None = None,
                 config: Path | None = None, live: bool = False):
        self.path = Path(path) if path else None
        self.key, self.model, self.quant = key, model, quant
        self.names = names                    # (/health's two parts (pretty, quant): what the page names the model with)
        self.config = Path(config) if config is not None else None      # (the run config: where "live_prices" is saved)
        self.book = prices if prices is not None else PriceBook(self.path.parent if self.path else None, live=live)
        self.lock = threading.Lock()
        self.save_failed = False                      # (one log line for a file that cannot be written, not one per request)
        self.started = time.time()
        self.start = zero()
        self.doc = {"since": time.time(), "models": {}}
        if self.path is not None:
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(d, dict) and isinstance(d.get("models"), dict):
                    self.doc = d
            except (OSError, ValueError):
                pass
        me = self.doc["models"].setdefault(key, {"since": time.time(), "bands": zero()})
        me.update(model=model, quant=quant)
        if names and names[0]:                         # (the two parts the page names this model with, as /health says
            me["name_pretty"], me["name_quant"] = str(names[0]), str(names[1] or "")
        else:                                          # (the config says nothing now: no old run's name beside it)
            me.pop("name_pretty", None)
            me.pop("name_quant", None)
        for b in BANDS:
            me["bands"].setdefault(b, zero()[b])
        if not isinstance(self.doc.get("prices"), dict):
            self.doc["prices"] = {}

    def price(self, model: str | None) -> dict | None:
        """The model's prices: the page's (Settings > Savings), else the file's / built-in ones; None: none known."""
        base = self.book.reference(model)
        own = self.doc.get("prices", {}).get(model or "")
        if not own:
            return base
        return {"api": (base or {}).get("api") or "your prices", **{k: v for k, v in (base or {}).items()
                                                                    if k == "peak"}, **own, "custom": True}

    @property
    def ref(self) -> dict | None:
        return self.price(self.model)

    def set_price(self, model: str, p: dict | None):
        """Settings > Savings: the model's own prices (None: back to the file's / built-in ones); every total follows."""
        model = str(model or "").strip()
        if not model or len(model) > 200:
            raise ValueError("which model's prices?")
        with self.lock:
            if p is None:
                self.doc["prices"].pop(model, None)
            else:
                self.doc["prices"][model] = check_price(model, p)
            self._save()

    def set_live(self, on: bool):
        """Settings > Savings: update the prices from OpenRouter once a day, or not (the shipped table only).
        Saved in the run config ("live_prices") so the next start keeps it; on starts the fetch right away."""
        on = bool(on)
        if self.config is not None:
            try:
                p = Path(self.config)
                cfg = json.loads(p.read_text(encoding="utf-8-sig"))
                cfg["live_prices"] = on
                tmp = p.with_name(p.name + ".tmp")
                tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
                os.replace(tmp, p)
            except (OSError, ValueError) as e:
                raise ValueError(f"the setting could not be saved in the run config ({e})") from None
        self.book.set_live(on)

    def price_list(self) -> list:
        """Every model counted here (and the running one) with its prices and the built-in ones (Settings)."""
        seen = []
        for m in [self.model] + [x.get("model") for x in self.doc["models"].values()] + list(self.doc["prices"]):
            if m and m not in seen:
                seen.append(m)
        out = []
        for m in seen:
            p, base = self.price(m), self.book.reference(m)
            out.append({"model": m, "api": (p or base or {}).get("api"), "custom": bool(p and p.get("custom")),
                        "running": m == self.model, "default": base,
                        **({k: p.get(k) for k in (*PRICE_KEYS, "peak")} if p else {k: None for k in (*PRICE_KEYS, "peak")}),
                        **self.name_fields(m)})
        return out

    def name_fields(self, model: str | None) -> dict:
        """The page's two name fields for a model counted here ({} when no run recorded them: the page keeps today's
        text).  Counted under the caller's lock; keyed on nothing new - the entry's model id, as the prices are.
        The running config's entry first: two configs of one model, and this run's name is the page's."""
        entries = self.doc["models"]
        for m in [entries.get(self.key)] + [x for k, x in entries.items() if k != self.key]:
            if m and m.get("model") == model and m.get("name_pretty"):
                return {"model_name_pretty": m["name_pretty"], "model_quant": m.get("name_quant") or ""}
        return {}

    def add(self, prompt_tokens: int, reused: int, output: int, now: float | None = None):
        """One finished request: `prompt_tokens` in all, `reused` of them from the conversation cache, `output` written."""
        now = time.time() if now is None else now
        reused = max(0, min(int(reused or 0), int(prompt_tokens or 0)))
        fresh = max(0, int(prompt_tokens or 0) - reused)
        band = "peak" if self.ref and self.ref.get("peak") and deepseek_peak(now) else "base"
        with self.lock:
            for c in (self.doc["models"][self.key]["bands"][band], self.start[band]):
                c["input"] += fresh
                c["cached"] += reused
                c["output"] += int(output or 0)
                c["requests"] += 1
            self._save()

    def _save(self):
        if self.path is None:
            return
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.doc, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
            self.save_failed = False
        except OSError as e:
            if not self.save_failed:
                print(f"[saguaro] could not save the token count ({self.path.name}): {e}; the count is kept in memory "
                      "(SAGUARO_SAVINGS_PATH names another file, 'off' keeps none)", flush=True)
            self.save_failed = True

    def summary(self) -> dict:
        """The sidebar's line (GET /metrics, every second): the total saved and this start's."""
        with self.lock:
            total = sum(cost(m.get("bands") or {}, self.price(m.get("model"))) or 0.0
                        for m in self.doc["models"].values())
            priced = any(self.price(m.get("model")) for m in self.doc["models"].values())
            return {"usd": round(total, 6), "start_usd": round(cost(self.start, self.ref) or 0.0, 6), "priced": priced}

    def report(self) -> dict:
        """The breakdown (GET /api/savings): every model's tokens and dollars, this start's, the prices and whether
        the live OpenRouter prices are on.  The page asking is also what starts the daily fetch, when it is due and
        the setting is on - on its thread, never blocking this answer (serve/savings.py's PriceBook)."""
        self.book.refresh()
        with self.lock:
            models, total, refs = [], 0.0, {}
            for key, m in self.doc["models"].items():
                ref = self.price(m.get("model"))
                usd = cost(m.get("bands") or {}, ref)
                total += usd or 0.0
                if ref is not None:
                    refs[m.get("model")] = ref
                models.append({"key": key, "model": m.get("model"), "quant": m.get("quant"), "since": m.get("since"),
                               "running": key == self.key, "tokens": tokens(m.get("bands") or {}),
                               "usd": None if usd is None else round(usd, 6),
                               **({"model_name_pretty": m["name_pretty"], "model_quant": m.get("name_quant") or ""}
                                  if m.get("name_pretty") else {})})
            models.sort(key=lambda x: (not x["running"], -(x["usd"] or 0)))
            me = self.ref
            start = cost(self.start, me)
            mine = self.doc["models"].get(self.key) or {}
            return {"usd": round(total, 6), "since": self.doc.get("since"), "prices_as_of": self.book.as_of(),
                    "prices_source": self.book.source(), "live": bool(self.book.live),
                    "start": {"model": self.model, "quant": self.quant, "started": self.started,
                              "tokens": tokens(self.start), "usd": None if start is None else round(start, 6),
                              **({"model_name_pretty": mine["name_pretty"], "model_quant": mine.get("name_quant") or ""}
                                 if mine.get("name_pretty") else {})},
                    "models": models,
                    "prices": [{**self.name_fields(k), "model": k, **v} for k, v in refs.items()] or
                              ([{"model": self.model, **me}] if me else []),
                    "price_list": self.price_list()}

    def reset(self):
        """Start over from $0: every model's count (this start's too)."""
        with self.lock:
            self.doc = {"since": time.time(), "prices": self.doc.get("prices", {}),   # (the page's prices stay)
                        "models": {self.key: {"since": time.time(), "bands": zero(),
                                              "model": self.model, "quant": self.quant,
                                              **({"name_pretty": self.names[0], "name_quant": self.names[1] or ""}
                                                 if self.names and self.names[0] else {})}}}
            self.start = zero()
            self.started = time.time()
            self._save()
