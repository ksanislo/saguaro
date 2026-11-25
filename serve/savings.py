"""serve/savings.py - what the tokens Saguaro served would have cost on the model's own API: the web app's "Saved" line.

Every finished request adds its tokens to the running model's count: the prompt it read new, the prompt it reused from
its own cache (an API bills that as cached input, far cheaper) and what it wrote.  The count is kept per model (per run
config) in token-savings.json beside the configs, so it adds up across starts; "this start" is the server's own.

The prices are the model's own API's (the developer's, not a reseller's), per 1M tokens, as published in October 2026
(Hy4-flash's: at its launch, November 2026).
DeepSeek charges twice its off-peak price at its peak hours (Monday to Friday, 01:00-04:00 and 06:00-10:00 UTC; its
Chinese public holidays are off-peak too, which this does not know - those days count as peak), so a request finished
then is counted at that price.  A model without a known API price is counted in tokens only.  Electricity is not
counted.

The page's Settings > Savings can change a model's prices (or give one to a model without them): they are kept in the
same file ("prices", by model name) and every total is counted again with them - the file keeps tokens, not dollars.

The default prices come from data/api-prices.json in this folder: {"as_of": "YYYY-MM", "prices": {model name: an entry}}
- the same entries the built-in table below holds.  A file that cannot be read or does not check out (check_price) is
ignored with one log line, and the built-ins are used; a model the file does not name keeps its built-in entry.

When the Savings page asks, the server fetches the newest copy of that file from GitHub at most once a day (its origin
repo, raw.githubusercontent.com; a short timeout; the copy is cached in the savings folder and used while it is newer
than the folder's own), unless SAGUARO_UPDATE_CHECK=0 said no outgoing requests.  The precedence: the page's own
prices > the fetched file > data/api-prices.json > the built-ins."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
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
PRICES_AS_OF = "2026-10"
# per 1M tokens, in US dollars; "peak": the factor at DeepSeek's peak hours
REFERENCES = [
    {"match": r"deepseek[-_.]?v?4\.1[-_.]?flash|deepseek[-_.]flash", "api": "DeepSeek",
     "input": 0.15, "cached": 0.003, "output": 0.60, "peak": 2.0},
    {"match": r"glm[-_.]?5\.3[-_.]?flash", "api": "Z.ai", "input": 0.15, "cached": 0.03, "output": 0.50},
    # MiniMax's own, flat (no context tiers or peak hours for the Flash model):
    # https://platform.minimax.io/docs/pricing/overview (read 2026-10-27)
    {"match": r"minimax[-_.]?m3\.1[-_.]?flash", "api": "MiniMax", "input": 0.15, "cached": 0.015, "output": 0.60},
    # Hy4-flash: Tencent's own (Tencent Cloud TokenHub, USD), https://www.tencentcloud.com/document/product/1300/78937
    # (read 2026-11-22; its launch discount - 50% off until 2026-11-30 - not counted).  Flat: no peak hours
    {"match": r"(?:^|[/_.-])hy[-_.]?4[-_.]?flash", "api": "Tencent", "input": 0.07, "cached": 0.003, "output": 0.281},
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
    return out


PRICE_FILE = "api-prices.json"                                  # data/api-prices.json, and the fetched copy's name
AS_OF = re.compile(r"^\d{4}-\d{2}$")


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


class PriceBook:
    """The default prices one Ledger counts with: the page's own (Settings > Savings) beat these, and within these:
    a fetched api-prices.json (valid and newer as_of than this folder's own) > data/api-prices.json > the built-ins.
    The fetch happens when the Savings page asks, at most once a day; SAGUARO_UPDATE_CHECK=0 turns it off."""

    FETCH_EVERY = 24 * 3600                   # at most one fetch a day ...
    RETRY_AFTER = 3600                        # ... and after one that did not work, ask again after this
    TIMEOUT = 8                               # seconds: a page load never waits on GitHub for longer

    def __init__(self, folder: Path | None = None, repo_file: Path = ROOT / "data" / PRICE_FILE):
        self.folder = Path(folder) if folder is not None else None
        self.repo_file = Path(repo_file)
        self.cache = self.folder / PRICE_FILE if self.folder is not None else None     # the fetched copy, in the ledger's folder
        self.disk = load_price_file(self.repo_file)
        self.fetched: dict | None = None
        self.lock = threading.Lock()          # one fetch at a time; the network happens outside it
        self.next_try = 0.0
        self.error: str | None = None
        if self.cache is not None:
            cached = load_price_file(self.cache)                        # last run's copy, if it is there and still newer
            if cached and self.beats_disk(cached):
                self.fetched = cached
            try:
                self.next_try = self.cache.stat().st_mtime + self.FETCH_EVERY
            except OSError:
                pass

    def beats_disk(self, doc: dict) -> bool:
        return self.disk is None or doc["as_of"] > self.disk["as_of"]

    def url(self) -> str:
        """The newest copy of the file, from the GitHub repository this folder was cloned from (serve/update.py)."""
        from serve import update                              # (late import: update.py's repo, no import loop)
        repo = update.origin_repo(ROOT) if (ROOT / ".git").exists() else update.REPO
        return f"https://raw.githubusercontent.com/{repo}/main/data/{PRICE_FILE}"

    def refresh(self):
        """The Savings page asked: fetch the newest copy if a day passed since the last try (nothing else does)."""
        if self.cache is None or os.environ.get("SAGUARO_UPDATE_CHECK", "1") == "0":
            return
        now = time.time()
        with self.lock:                                       # the claim happens inside the lock, the fetch outside it
            if now < self.next_try:
                return
            self.next_try = now + self.RETRY_AFTER            # (a success moves it to a day ahead)
        try:
            req = urllib.request.Request(self.url(), headers={"User-Agent": "saguaro/savings"})
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                text = r.read().decode("utf-8", "replace")
        except (OSError, ValueError) as e:                    # offline, a proxy, GitHub's limit: the prices below count
            self.error = str(getattr(e, "reason", None) or e)
            return
        doc, why = parse_price_file(text)
        if doc is None:                                       # GitHub answered, but not with prices: one log line
            self.error = why
            print(f"[saguaro] the prices fetched from GitHub were ignored ({why}): the prices below them stay", flush=True)
            return
        try:                                                  # the cache is next to the ledger's file; unwritable: this run still counts
            tmp = self.cache.with_name(self.cache.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.cache)
        except OSError:
            pass
        with self.lock:
            self.error = None
            self.next_try = time.time() + self.FETCH_EVERY
            self.fetched = doc if self.beats_disk(doc) else None      # (not newer than the folder's own: that one stays)

    def source(self) -> dict:
        """Where the default prices come from right now, and since which month (the Savings page says it in one line)."""
        if self.fetched is not None:
            return {"from": "github", "as_of": self.fetched["as_of"]}
        if self.disk is not None:
            return {"from": "file", "as_of": self.disk["as_of"]}
        return {"from": "built-in", "as_of": PRICES_AS_OF}

    def as_of(self) -> str:
        active = self.fetched or self.disk
        return active["as_of"] if active else PRICES_AS_OF

    def reference(self, model: str | None) -> dict | None:
        """The model's default prices: the file's entry whose "match" fits its name, else the built-in one."""
        m = str(model or "")
        for doc in ([self.fetched] if self.fetched else []) + ([self.disk] if self.disk else []):
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
                 names: tuple | None = None, prices: PriceBook | None = None):
        self.path = Path(path) if path else None
        self.key, self.model, self.quant = key, model, quant
        self.names = names                    # (/health's two parts (pretty, quant): what the page names the model with)
        self.book = prices if prices is not None else PriceBook(self.path.parent if self.path else None)
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
        """The breakdown (GET /api/savings): every model's tokens and dollars, this start's, and the prices.
        The page asking is also what fetches a newer price file - at most once a day (serve/savings.py's PriceBook)."""
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
                    "prices_source": self.book.source(),
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
