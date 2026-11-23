#!/usr/bin/env python3
# tools/qfn/kl.py (from bench/strata-sm75/qkl.py) - QFN's next-token distributions against a reference run, teacher-forced:
# a switch against the base arm on one box, or one backend against another (AMD HIP vs the T4 reference; the same
# kl-set.json ids, so both runs score the same positions).  Quality of rounding-changing switches: KL against the base arm.
# Each prompt = a long head (8k-28k tokens) read by the BATCHED prompt path (where MMQ_KQ / BF16_TC act), a turn boundary,
# then a ~1k-token tail read through the verify windows (--short-read) with SAGUARO_LOGPOS / SAGUARO_LOGPOS_TOPK=64: the
# log-probabilities of every tail position, which see the K/V the prompt path wrote (upstream's "after a 16K prompt"
# method, docs/UNSLOTH_Q4.md).  SAGUARO_DUMP_FIRST_LOGITS adds the full-vocabulary logits after each whole prompt.
#   kl.py build --out kl-set.json                        tokenize the prompt set (CPU; run once, the ids are frozen)
#   kl.py run --config CFG.json --set kl-set.json --out DIR   one engine, every prompt once (GPUs: the lead only)
#   kl.py compare REF_DIR RUN_DIR [...] [--json F]    the table + verdict per run against REF (the base arm)
import argparse, json, math, os, pathlib, queue, re, subprocess, sys, threading, time

IM_START, IM_END = 248045, 248046
TOPK = 64
TAIL = 1024
TOK_DIR = os.environ.get("QFN_TOKENIZER", "/opt/llm/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/pack/tokenizer")
SRC = str(pathlib.Path(__file__).resolve().parents[2] / "engines" / "qfn")
# determinism: what the windows compute must not depend on timing (PCIe share probes, adaptive expert moves, CPU share)
DROP_VALUE = {"--expert-profile-save", "--conversation-cache-mib", "--conversation-cache-slots", "--short-read",
              "--adapt-every", "--pcie-frac", "--suffix-draft", "--prompt-cache"}
FIXED = ["--adapt-every", "100000", "--pcie-frac", "0", "--suffix-draft", "0", "--prompt-cache", "6"]
FIXED_ENV = {"SAGUARO_ROUTE_TAIL_SKIP": "0", "SAGUARO_PREFILL_CPU_SHARE": "0"}
# pass rule (vs base): upstream accepted summation-order switches at mean KL ~0.01 (BF16_TC #540: 8.4e-3 on V100;
# SELECT_SIMT #2071: 0.01); a rounding change may only flip near-ties, never a confident top token
RULE = {"kl": 0.01, "top1": 0.98, "top1_conf": 0.995, "conf_gap": 0.5, "ppl": 0.005, "first_kl": 0.01}


# ------------------------------------------------------------------------------------------------ build
def tokenizer():
    sys.path.insert(0, SRC + "/tools")
    import strata_tokenizer as ST
    tp = pathlib.Path(TOK_DIR)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tp / "token_type.json").read_text()))


def lines(path, a=1, b=None):
    L = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    return "".join(L[a - 1:b])


def doc_prompt(tok, name, kind, text, head):
    """user: <ask> + the first `head` tokens of text; assistant (the scored tail): the next TAIL tokens of it."""
    ids = tok.encode(text)
    if len(ids) < head + TAIL:
        head = len(ids) - TAIL
    ask = tok.encode(f"user\n[{name}] Continue the following {kind} exactly from where it stops.\n\n")
    hdr = tok.encode("assistant\n<think>\n\n</think>\n\n", parse_special=True)
    return [IM_START] + ask + ids[:head] + [IM_END] + tok.encode("\n") + [IM_START] + hdr + ids[head:head + TAIL]


TOOLS = [("read_file", "Read a file of the repository, optionally a line range.",
          {"path": "string", "start_line": "integer", "end_line": "integer"}),
         ("grep", "Search the repository with a regular expression; returns file:line:text matches.",
          {"pattern": "string", "path": "string"}),
         ("run_command", "Run a shell command in the repository and return its output.", {"command": "string"}),
         ("edit_file", "Replace an exact string in a file.", {"path": "string", "old": "string", "new": "string"})]


def agent_prompt(tok):
    """A coding-agent transcript (Qwen tool-call format) whose tool responses are real files of the engine; the scored
    tail is the last tool response: the next part of a file the agent already read the start of."""
    sp = lambda s: tok.encode(s, parse_special=True)   # noqa: E731
    tools = "\n".join(json.dumps({"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": {k: {"type": v} for k, v in p.items()}, "required": list(p)[:1]}}})
        for n, d, p in TOOLS)
    sysmsg = ("system\nYou are a coding agent working in the user's repository (a C++/CUDA inference engine with a Python "
              "server). Read before you change anything, keep edits small and explain what you did.\n\n# Tools\n\nYou may "
              "call one or more functions to assist with the user query.\n\nYou are provided with function signatures "
              "within <tools></tools> XML tags:\n<tools>\n" + tools + "\n</tools>\n\nFor each function call, return a json "
              "object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
              "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
    ids = [IM_START] + sp(sysmsg) + [IM_END] + sp("\n")

    def turn(role, body):
        return [IM_START] + sp(role + "\n" + body) + [IM_END] + sp("\n")

    def call(think, name, args):
        return turn("assistant", f"<think>\n{think}\n</think>\n\n<tool_call>\n" +
                    json.dumps({"name": name, "arguments": args}) + "\n</tool_call>")

    def resp(text):
        return turn("user", "<tool_response>\n" + text + "</tool_response>")

    def grep(pat, rel):
        out = []
        for i, l in enumerate(pathlib.Path(SRC, rel).read_text(errors="replace").splitlines(), 1):
            if re.search(pat, l):
                out.append(f"{rel}:{i}:{l}")
        return "\n".join(out[:60]) + "\n"

    ids += turn("user", "How does the engine write the teacher-forced log-probabilities (STRATA_LOGPOS) and how does "
                        "tools/quant_quality.py turn them into KL numbers? I want to add a top-5 overlap column.")
    steps = [("Find where STRATA_LOGPOS is read first.", "grep", {"pattern": "STRATA_LOGPOS", "path": "src"},
              grep(r"STRATA_LOGPOS", "src/program/generate.cpp") + grep(r"STRATA_LOGPOS", "src/core/verify.cpp")),
             ("The prompt's window loop is in generate.cpp; read it.", "read_file",
              {"path": "src/program/generate.cpp", "start_line": 12740, "end_line": 12900},
              lines(SRC + "/src/program/generate.cpp", 12740, 12900)),
             ("Now the writer itself in verify.cpp.", "read_file",
              {"path": "src/core/verify.cpp", "start_line": 3090, "end_line": 3260},
              lines(SRC + "/src/core/verify.cpp", 3090, 3260)),
             ("Check how the docs describe the yardstick.", "read_file",
              {"path": "docs/QUANTS.md", "start_line": 1, "end_line": 120}, lines(SRC + "/docs/QUANTS.md", 1, 120)),
             ("Read the tool from the top.", "read_file",
              {"path": "tools/quant_quality.py", "start_line": 1, "end_line": 260},
              lines(SRC + "/tools/quant_quality.py", 1, 260))]
    for think, name, args, out in steps:
        ids += call(think, name, args) + resp(out)
    ids += call("The compare function is further down; read the rest.", "read_file",
                {"path": "tools/quant_quality.py", "start_line": 260, "end_line": 380})
    tail = tok.encode(lines(SRC + "/tools/quant_quality.py", 260, 380))[:TAIL]
    return ids + [IM_START] + sp("user\n<tool_response>\n") + tail


def cuts(ids, short_read, root=2048):
    """The engine's split of a fresh prompt (generate.cpp: root_at, turn_at, n - 1) and whether each part goes through
    the windows (<= short_read tokens) - the head must be batched, the tail windowed."""
    n = len(ids)
    turn_at = max(i for i in range(1, n) if ids[i] == IM_START)
    root_at = -1
    for i in range(1, turn_at):
        if ids[i] == IM_START:
            root_at = i if i >= root else -1
            break
    parts, at = [], 0
    for to in sorted({x for x in (root_at, turn_at, n - 1) if x > 0}):
        parts.append((at, to, to - at <= short_read))
        at = to
    return parts


def cmd_build(a):
    tok = tokenizer()
    D = "/data/llm/docs/"
    ops = "".join(pathlib.Path(D + f).read_text() + "\n\n" for f in (
        "README.md", "TOPOLOGY.md", "RUNBOOK.md", "OPERATIONS.md", "ROLLBACK.md", "MODELS.md", "REMOTE-ACCESS.md",
        "BENCHMARKING.md", "MEASURED-FACTS.md", "ALTERNATIVES-TRIED.md", "LAYOUT.md"))
    ops += pathlib.Path("/data/llm/prompts/h3-director.md.long-version").read_text()
    notes = "".join(pathlib.Path("/data/llm/logs/strata/" + f).read_text() + "\n\n"
                    for f in ("README.md", "PLAN-multicard.md", "STATE-multicard.md"))
    P = [("prose-ops", doc_prompt(tok, "prose-ops", "operations notes", ops, 20000)),
         ("code-cpp", doc_prompt(tok, "code-cpp", "C++ source file", lines(SRC + "/src/program/generate.cpp", 11900), 28000)),
         ("agent-tools", agent_prompt(tok)),
         ("yaml-config", doc_prompt(tok, "yaml-config", "YAML configuration file",
                                    pathlib.Path(D + "llama-swap-config-ANNOTATED-archive-20260926.yaml").read_text(), 12000)),
         ("code-py", doc_prompt(tok, "code-py", "Python source file", lines(SRC + "/serve/server.py", 1), 14000)),
         ("prose-notes", doc_prompt(tok, "prose-notes", "engineering notes", notes, 11000))]
    short_read = max(len(ids) - 1 - max(i for i in range(1, len(ids)) if ids[i] == IM_START) for _, ids in P) + 8
    out = []
    for name, ids in P:
        parts = cuts(ids, short_read)
        if any(w for _, _, w in parts[:-1]) or not parts[-1][2] or parts[-1][1] - parts[-1][0] < TAIL:
            raise SystemExit(f"{name}: bad split {parts}")
        win = parts[-1][1] - parts[-1][0]
        print(f"{name:12s} {len(ids):6d} tokens: batched {parts[-1][0]:6d}, windows {win} (scored)")
        out.append({"name": name, "ids": ids, "turn_at": parts[-1][0], "scored": win})
    pathlib.Path(a.out).write_text(json.dumps({"short_read": short_read, "topk": TOPK, "prompts": out}))
    print(f"wrote {a.out}: {len(out)} prompts, {sum(len(p['ids']) for p in out)} tokens, --short-read {short_read}")
    return 0


# ------------------------------------------------------------------------------------------------ run
def engine_cmd(cfg, short_read):
    src, args, i = list(cfg["args"]), [], 0
    while i < len(src):
        if src[i] in DROP_VALUE:
            i += 2
            continue
        args.append(src[i])
        i += 1
    if len(cfg.get("gpu") or []) > 1 and "--layer-split" not in args:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    return [cfg["exe"], "--serve", *args, *FIXED, "--short-read", str(short_read)]


def cmd_run(a):
    cfg = json.load(open(a.config))
    st = json.load(open(a.set))
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("first.*")) + [out / "logpos.tsv", out / "meta.json"]:
        f.unlink(missing_ok=True)
    env = dict(os.environ)
    if cfg.get("gpu"):   # (an AMD config says "backend": "hip": HIP's own device list)
        if cfg.get("backend") == "hip":
            env["HIP_VISIBLE_DEVICES"] = ",".join(map(str, cfg["gpu"]))
        else:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["gpu"]))
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    env.update(FIXED_ENV)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    env.update({"SAGUARO_LOGPOS": str(out / "logpos.tsv"), "SAGUARO_LOGPOS_TOPK": str(TOPK),
                "SAGUARO_DUMP_FIRST_LOGITS": str(out / "first")})
    cmd = engine_cmd(cfg, st["short_read"])
    print("engine:", " ".join(cmd), flush=True)
    print("env:", " ".join(f"{k}={v}" for k, v in sorted(env.items()) if k.startswith(("SAGUARO_", "CUDA_", "HIP_"))), flush=True)
    if a.dry_run:
        return 0
    errf = open(out / "engine.log", "w")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True, cwd=cfg.get("cwd"),
                         env=env, bufsize=1)
    if a.pid_file:
        pathlib.Path(a.pid_file).write_text(str(p.pid))
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l) for l in p.stdout] + [q.put(None)], daemon=True).start()

    def wait_for(prefixes, timeout):
        t_end = time.time() + timeout
        while True:
            try:
                l = q.get(timeout=max(1.0, t_end - time.time()))
            except queue.Empty:
                l = None
                if time.time() < t_end:
                    continue
            if l is None:
                return None
            if l.startswith(prefixes):
                return l.strip()

    def fail(msg):
        p.kill()
        raise SystemExit(f"{a.out}: {msg}; see {out}/engine.log")

    t0 = time.time()
    if wait_for(("READY",), 1800) is None:
        fail("the engine did not get READY")
    meta = {"config": a.config, "cmd": cmd, "env": {k: v for k, v in env.items() if k.startswith("SAGUARO_")},
            "load_s": round(time.time() - t0, 1), "prompts": []}
    print(f"READY after {meta['load_s']} s", flush=True)

    def nlines():
        return sum(1 for _ in open(out / "logpos.tsv")) if (out / "logpos.tsv").exists() else 0

    only = set(a.only.split(",")) if a.only else None
    for k, pr in enumerate(st["prompts"]):
        if only and pr["name"] not in only:
            continue
        n0, t1 = nlines(), time.time()
        p.stdin.write("GEN 1 " + ",".join(map(str, pr["ids"])) + "\n")
        p.stdin.flush()
        done = wait_for(("DONE", "ERR"), 1800)
        if done is None or not done.startswith("DONE") or "nonfinite" in done:
            fail(f"{pr['name']}: the engine answered {done!r}")
        n1 = nlines()
        if n1 - n0 != pr["scored"]:
            fail(f"{pr['name']}: {n1 - n0} positions scored, expected {pr['scored']} (the split differs from the plan)")
        if not (out / f"first.{k}").exists():
            print(f"    {pr['name']}: no first-window logits (first.{k}): the full-vocab column stays empty", flush=True)
        meta["prompts"].append({"name": pr["name"], "first_line": n0, "end_line": n1, "first": f"first.{k}",
                                "s": round(time.time() - t1, 1)})
        print(f"{pr['name']:12s} {len(pr['ids']):6d} tokens, {n1 - n0} scored, {time.time() - t1:6.1f} s", flush=True)
    if a.greedy > 0:   # greedy answers after each prompt's head (the same load): tokens + where two runs part
        meta["greedy"] = []
        for k, pr in enumerate(st["prompts"]):
            if only and pr["name"] not in only:
                continue
            head = pr["ids"][:len(pr["ids"]) - pr["scored"]]
            t1 = time.time()
            p.stdin.write(f"GEN {a.greedy} " + ",".join(map(str, head)) + "\n")
            p.stdin.flush()
            toks = []
            while True:
                l = wait_for(("T ", "DONE", "ERR"), 1800)
                if l is None or l.startswith("ERR"):
                    fail(f"{pr['name']}: greedy answer: {l!r}")
                if l.startswith("DONE"):
                    break
                toks.append(int(l.split()[1]))
            meta["greedy"].append({"name": pr["name"], "tokens": toks, "s": round(time.time() - t1, 1)})
            print(f"{pr['name']:12s} greedy {len(toks)} tokens, {time.time() - t1:6.1f} s", flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    log = (out / "engine.log").read_text(errors="replace").splitlines()
    meta["engine_lines"] = [l for l in log if re.search(r"expert cache \d+ slots|layer split: CUDA\d runs|prompt chunk "
                                                       r"auto|K-quant prompt kernels|BF16", l)][:12]
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ compare
def read_run(d):
    d = pathlib.Path(d)
    meta = json.loads((d / "meta.json").read_text())
    rows = (d / "logpos.tsv").read_text().splitlines()
    res = {}
    for pr in meta["prompts"]:
        m = {}
        for l in rows[pr["first_line"]:pr["end_line"]]:
            f = l.split("\t")
            m[int(f[0])] = (int(f[1]), float(f[2]), [(int(x), float(y)) for x, y in (z.split(":") for z in f[8:8 + TOPK])])
        res[pr["name"]] = (m, d / pr["first"])
    return meta, res


def first_kl(fr, fa):
    import numpy as np

    def logp(f):
        x = np.fromfile(f, dtype=np.float32).astype(np.float64)
        x -= x.max()
        return x - np.log(np.exp(x).sum())
    lr, la = logp(fr), logp(fa)
    return float((np.exp(lr) * (lr - la)).sum()), bool(lr.argmax() == la.argmax())


def compare(ref, run):
    out, tot = {}, {"n": 0, "agree": 0, "conf": 0, "conf_agree": 0, "nll_r": 0.0, "nll_a": 0.0, "kls": [],
                    "fkl": [], "fagree": 0}
    for name, (rr, rf) in ref.items():
        if name not in run:   # a run of some prompts only (qkl.py run --only)
            continue
        ra, af = run[name]
        n = agree = conf = conf_agree = 0
        kls, nll_r, nll_a = [], 0.0, 0.0
        for q in sorted(set(rr) & set(ra)):
            tr, lr, kr = rr[q]
            ta, la, ka = ra[q]
            if tr != ta:
                raise SystemExit(f"{name} pos {q}: target {tr} != {ta} (not the same token set)")
            n += 1
            same = kr[0][0] == ka[0][0]
            agree += same
            if kr[0][1] - kr[1][1] >= RULE["conf_gap"]:
                conf += 1
                conf_agree += same
            nll_r -= lr
            nll_a -= la
            # KL(ref || run) over ref's top-64, both renormalized there; a token run does not list gets its 64th logp
            da, floor = dict(ka), min(b for _, b in ka)
            pr = [math.exp(b) for _, b in kr]
            pa = [math.exp(da.get(i, floor)) for i, _ in kr]
            sr, sa = sum(pr), sum(pa)
            kls.append(max(0.0, sum((x / sr) * math.log((x / sr) / (y / sa)) for x, y in zip(pr, pa) if x > 0)))
        fk, fs = first_kl(rf, af) if rf.exists() and af.exists() else (None, False)
        fkl = [fk] if fk is not None else []
        out[name] = summary(n, agree, conf, conf_agree, kls, nll_r, nll_a, fkl, int(fs))
        for k, v in (("n", n), ("agree", agree), ("conf", conf), ("conf_agree", conf_agree), ("nll_r", nll_r),
                     ("nll_a", nll_a), ("fagree", int(fs))):
            tot[k] += v
        tot["kls"] += kls
        tot["fkl"] += fkl
    out["ALL"] = summary(tot["n"], tot["agree"], tot["conf"], tot["conf_agree"], tot["kls"], tot["nll_r"], tot["nll_a"],
                         tot["fkl"], tot["fagree"])
    return out


def summary(n, agree, conf, conf_agree, kls, nll_r, nll_a, fkl, fagree):
    s = sorted(kls)
    n1 = max(n, 1)
    return {"positions": n, "top1": agree / n1, "top1_conf": conf_agree / max(conf, 1), "conf": conf,
            "flips": n - agree, "conf_flips": conf - conf_agree, "kl": sum(kls) / n1,
            "kl_p99": s[int(0.99 * (len(s) - 1))] if s else float("nan"), "kl_max": s[-1] if s else float("nan"),
            "ppl_ref": math.exp(nll_r / n1), "ppl": math.exp(nll_a / n1),
            "first_kl": sum(fkl) / len(fkl) if fkl else float("nan"),
            "first_kl_max": max(fkl) if fkl else float("nan"), "first_agree": fagree, "first_n": len(fkl)}


def verdict(v):
    why = []
    if v["kl"] >= RULE["kl"]:
        why.append(f"KL {v['kl']:.4f} >= {RULE['kl']}")
    if v["top1"] < RULE["top1"]:
        why.append(f"top-1 {100 * v['top1']:.2f}% < {100 * RULE['top1']:.0f}%")
    if v["top1_conf"] < RULE["top1_conf"]:
        why.append(f"confident top-1 {100 * v['top1_conf']:.2f}% < {100 * RULE['top1_conf']:.1f}%")
    if v["ppl"] > v["ppl_ref"] * (1 + RULE["ppl"]):
        why.append(f"PPL +{100 * (v['ppl'] / v['ppl_ref'] - 1):.2f}% > +{100 * RULE['ppl']:.1f}%")
    if v["first_n"] and v["first_kl"] >= RULE["first_kl"]:
        why.append(f"first-token KL {v['first_kl']:.4f} >= {RULE['first_kl']}")
    if not why:
        return "PASS"
    # only the overall top-1 missed, with a small KL and no confident flip: near-tie flips, a judgement call
    return ("MARGINAL (" if len(why) == 1 and why[0].startswith("top-1") else "FAIL (") + "; ".join(why) + ")"


def verdict_floor(v, f):
    """The rule with each limit widened to what the floor arm (a change that cannot cost quality: another summation
    order or MORE precision) already shows against the same reference: a prompt that flips on any rounding change
    is judged against that flip rate, not against 0."""
    why = []
    kl = max(RULE["kl"], 1.5 * f["kl"])
    if v["kl"] >= kl:
        why.append(f"KL {v['kl']:.4f} >= {kl:.4f}")
    t1 = min(RULE["top1"], f["top1"] - 0.01)
    if v["top1"] < t1:
        why.append(f"top-1 {100 * v['top1']:.2f}% < {100 * t1:.2f}%")
    tc = min(RULE["top1_conf"], f["top1_conf"] - 0.005)
    if v["top1_conf"] < tc:
        why.append(f"confident top-1 {100 * v['top1_conf']:.2f}% < {100 * tc:.2f}%")
    dp = max(RULE["ppl"], 1.5 * abs(f["ppl"] / f["ppl_ref"] - 1))
    if v["ppl"] > v["ppl_ref"] * (1 + dp):
        why.append(f"PPL +{100 * (v['ppl'] / v['ppl_ref'] - 1):.2f}% > +{100 * dp:.2f}%")
    fk = max(RULE["first_kl"], 1.5 * f["first_kl"]) if f["first_n"] else RULE["first_kl"]
    if v["first_n"] and v["first_kl"] >= fk:
        why.append(f"first-token KL {v['first_kl']:.4f} >= {fk:.4f}")
    return why


def cmd_compare(a):
    _, ref = read_run(a.ref)
    print(f"reference: {a.ref}   (KL = KL(ref || run) over ref's top-{TOPK} per scored position; conf = positions where "
          f"ref's top-2 gap >= {RULE['conf_gap']} nats; first = full-vocab KL / same argmax after the whole prompt)")
    hdr = (f"{'run':10s} {'prompt':12s} {'pos':>5s} {'top1':>7s} {'flips':>5s} {'conf top1':>9s} {'KL mean':>9s} "
           f"{'KL p99':>8s} {'KL max':>7s} {'PPL':>7s} {'dPPL':>7s} {'first KL':>9s} {'first=':>6s}")
    allres, verdicts = {}, []
    for d in a.runs:
        meta, run = read_run(d)
        r = compare(ref, run)
        name = pathlib.Path(d).name
        allres[name] = r
        print("\n" + hdr)
        for seg, v in r.items():
            if seg != "ALL" and not a.prompts:
                continue
            print(f"{name:10s} {seg:12s} {v['positions']:5d} {100 * v['top1']:6.2f}% {v['flips']:5d} "
                  f"{100 * v['top1_conf']:8.2f}% {v['kl']:9.6f} {v['kl_p99']:8.5f} {v['kl_max']:7.4f} {v['ppl']:7.4f} "
                  f"{100 * (v['ppl'] / v['ppl_ref'] - 1):+6.2f}% {v['first_kl']:9.6f} {v['first_agree']:3d}/{v['first_n']}")
        verdicts.append(f"{name:10s} {verdict(r['ALL'])}")
        for l in meta.get("engine_lines", []):
            if "slots" in l or "K-quant" in l:
                print(f"    {l[:150]}")
    print("\nPASS rule (each vs base): mean KL < %.2f, top-1 >= %.0f%%, top-1 >= %.1f%% where base's top-2 gap >= %.1f nats, "
          "PPL <= base +%.1f%%, mean first-token full-vocab KL < %.2f" % (RULE["kl"], 100 * RULE["top1"],
          100 * RULE["top1_conf"], RULE["conf_gap"], 100 * RULE["ppl"], RULE["first_kl"]))
    print("\n".join(verdicts))
    rg = json.loads((pathlib.Path(a.ref) / "meta.json").read_text()).get("greedy")
    for d in a.runs:
        g = json.loads((pathlib.Path(d) / "meta.json").read_text()).get("greedy")
        if not rg or not g:
            continue
        gr = {x["name"]: x["tokens"] for x in rg}
        same = tot = 0
        parts = []
        for x in g:
            r = gr.get(x["name"])
            if r is None:
                continue
            n = min(len(r), len(x["tokens"]))
            k = next((i for i in range(n) if r[i] != x["tokens"][i]), n)
            same += k
            tot += n
            parts.append(f"{x['name']} {k}/{n}")
        print(f"greedy {pathlib.Path(d).name}: identical prefix {same} of {tot} tokens ({100 * same / max(tot, 1):.1f}%): "
              + ", ".join(parts))
    if a.floor:
        fname = pathlib.Path(a.floor).name
        fl = allres.get(fname) or compare(ref, read_run(a.floor)[1])
        print(f"\nAgainst the floor arm {fname} (each prompt: KL < max(0.01, 1.5 x floor), top-1 >= min(98%, floor - 1%), "
              f"confident top-1 >= min(99.5%, floor - 0.5%), dPPL <= max(0.5%, 1.5 x |floor dPPL|), first KL < max(0.01, "
              f"1.5 x floor)):")
        print(f"{'floor':10s} " + "  ".join(f"{seg} KL {v['kl']:.4f} top1 {100 * v['top1']:.1f}%"
                                         for seg, v in fl.items() if seg != "ALL"))
        for name, r in allres.items():
            if name == fname:
                continue
            bad = {seg: verdict_floor(v, fl[seg]) for seg, v in r.items() if seg != "ALL" and seg in fl}
            bad = {k: w for k, w in bad.items() if w}
            print(f"{name:10s} " + ("PASS" if not bad else "FAIL (" + "; ".join(f"{k}: " + ", ".join(w) for k, w in bad.items()) + ")"))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(allres, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--env", action="append")
    p.add_argument("--pid-file")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", help="comma list of prompt names (a diagnosis run of some prompts)")
    p.add_argument("--greedy", type=int, default=0, help="then N greedy tokens after each prompt's head (same load)")
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--prompts", action="store_true")
    p.add_argument("--json")
    p.add_argument("--floor", help="a run dir (also one of the runs, or not): verdicts again with limits widened to it")
    a = ap.parse_args()
    return {"build": cmd_build, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
