#!/usr/bin/env python3
"""tools/ik2/conv_check.py - saguaro-ik2's kept and saved conversations, on a card: a conversation set aside for
another and taken up again, saved to a folder and resumed by a new engine, its last message edited, and the branch
left behind continued - each without reading again, each with the tokens an uninterrupted engine gives (greedy).

    tools/ik2/conv_check.py --gpus 2 [--build build-ik2b] [--ctx 16384] [--body 3000]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ik2"))
sys.path.insert(0, str(ROOT / "tools"))
import bench as B       # noqa: E402
import t4_check as TC   # noqa: E402

HEAD = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
OPEN = "<|message_user|><|content_text|>"
ASK = "<|end_message|><|message_model|><|content_text|>"
END = "<|end_message|><|content_model_end_sampling|>"


class Run:
    def __init__(self, a, gpus, tag, env):
        e = TC.env_for(gpus)
        e.update(SAGUARO_IK2_STATS="1", **env)
        args = ["--model", TC.MODEL, "--serve", "--max-context", str(a.ctx), "--gpus", str(len(gpus)), "--chunk", "1024"]
        self.eng = TC.Engine(a.build / "saguaro-ik2", args, e, a.out / ("engine_%s.log" % tag))
        if self.eng.until(lambda l: l.startswith("READY "), 1800) is None:
            raise RuntimeError("engine did not come up: " + " | ".join(self.eng.lines[-6:]))

    def gen(self, ids, n=48):
        t0 = time.time()
        self.eng.p.stdin.write("GEN %d temperature=0 %s\n" % (n, ",".join(map(str, ids))))
        self.eng.p.stdin.flush()
        got = self.eng.until(lambda l: l.startswith("DONE "), 1800)
        if got is None:
            raise RuntimeError("engine ended: " + " | ".join(self.eng.lines[-6:]))
        d = got[-1].split()
        notes = [l[6:] for l in got if l.startswith("NOTE ")]
        return dict(toks=[int(l.split()[1]) for l in got if l.startswith("T ")], prompt=int(d[2]), reused=int(d[8]),
                    read_s=float(d[3]) / 1e3, wall=time.time() - t0, notes=notes)

    def close(self):
        self.eng.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="2")
    ap.add_argument("--build", type=Path, default=ROOT / "build-ik2b")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--body", type=int, default=3000)
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    a.out = a.build / "conv_check"
    a.out.mkdir(parents=True, exist_ok=True)
    folder = a.out / "slots"
    shutil.rmtree(folder, ignore_errors=True)
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_gguf(TC.MODEL)
    text = tok.encode(B.corpus(), parse_special=False)
    enc = lambda s: tok.encode(s, parse_special=True)   # noqa: E731

    def first(k):   # conversation k's first prompt
        return enc(HEAD + OPEN) + text[k * 7000:k * 7000 + a.body] + enc("\n\nSummarise the text above in three sentences." + ASK)

    def turn(prev, ans, q):
        return prev + ans + (enc(END) if not ans or ans[-1] != enc(END)[-1] else []) + enc(OPEN + q + ASK)

    Q2, Q2B, Q3 = "Now list three key terms from it.", "Now name its single most important idea.", "Explain the first term."
    res = {}

    def step(name, r, want_toks=None, min_reused=None):
        ok = True
        if want_toks is not None:
            ok = ok and r["toks"] == want_toks
        if min_reused is not None:
            ok = ok and r["reused"] >= min_reused
        res[name] = ok
        print("%-44s %s  prompt %5d, kept %5d, read %.2f s, wall %.2f s%s%s" %
              (name, "PASS" if ok else "FAIL", r["prompt"], r["reused"], r["read_s"], r["wall"],
               "" if want_toks is None else ("; tokens %s" % ("alike" if r["toks"] == want_toks else "DIFFER")),
               "".join("\n      note: " + n for n in r["notes"])), flush=True)

    B.wait_free(gpus)
    line = B.claim("conv-check", gpus)
    try:
        # the reference: one engine, conversation A alone
        ref = Run(a, gpus, "ref", {"SAGUARO_IK2_KEEP": "0"})
        a1 = ref.gen(first(0))
        pa2 = turn(first(0), a1["toks"], Q2)
        a2 = ref.gen(pa2)
        ref.close()
        ref = Run(a, gpus, "ref2", {"SAGUARO_IK2_KEEP": "0"})   # ... and its turn 2 asked another way, straight after turn 1
        ref.gen(first(0))
        pa2b = turn(first(0), a1["toks"], Q2B)
        a2b = ref.gen(pa2b)
        ref.close()
        print("reference: turn 1 %d tokens, turn 2 %d, turn 2 (edited) %d" % (len(a1["toks"]), len(a2["toks"]), len(a2b["toks"])))

        e = Run(a, gpus, "keep", {"SAGUARO_IK2_SLOT_DIR": str(folder)})
        r = e.gen(first(0))
        step("A, turn 1", r, a1["toks"])
        r = e.gen(first(1))
        step("B, turn 1 (A is set aside)", r)
        b1 = r["toks"]
        r = e.gen(pa2)
        step("A, turn 2 after B (nothing read again)", r, a2["toks"], len(first(0)))
        r = e.gen(turn(first(1), b1, Q2))
        step("B, turn 2 after A", r, None, len(first(1)))
        r = e.gen(pa2)
        step("A, turn 2 again", r, a2["toks"], len(first(0)))
        r = e.gen(pa2b)
        step("A, turn 2 edited (back to its message)", r, a2b["toks"], len(first(0)))
        pa3 = turn(pa2, a2["toks"], Q3)
        r = e.gen(pa3)
        step("A, turn 3 of the first branch (kept)", r, None, len(pa2))
        a3 = r["toks"]
        e.close()   # (QUIT: the conversations go to the folder)
        files = sorted(p.name for p in folder.glob("*.ik2c"))
        print("folder after the engine ended: %d file(s), %.1f MiB" %
              (len(files), sum(p.stat().st_size for p in folder.glob("*.ik2c")) / 2**20))
        res["saved at the end"] = len(files) >= 2

        e = Run(a, gpus, "resume", {"SAGUARO_IK2_SLOT_DIR": str(folder)})
        r = e.gen(pa3)
        step("new engine: A, turn 3 again (from the folder)", r, a3, len(pa2))
        r = e.gen(turn(first(1), b1, Q2))
        step("new engine: B, turn 2 (from the folder)", r, None, len(first(1)))
        e.close()
    finally:
        time.sleep(1)
        print("cards after exit:", {g: u for g, (u, _) in TC.smi(gpus).items()})
        B.release(line)
    print()
    ok = bool(res) and all(res.values())
    print("CONV CHECK PASSED" if ok else "CONV CHECK FAILED: " + ", ".join(k for k, v in res.items() if not v))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
