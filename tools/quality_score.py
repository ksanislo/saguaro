"""tools/quality_score.py - a task-score runner for QFN quant quality (P125h). The README's Quality section has no
Qwen3.8-Flash-Next numbers, and a full-precision reference (~250 GB) cannot be run here, so the quality of each
quant (UD-Q4_K_XL, Q8_0, GSQ-RCO IQ3_S, Coder IQ1_M) is a TASK SCORE on a fixed question set: run this against a
started Saguaro server - one server per quant, the same set and flags for all of them - and compare the scores.

    python3 tools/quality_score.py http://127.0.0.1:8080 --set gpqa-diamond --set-file gpqa.jsonl
    python3 tools/quality_score.py http://127.0.0.1:8080 --set agentic --set-file agent.jsonl --limit 2 --key-file ~/.saguaro_key

Every question is one normal chat request: thinking ON (chat_template_kwargs enable_thinking), temperature 0,
a max_tokens cap (--max-tokens, default 8192) and a per-question timeout (--timeout, default 600 s). The runner
extracts the answer the way the set defines it - for gpqa-diamond the multiple-choice letter (the reply names
it, or it names exactly one of the choices), for agentic the tool call (its name and arguments must match the
question's "expect") - appends one JSON line per question to --out (id, answer, correct, tokens, seconds), and
resumes: ids already in --out are skipped, so an interrupted run picks up where it stopped. At the end it prints
the accuracy with a 95 % confidence interval (the Wilson score interval).

The question set comes from a LOCAL file (--set-file): a JSON array or one JSON object per line (JSONL):
  gpqa-diamond  {"id": "q1", "question": "...", "choices": ["...", "...", "...", "..."], "answer": "C"}
                ("id" optional; "answer" may also be the choice's text)
  agentic       {"id": "t1", "question": "...", "tools": [the tool schemas, OpenAI form],
                 "expect": {"name": "calc", "arguments": {"expr": "2 + 2"}}}
NO copyrighted question text ships with Saguaro: the user points at their own file.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 600.0


# ------------------------------------------------------------------ the score

def wilson_interval(correct: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The accuracy of correct/n with a 95 % Wilson score interval (normal z): never outside 0..1, and it does
    not collapse to 0..0 when a short run scores everything or nothing."""
    if n <= 0:
        return 0.0, 1.0
    p = correct / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


# ------------------------------------------------------------------ question sets

def load_set(path: str | Path) -> list[dict]:
    """A question set from a local file: a JSON array, or one JSON object per line (JSONL); blank lines skipped.
    Every record needs a question; an id is filled in when the record has none (q1, q2, ... in file order)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    else:
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and isinstance(data.get("questions"), list):
            records = data["questions"]
        else:
            records = [data]
    questions = []
    for rec in records:
        if not isinstance(rec, dict) or not str(rec.get("question") or "").strip():
            raise ValueError(f"{path}: a record with no question: {str(rec)[:80]}")
        questions.append(rec)
    for i, q in enumerate(questions):
        q.setdefault("id", f"q{i + 1}")
    return questions


# ------------------------------------------------------------------ answer extraction

def choice_texts(question: dict) -> list[str]:
    return [c if isinstance(c, str) else str((c or {}).get("text", "")) for c in (question.get("choices") or [])]


def extract_letter(text: str, question: dict) -> str | None:
    """The multiple-choice letter a GPQA-style reply picks, or None when it never commits: the word answer /
    option / choice / letter followed by a letter (the LAST such mention wins - a reply that rules the other
    options out before answering is read by its ending); a letter in brackets; a lone letter on its own line;
    and if none of those, exactly one of the choices named by its own text (a reply naming two is not scored)."""
    n = len(choice_texts(question)) or 4
    up = "ABCDEFGH"[:n]
    hits = re.findall(r"(?i:(?:final\s+)?(?:answer|option|choice|letter))[\s\S]{0,20}?\(?([" + up + r"])\)?\b",
                      text or "")        # the word may be written any way, the letter not: "answer is a cat" names nothing
    if not hits:
        hits = re.findall(r"[\(\[<]([" + up + up.lower() + r"])[\)\]>]", text or "")
    if not hits:
        hits = re.findall(r"(?m)^[ \t]*([" + up + r"])[.):]?[ \t]*$", text or "")
    if hits:
        return hits[-1].upper()
    low = " " + (text or "").lower() + " "
    named = [i for i, c in enumerate(choice_texts(question))
             if c.strip() and re.search(r"\b" + re.escape(c.lower()) + r"\b", low)]
    return "ABCDEFGH"[named[0]] if len(named) == 1 else None


def norm_value(v):
    """An argument value in comparable form: numbers as float (a schema's 5 and "5" mean one value), strings
    trimmed, containers walked."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        try:                              # "5" and 5 mean one value (a schema-less call may send either)
            return float(v)
        except ValueError:
            return v
    if isinstance(v, list):
        return [norm_value(x) for x in v]
    if isinstance(v, dict):
        return {k: norm_value(x) for k, x in v.items()}
    return v


def norm_args(a) -> dict:
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return {"__raw__": a.strip()}
    return {k: norm_value(v) for k, v in a.items()} if isinstance(a, dict) else {"__raw__": norm_value(a)}


def extract_call(msg: dict) -> dict | None:
    """The model's first tool call, as {"name", "arguments"} with normalized arguments; None if it called nothing."""
    for c in (msg.get("tool_calls") or []):
        fn = c.get("function") or {}
        if fn.get("name"):
            return {"name": fn["name"], "arguments": norm_args(fn.get("arguments") or {})}
    return None


def expected_letter(question: dict) -> str:
    """What the record marks correct, as a letter: an "answer" that is already one, else the choice it spells."""
    want = str(question.get("answer", "")).strip()
    texts = choice_texts(question)
    if want.upper() in [l.upper() for l in "ABCDEFGH"[:len(texts) or 4]] and len(want) == 1:
        return want.upper()
    for i, c in enumerate(texts):
        if c.strip().lower() == want.lower():
            return "ABCDEFGH"[i]
    return want.upper()[:1]


def is_correct(question: dict, answer) -> bool:
    if isinstance(answer, dict):                     # agentic: the call's name matches and every expected argument does
        want = question.get("expect") or {}
        if not want or answer.get("name") != want.get("name"):
            return False
        got = answer.get("arguments") or {}
        return all(norm_value(got.get(k)) == norm_value(v) for k, v in (want.get("arguments") or {}).items())
    return bool(answer) and answer == expected_letter(question)


# ------------------------------------------------------------------ the request

def ask(url: str, key: str, question: dict, timeout: float, max_tokens: int) -> dict:
    """One question as a normal chat request (thinking on, temperature 0): the reply's visible text, its tool
    calls, the completion tokens and the seconds it took."""
    body = {"model": "strata", "messages": [{"role": "user", "content": str(question["question"])}],
            "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True}}     # thinking ON: the reasoning the quant actually does
    if question.get("tools"):
        body["tools"] = question["tools"]
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    secs = round(time.time() - t0, 2)
    choice = (out.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return {"text": msg.get("content") or "", "thinking": msg.get("reasoning_content") or "",
            "calls": msg.get("tool_calls") or [], "finish": choice.get("finish_reason"),
            "tokens": (out.get("usage") or {}).get("completion_tokens"), "seconds": secs}


def answer_for(question: dict, reply: dict):
    """The answer, extracted the way the set defines it: a tool call for an agentic question (it has an "expect"),
    the multiple-choice letter otherwise. Empty string / None when the reply never commits to one."""
    if question.get("expect") is not None:
        return extract_call({"tool_calls": reply["calls"]})
    return extract_letter(reply["text"], question) or ""


# ------------------------------------------------------------------ the run

def read_done(out_path: Path) -> dict:
    """The ids already scored in --out (one JSON line per question, blank and half-written lines skipped),
    so a partial run resumes instead of starting over."""
    done = {}
    if not out_path.exists():
        return done
    for ln in out_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(ln)
            if isinstance(rec, dict) and "id" in rec:
                done[str(rec["id"])] = rec
        except ValueError:
            continue
    return done


def run(url: str, key: str, questions: list[dict], out_path: Path, timeout: float, max_tokens: int) -> int:
    done = read_done(out_path)
    todo = [q for q in questions if str(q["id"]) not in done]
    if len(todo) < len(questions):
        print(f"resuming: {len(questions) - len(todo)} of {len(questions)} already in {out_path}", flush=True)
    with out_path.open("a", encoding="utf-8") as f:
        for q in todo:
            rec = {"id": q["id"]}
            try:
                reply = ask(url, key, q, timeout, max_tokens)
                answer = answer_for(q, reply)
                rec.update({"answer": answer if answer is not None else "", "correct": is_correct(q, answer),
                            "tokens": reply["tokens"], "seconds": reply["seconds"]})
                if reply.get("finish") not in ("stop", "tool_calls", None):
                    rec["finish"] = reply["finish"]        # (length/timeout/cancel: the cap, not the model, cut it off)
            except (OSError, urllib.error.URLError, ValueError, KeyError) as e:
                rec.update({"answer": "", "correct": False, "tokens": None, "seconds": None,
                            "error": f"{type(e).__name__}: {e}"[:200]})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[str(q["id"])] = rec
            got = rec["answer"] if isinstance(rec["answer"], str) else json.dumps(rec["answer"], ensure_ascii=False)
            print(f"{q['id']}: {'correct' if rec['correct'] else 'WRONG'}  {got[:60]!r}"
                  + (f"  ({rec['tokens']} tok, {rec['seconds']} s)" if rec.get("error") is None
                     else f"  ERROR {rec['error']}"), flush=True)
    scored = [done[str(q["id"])] for q in questions if str(q["id"]) in done]
    correct = sum(1 for r in scored if r.get("correct"))
    lo, hi = wilson_interval(correct, len(scored))
    print(f"accuracy: {correct}/{len(scored)} = {100.0 * correct / len(scored):.1f} %   "
          f"95 % CI [{100 * lo:.1f}, {100 * hi:.1f}] %")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 epilog="the question set lives in your own --set-file; no copyrighted text ships")
    ap.add_argument("base_url", help="the running server, e.g. http://127.0.0.1:8080")
    ap.add_argument("--key-file", help="a file holding the API key (sent as Authorization: Bearer ...)")
    ap.add_argument("--set", choices=["gpqa-diamond", "agentic"], dest="set_name",
                    help="how answers are scored (default: what the records in --set-file look like)")
    ap.add_argument("--set-file", help="the questions, a local JSON array or JSONL file (gpqa-diamond needs one)")
    ap.add_argument("--limit", type=int, help="score only the first N questions")
    ap.add_argument("--out", default="results.jsonl", help="the results file, one JSON line per question "
                                                          "(append/resume; default results.jsonl)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per question")
    a = ap.parse_args()
    if a.set_file:
        try:
            questions = load_set(a.set_file)
        except (OSError, ValueError) as e:
            print(f"the question set: {e}", file=sys.stderr)
            return 2
    else:
        print("--set-file is needed: Saguaro ships no copyrighted question text (GPQA-Diamond is CC BY 4.0 - "
              "take your own copy and point --set-file at it).", file=sys.stderr)
        return 2
    kind = a.set_name or ("agentic" if any(q.get("expect") is not None for q in questions) else "gpqa-diamond")
    bad = [q["id"] for q in questions
           if (kind == "agentic" and (q.get("expect") or {}).get("name") is None)
           or (kind != "agentic" and not choice_texts(q))]
    if bad:
        need = "expect.name" if kind == "agentic" else "choices"
        print(f"{kind}: records with no {need}: " + ", ".join(map(str, bad)), file=sys.stderr)
        return 2
    if a.limit is not None:
        questions = questions[:max(0, a.limit)]
    key = Path(a.key_file).read_text(encoding="utf-8").strip() if a.key_file else ""
    return run(a.base_url, key, questions, Path(a.out), a.timeout, a.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
