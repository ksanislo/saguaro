#!/usr/bin/env bash
# scratch/card-test-mm3-read.sh - P124j on the card: a fresh ~30K low prompt read on saguaro-mm3, cut ONCE by a
# short normal request, must RESUME with the engine's own DONE reusing what it read: the resumed read's engine
# figure "> 0 reused" (before P124j the server never cut an mm3 read at all, and a STOP landing on the read's
# last chunk fell through to decode - src/mm3/mm3_main.cpp).
#
#   CFG=/path/to/<mm3>.json scratch/card-test-mm3-read.sh
#
# What it does: a test copy of the card's mm3 config on port 8092 with its OWN session dir and its OWN engine log
# and NO api_key (nothing of the live box's files is written, the test needs no key), then one phase:
#   phase L: a fresh ~30K-token prompt asked at "low" priority; at read position CUT_AT (default 10000) a short
#            normal request runs through and cuts the read; the server must log "prompt read paused at N of M
#            prompt tokens for a normal-priority request" and then the numberless continue line (the line driver
#            has no RESUME figure - P124b), and the low request's own line in the ENGINE log
#            "prompt <N> tokens = <R> reused + <K> read" must show R > 0: the cut read's DONE kept the conversation.
# PASS: exactly one cut, the resume line, and R > 0 for the big prompt. One read through the real server plus the
# model load; no benchmarks, nothing destructive.
set -u -o pipefail

PYTHON=${PYTHON:-/opt/llm/saguaro/.venv/bin/python}
PORT=${PORT:-8092}
CFG=${CFG:?the card model config JSON (the saguaro-mm3 one the live server uses)}
TOKENS=${TOKENS:-30000}
CUT_AT=${CUT_AT:-10000}
IDLE=${IDLE:-2}                      # SAGUARO_PRIORITY_IDLE_S for the resume (the live default 30 only slows this)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$HERE")
DIR="$HERE/mm3-read"
LOG="$DIR/serve.log"
ELOG="$DIR/engine.log"
mkdir -p "$DIR" "$DIR/sessions"
rm -f "$LOG" "$ELOG" "$DIR"/resp_*.json "$DIR"/wall_* "$DIR"/body_*.json "$DIR/prompt_*.txt"

fail() { echo "FAIL: $*" >&2
         [ -f "$DIR/serve.pid" ] && kill "$(cat "$DIR/serve.pid")" 2>/dev/null
         echo "--- last of $LOG ---"; tail -40 "$LOG" 2>/dev/null
         echo "--- last of $ELOG ---"; tail -20 "$ELOG" 2>/dev/null; exit 1; }

# ---- the test copy of the config: same engine, its own session dir and log, no api_key ------------------------
CFG_T="$DIR/mm3-read.json"
"$PYTHON" - "$CFG" "$CFG_T" "$DIR/sessions" "$ELOG" <<'PY' || fail "could not write the test config"
import json, sys
cfg = json.load(open(sys.argv[1]))
args = list(cfg.get("args") or [])
if "--session-dir" in args:
    args[args.index("--session-dir") + 1] = sys.argv[3]
else:
    args += ["--session-dir", sys.argv[3]]
cfg["args"] = args
cfg["log"] = sys.argv[4]              # the engine's stderr: its "prompt N tokens = R reused" line lands here
for k in ("api_key", "api_keys", "apiKey"):
    cfg.pop(k, None)                  # the test server runs without a key: nothing of the live one is copied
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY

# ---- start the server --------------------------------------------------------------------------------------
cd "$ROOT_DIR" || exit 1
export SAGUARO_PRIORITY_IDLE_S="$IDLE"
nohup "$PYTHON" -m serve.server --config "$CFG_T" --port "$PORT" >"$LOG" 2>&1 &
echo $! > "$DIR/serve.pid"
echo "server pid $(cat "$DIR/serve.pid") on :$PORT, log $LOG, engine log $ELOG (loading the model ...)"
for _ in $(seq 1 240); do
    curl -fs -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$(cat "$DIR/serve.pid")" 2>/dev/null || fail "the server died while loading; see $LOG"
    sleep 5
done
curl -fs -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || fail "no /health after 20 min"
MODEL=$("$PYTHON" - "$PORT" <<'PY'
import json, sys, urllib.request
print(json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models"))["data"][0]["id"])
PY
) || fail "no model id on :$PORT"
echo "model: $MODEL"

# ---- prompts: fresh random words, ~4 chars a token ------------------------------------------------------------
prompt() { "$PYTHON" - "$1" "$2" <<'PY'
import random, sys
rng = random.Random(f"p124j-{sys.argv[2]}")
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu analysis bucket canyon engine floor "
         "gamma hinge input junction latent metric needle oracle pixel query radio sector token unity vector "
         "window xenon yield zenith").split()
out, n = [], 0
while n < int(sys.argv[1]):
    w = rng.choice(words)
    out.append(w)
    n += len(w) + 1
print(f"P124j mm3-read probe {sys.argv[2]}: " + " ".join(out))
PY
}
prompt "$((TOKENS * 4))" L > "$DIR/prompt_L.txt" || fail "prompt L"

ask() {   # ask <prompt-file> <tag> <priority>: one request; the wall seconds land in $DIR/wall_<tag>
    "$PYTHON" - "$1" "$MODEL" "$3" <<'PY' > "$DIR/body_$2.json"
import json, sys
print(json.dumps({"model": sys.argv[2], "priority": sys.argv[3], "temperature": 0, "max_tokens": 40,
                  "messages": [{"role": "user", "content": open(sys.argv[1]).read()}]}))
PY
    curl -s -o "$DIR/resp_$2.json" -w "%{time_total}" -m 3600 -X POST \
        "http://127.0.0.1:$PORT/v1/chat/completions" -H "content-type: application/json" \
        --data-binary "@$DIR/body_$2.json" > "$DIR/wall_$2"
}
short() {   # one short normal request - the one that cuts the low read
    "$PYTHON" - "$MODEL" > "$DIR/body_short.json" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "priority": "normal", "max_tokens": 24, "temperature": 0,
                  "messages": [{"role": "user", "content": "Reply with the word OK."}]}))
PY
    curl -s -o /dev/null -m 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "content-type: application/json" --data-binary "@$DIR/body_short.json"
}
bytes_now() { wc -c < "$LOG" | tr -d ' '; }
lines_from() { tail -c +"$(( $1 + 1 ))" "$LOG"; }
maxpos() { lines_from "$1" | tr -d ',' | grep -oE "reading the prompt: [0-9]+ of" | awk '{print $4}' | sort -n | tail -1; }
count() { lines_from "$1" | grep -cF "$2" || true; }

# ---- phase L: the fresh low read, cut once, then finished ------------------------------------------------------
echo "== phase L: a fresh ${TOKENS}-token-ish low read, cut at ${CUT_AT} by one short normal request =="
MARK_L=$(bytes_now)
ask "$DIR/prompt_L.txt" L low &
LOW_PID=$!
while :; do
    p=$(maxpos "$MARK_L"); p=${p:-0}
    [ "$p" -ge "$CUT_AT" ] && break
    kill -0 "$LOW_PID" 2>/dev/null || fail "the low read ended before position $CUT_AT"
    sleep 1
done
echo "-- read position ${p} >= ${CUT_AT}: running the short normal request"
short || fail "the short normal request failed"
for _ in $(seq 1 180); do
    [ "$(count "$MARK_L" 'prompt read paused')" -ge 1 ] && break
    sleep 1
done
[ "$(count "$MARK_L" 'prompt read paused')" -ge 1 ] || fail "no 'prompt read paused' line - the mm3 read was not cut"
[ "$(count "$MARK_L" 'for a normal-priority request')" -ge 1 ] || fail "the pause did not name the normal request"
for _ in $(seq 1 600); do        # the resume waits out SAGUARO_PRIORITY_IDLE_S; the line then says it continues
    [ "$(count "$MARK_L" 'prompt read continues from where the engine kept it')" -ge 1 ] && break
    sleep 1
done
[ "$(count "$MARK_L" 'prompt read continues from where the engine kept it')" -ge 1 ] \
    || fail "no continue line for the resumed read"
wait "$LOW_PID" || fail "the low request failed (see $DIR/resp_L.json): prompt too long for this config?"
grep -qE 'finish' "$DIR/resp_L.json" 2>/dev/null || fail "the low request did not answer"
[ "$(count "$MARK_L" 'prompt read paused')" -eq 1 ] || fail "not exactly one pause line"

# ---- the verdict: the engine's own figure for the big prompt's end line ---------------------------------------
kill "$(cat "$DIR/serve.pid")" 2>/dev/null
REUSED=$(grep -oE "prompt [0-9]+ tokens = [0-9]+ reused" "$ELOG" \
         | awk -v min="$((TOKENS / 2))" '{gsub(/[^0-9 ]/, ""); if ($1 >= min) print $2}' | tail -1)
[ -n "$REUSED" ] || fail "no 'prompt N tokens = R reused' line for the ${TOKENS}-token prompt in $ELOG"
WALL_L=$(cat "$DIR/wall_L")
echo "   resumed read's engine figure: $REUSED reused of the ~${TOKENS}-token prompt   wall: ${WALL_L}s"
[ "$REUSED" -gt 0 ] || fail "the resumed read's DONE reported 0 reused: the kept read was lost"
echo "PASS: P124j - the cut low mm3 read was resumed and the engine itself reports $REUSED tokens reused"
