#!/usr/bin/env bash
# scratch/card-test-lost-read.sh - P124b on the card: a fresh ~100K low prompt read interrupted three times by a
# short normal request must RESUME with the engine's own reuse every time, not re-read from 0.
#
#   CFG=/path/to/<model>.json scratch/card-test-lost-read.sh
#
# What it does: a test copy of the card's QFN config on port 8092 with its OWN session dir (nothing of the live
# box's files is written), then:
#   phase L: a fresh ~100K-token prompt asked at "low" priority; at read positions 20000 / 40000 / 60000 a short
#            normal request runs through and cuts the read; every cut must log
#            "the low prompt read continued: M of N tokens reused" with M > 0 (P124b: M is the engine's own RESUME
#            figure - before the fix the line said "continues from where the engine kept it" while the engine had
#            dropped the part read and M was really 0), and the engine's own read figures must add up to about one
#            read of the prompt, never a second whole "0 reused + N read" pass;
#   phase C: a SECOND fresh prompt of the same size, uninterrupted: the control read time.
# PASS: three cuts, M > 0 at each one (growing), and the interrupted read's engine read time within RATIO of the
# control's (default 1.30 - the lost-read shape added a whole second read of the prompt: +100%).  One pair of
# reads through the real server plus the model load; no benchmarks, nothing destructive.
set -u -o pipefail

PYTHON=${PYTHON:-/opt/llm/saguaro/.venv/bin/python}
PORT=${PORT:-8092}
CFG=${CFG:?the card model config JSON (the one the live server uses)}
TOKENS=${TOKENS:-100000}
CUTS=${CUTS:-"20000 40000 60000"}
RATIO=${RATIO:-1.30}
IDLE=${IDLE:-2}                      # SAGUARO_PRIORITY_IDLE_S for the resume (the live default 30 only slows this)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$HERE")
DIR="$HERE/lost-read"
LOG="$DIR/serve.log"
mkdir -p "$DIR" "$DIR/sessions"
rm -f "$LOG" "$DIR"/resp_*.json "$DIR"/wall_* "$DIR"/body_*.json "$DIR/prompt_*.txt"
CUT_N=$(echo "$CUTS" | wc -w)

fail() { echo "FAIL: $*" >&2
         [ -f "$DIR/serve.pid" ] && kill "$(cat "$DIR/serve.pid")" 2>/dev/null
         echo "--- last of $LOG ---"; tail -40 "$LOG" 2>/dev/null; exit 1; }

# ---- the test copy of the config: same engine, its own session dir ------------------------------------------
CFG_T="$DIR/lost-read.json"
"$PYTHON" - "$CFG" "$CFG_T" "$DIR/sessions" <<'PY' || fail "could not write the test config"
import json, sys
cfg = json.load(open(sys.argv[1]))
args = list(cfg.get("args") or [])
if "--session-dir" in args:
    args[args.index("--session-dir") + 1] = sys.argv[3]
else:
    args += ["--session-dir", sys.argv[3]]
cfg["args"] = args
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY

# ---- start the server --------------------------------------------------------------------------------------
cd "$ROOT_DIR" || exit 1
export SAGUARO_PRIORITY_IDLE_S="$IDLE"
nohup "$PYTHON" -m serve.server --config "$CFG_T" --port "$PORT" >"$LOG" 2>&1 &
echo $! > "$DIR/serve.pid"
echo "server pid $(cat "$DIR/serve.pid") on :$PORT, log $LOG (loading the model ...)"
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

# ---- prompts: fresh random words, ~4 chars a token, a different seed per phase ------------------------------
prompt() { "$PYTHON" - "$1" "$2" <<'PY'
import random, sys
rng = random.Random(f"p124b-{sys.argv[2]}")
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu analysis bucket canyon engine floor "
         "gamma hinge input junction latent metric needle oracle pixel query radio sector token unity vector "
         "window xenon yield zenith").split()
out, n = [], 0
while n < int(sys.argv[1]):
    w = rng.choice(words)
    out.append(w)
    n += len(w) + 1
print(f"P124b lost-read probe {sys.argv[2]}: " + " ".join(out))
PY
}
prompt "$((TOKENS * 4))" L > "$DIR/prompt_L.txt" || fail "prompt L"
prompt "$((TOKENS * 4))" C > "$DIR/prompt_C.txt" || fail "prompt C"

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
read_ms() {   # the engine's own read milliseconds, summed over the legs logged since a byte offset
    lines_from "$1" | grep -oE "reused \+ [0-9]+( of [0-9]+)? read in [0-9]+ ms" \
        | awk '{s += $(NF-1)} END {printf "%.0f", s + 0}'
}

# ---- phase L: the fresh low read, cut three times -----------------------------------------------------------
echo "== phase L: a fresh ${TOKENS}-token-ish low read, ${CUT_N} normal interruptions =="
MARK_L=$(bytes_now)
ask "$DIR/prompt_L.txt" L low &
LOW_PID=$!
i=0
for target in $CUTS; do
    i=$((i + 1))
    while :; do
        p=$(maxpos "$MARK_L"); p=${p:-0}
        [ "$p" -ge "$target" ] && break
        kill -0 "$LOW_PID" 2>/dev/null || fail "the low read ended before position $target (cut $i)"
        sleep 1
    done
    echo "-- cut $i at read position ${p} >= ${target}: running a short normal request"
    short || fail "the short normal request (cut $i) failed"
    for _ in $(seq 1 180); do
        [ "$(count "$MARK_L" 'prompt read paused')" -ge "$i" ] && break
        sleep 1
    done
    [ "$(count "$MARK_L" 'prompt read paused')" -ge "$i" ] || fail "no 'prompt read paused' line for cut $i"
    for _ in $(seq 1 600); do        # the resume waits out SAGUARO_PRIORITY_IDLE_S; the line then carries M
        [ "$(count "$MARK_L" 'prompt read continued')" -ge "$i" ] && break
        sleep 1
    done
    [ "$(count "$MARK_L" 'prompt read continued')" -ge "$i" ] || fail "no 'prompt read continued' line for cut $i"
done
wait "$LOW_PID" || fail "the low request failed (see $DIR/resp_L.json): prompt too long for this config?"
grep -qE 'finish' "$DIR/resp_L.json" 2>/dev/null || fail "the low request did not answer"
[ "$(count "$MARK_L" 'prompt read paused')" -eq "$CUT_N" ] || fail "not exactly $CUT_N pause lines"
mapfile -t REUSED < <(lines_from "$MARK_L" \
    | grep -oE "prompt read continued: [0-9,]+ of [0-9,]+ tokens reused" \
    | sed -E 's/^prompt read continued: ([0-9,]+) of .*$/\1/' | tr -d ',')
[ "${#REUSED[@]}" -eq "$CUT_N" ] || fail "continue lines without figures: ${REUSED[*]:-none}"
prev=0
for r in "${REUSED[@]}"; do
    [ "$r" -gt 0 ] || fail "a resumed read reported 0 reused - the P124b loss (figures: ${REUSED[*]})"
    [ "$r" -ge "$prev" ] || fail "the reuse did not grow across the cuts: ${REUSED[*]}"
    prev=$r
done
READ_L=$(read_ms "$MARK_L")
WALL_L=$(cat "$DIR/wall_L")
echo "   reused per resume: ${REUSED[*]}   engine read ms (legs summed): $READ_L   wall: ${WALL_L}s"

# ---- phase C: a fresh prompt of the same size, uninterrupted -------------------------------------------------
echo "== phase C: the control read (same size, no interruption) =="
MARK_C=$(bytes_now)
ask "$DIR/prompt_C.txt" C low || fail "the control request failed (see $DIR/resp_C.json)"
grep -qE 'finish' "$DIR/resp_C.json" 2>/dev/null || fail "the control did not answer"
[ "$(count "$MARK_C" 'prompt read paused')" -eq 0 ] || fail "the control was cut too (a request leaked from L?)"
READ_C=$(read_ms "$MARK_C")
WALL_C=$(cat "$DIR/wall_C")
[ "$READ_C" -gt 0 ] || fail "no engine read figures for the control"
echo "   engine read ms: $READ_C   wall: ${WALL_C}s"

# ---- the verdict ---------------------------------------------------------------------------------------------
kill "$(cat "$DIR/serve.pid")" 2>/dev/null
ok=$(awk -v a="$READ_L" -v c="$READ_C" -v r="$RATIO" 'BEGIN { print (a <= c * r + 15000) ? "yes" : "no" }')
echo "interrupted read ${READ_L} ms vs control ${READ_C} ms (ratio $(awk -v a="$READ_L" -v c="$READ_C" \
    'BEGIN { printf "%.2f", a / c }')), resumes with M = ${REUSED[*]}"
[ "$ok" = yes ] || fail "the interrupted read cost ${READ_L} ms against a ${READ_C} ms control: the lost-read shape"
echo "PASS: P124b - each paused low read resumed with M > 0 and the total cost one read, not two"
