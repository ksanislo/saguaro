#!/usr/bin/env bash
# tools/card_tests/lost_read.sh - P124b on the card: a fresh ~100K low prompt read interrupted three times by a
# short normal request must RESUME with the engine's own reuse every time, not re-read from 0.
#
#   CFG=/path/to/<model>.json ./tools/card_tests/lost_read.sh
#   ./tools/card_tests/lost_read.sh --out scratch/lost-read-2      # write under a different folder
#
# What it does: a test copy of the card model config on port 8092 with its OWN session dir, its OWN engine log
# and the api_key dropped (the readiness poll and the requests send none) - everything it writes lands under
# its --out folder (default scratch/lost-read) and nothing of the live box's config, log or sessions is
# touched (its config is only read).  Then:
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
# It waits until the port is free before starting, and on exit kills whatever still holds the port (the
# server's child engine can outlive the launcher).  Needs: curl + a python (the install's .venv by default).
set -u -o pipefail

OUT=scratch/lost-read
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT=${2:?--out wants a folder}; shift 2 ;;
        *) echo "unknown argument: $1 (see the header)"; exit 1 ;;
    esac
done
case "$OUT" in ""|.|..) echo "--out wants a dedicated folder, not '$OUT'"; exit 1 ;; esac

PYTHON=${PYTHON:-/opt/llm/saguaro/.venv/bin/python}
PORT=${PORT:-8092}
CFG=${CFG:?the card model config JSON (the one the live server uses; it is only read)}
TOKENS=${TOKENS:-100000}
CUTS=${CUTS:-"20000 40000 60000"}
RATIO=${RATIO:-1.30}
IDLE=${IDLE:-2}                      # SAGUARO_PRIORITY_IDLE_S for the resume (the live default 30 only slows this)
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR" || exit 1
CUT_N=$(echo "$CUTS" | wc -w)
DIR=$OUT
SERVE_LOG="$DIR/serve.log"
ELOG="$DIR/engine.log"               # the engine's own stderr, kept inside the out folder (never the install's log)

port_pids() {   # whatever still holds the port (a previous run's engine child outlived its launcher)
    ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE "pid=[0-9]+" | cut -d= -f2
    lsof -nP -t -i "TCP:$PORT" -s TCP:LISTEN 2>/dev/null
    true
}

cleanup() {   # on exit the port is ours again: kill the server and whatever still holds it (its engine child).
              # only once WE started one - a failure before that must leave a live server on the port alone
    [ -f "$DIR/serve.pid" ] || return 0
    kill "$(cat "$DIR/serve.pid")" 2>/dev/null
    for p in $(port_pids); do kill "$p" 2>/dev/null; done
    true
}

fail() { echo "FAIL: $*" >&2
         cleanup
         echo "--- last of $SERVE_LOG ---"; tail -40 "$SERVE_LOG" 2>/dev/null
         echo "--- last of $ELOG ---"; tail -20 "$ELOG" 2>/dev/null; exit 1; }

rm -rf "$DIR"; mkdir -p "$DIR" "$DIR/sessions"

# ---- the test copy of the config: same engine, its own session dir, log and no key ---------------------------
CFG_T="$DIR/lost-read.json"
"$PYTHON" - "$CFG" "$CFG_T" "$DIR/sessions" "$ELOG" <<'PY' || fail "could not write the test config"
import json, sys
cfg = json.load(open(sys.argv[1]))
args = list(cfg.get("args") or [])
if "--session-dir" in args:
    args[args.index("--session-dir") + 1] = sys.argv[3]
else:
    _env = cfg.setdefault("env", {})            # (GLM/DS4/ik2 name it in the env: *_SLOT_DIR)
    _slot = [k for k in _env if k.endswith("_SLOT_DIR")]
    for _k in _slot:
        _env[_k] = sys.argv[3]
    if not _slot:
        args += ["--session-dir", sys.argv[3]]
cfg["args"] = args
cfg.pop("api_key", None)   # (a test copy: no key, the readiness poll and the requests send none)
cfg["log"] = sys.argv[4]   # the engine's stderr lands in the out folder, not in the install's own log
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY

for _ in $(seq 1 60); do             # start only on a free port: a held one means the old engine is still in there
    [ -z "$(port_pids)" ] && break
    sleep 1
done
[ -z "$(port_pids)" ] || { echo "port $PORT is still held by $(port_pids | tr '\n' ' ') - stop the server on it first"; exit 1; }

# ---- start the server --------------------------------------------------------------------------------------
export SAGUARO_PRIORITY_IDLE_S="$IDLE"
nohup "$PYTHON" -m serve.server --config "$CFG_T" --port "$PORT" >"$SERVE_LOG" 2>&1 &
echo $! > "$DIR/serve.pid"
trap cleanup EXIT
echo "server pid $(cat "$DIR/serve.pid") on :$PORT, log $SERVE_LOG (loading the model ...)"
for _ in $(seq 1 240); do
    curl -fs -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$(cat "$DIR/serve.pid")" 2>/dev/null || fail "the server died while loading; see $SERVE_LOG"
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
bytes_now() { wc -c < "$1" 2>/dev/null | tr -d ' '; }
lines_from() { tail -c +"$(( $2 + 1 ))" "$1"; }
serve_lines() { lines_from "$SERVE_LOG" "$1"; }      # the server's own lines: [saguaro] progress, pause, continue
engine_lines() { lines_from "$ELOG" "$1"; }          # the engine's stderr: its "reused + N read in M ms" figures
maxpos() { serve_lines "$1" | tr -d ',' | grep -oE "reading the prompt: [0-9]+ of" | awk '{print $4}' | sort -n | tail -1; }
count() { serve_lines "$1" | grep -cF "$2" || true; }
read_ms() {   # the engine's own read milliseconds, summed over the legs logged since a byte offset
    engine_lines "$1" | grep -oE "reused \+ [0-9]+( of [0-9]+)? read in [0-9]+ ms" \
        | awk '{s += $(NF-1)} END {printf "%.0f", s + 0}'
}

# ---- phase L: the fresh low read, cut three times -----------------------------------------------------------
echo "== phase L: a fresh ${TOKENS}-token-ish low read, ${CUT_N} normal interruptions =="
MARK_L=$(bytes_now "$SERVE_LOG")
MARK_EL=$(bytes_now "$ELOG")
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
mapfile -t REUSED < <(serve_lines "$MARK_L" \
    | grep -oE "prompt read continued: [0-9,]+ of [0-9,]+ tokens reused" \
    | sed -E 's/^prompt read continued: ([0-9,]+) of .*$/\1/' | tr -d ',')
[ "${#REUSED[@]}" -eq "$CUT_N" ] || fail "continue lines without figures: ${REUSED[*]:-none}"
prev=0
for r in "${REUSED[@]}"; do
    [ "$r" -gt 0 ] || fail "a resumed read reported 0 reused - the P124b loss (figures: ${REUSED[*]})"
    [ "$r" -ge "$prev" ] || fail "the reuse did not grow across the cuts: ${REUSED[*]}"
    prev=$r
done
READ_L=$(read_ms "$MARK_EL")
WALL_L=$(cat "$DIR/wall_L")
echo "   reused per resume: ${REUSED[*]}   engine read ms (legs summed): $READ_L   wall: ${WALL_L}s"

# ---- phase C: a fresh prompt of the same size, uninterrupted -------------------------------------------------
echo "== phase C: the control read (same size, no interruption) =="
MARK_C=$(bytes_now "$SERVE_LOG")
MARK_EC=$(bytes_now "$ELOG")
ask "$DIR/prompt_C.txt" C low || fail "the control request failed (see $DIR/resp_C.json)"
grep -qE 'finish' "$DIR/resp_C.json" 2>/dev/null || fail "the control did not answer"
[ "$(count "$MARK_C" 'prompt read paused')" -eq 0 ] || fail "the control was cut too (a request leaked from L?)"
READ_C=$(read_ms "$MARK_EC")
WALL_C=$(cat "$DIR/wall_C")
[ "$READ_C" -gt 0 ] || fail "no engine read figures for the control"
echo "   engine read ms: $READ_C   wall: ${WALL_C}s"

# ---- the verdict ---------------------------------------------------------------------------------------------
cleanup
ok=$(awk -v a="$READ_L" -v c="$READ_C" -v r="$RATIO" 'BEGIN { print (a <= c * r + 15000) ? "yes" : "no" }')
echo "interrupted read ${READ_L} ms vs control ${READ_C} ms (ratio $(awk -v a="$READ_L" -v c="$READ_C" \
    'BEGIN { printf "%.2f", a / c }')), resumes with M = ${REUSED[*]}"
[ "$ok" = yes ] || fail "the interrupted read cost ${READ_L} ms against a ${READ_C} ms control: the lost-read shape"
echo "PASS: P124b - each paused low read resumed with M > 0 and the total cost one read, not two"
