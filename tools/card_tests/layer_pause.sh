#!/usr/bin/env bash
# tools/card_tests/layer_pause.sh - P125j on the card: a STOP that catches a low prompt read mid-chunk
# pauses it between two LAYERS, so the normal request that cut it waits one layer pass, not a chunk - and
# the cut read still resumes with the engine's own reuse, reads every token of the prompt exactly once,
# and answers exactly what an uninterrupted read of the same prompt answers.
#
#   CFG=/path/to/<model>.json ./tools/card_tests/layer_pause.sh
#   ./tools/card_tests/layer_pause.sh --out scratch/layer-pause-2   # write under a different folder
#
# What it does: a test copy of the card model config on port 8092 with its OWN session dir, its OWN engine
# log and the api_key dropped (the readiness poll and the requests send none) - everything it writes lands
# under its --out folder (default scratch/layer-pause) and nothing of the live box's config, log or
# sessions is touched (its config is only read).  SAGUARO_LAYER_PAUSE=1 and SAGUARO_PRIORITY_IDLE_S=0
# (P125j's defaults) are exported for the server it starts.  Then:
#   phase L: a fresh ~60K-token prompt asked at "low" priority (temperature 0, 64 tokens); at CUTS random
#            read positions a short NORMAL request runs through and cuts the read.  Per cut: the engine must
#            log "the read paused at chunk" (a between-LAYERS pause, not the old chunk-end stop), the server
#            line must name it ("inside chunk c at layer l"), the normal request's first token must land
#            within FIRST_TOKEN_S seconds (1 by default - the whole point of P125j), and the low read must
#            resume with the engine's own "prompt read continued: M of N tokens reused", M > 0.  This holds on
#            the BATCH lane too (P125m): with --batch the read runs on the engine's own reader thread ("read N
#            tokens on a thread of its own ... concurrent") and must pause there just the same - a config
#            printing that line while no 'prompt read paused' line ever comes is the rc124-cand3 card FAIL.
#            Over the
#            whole phase the engine's own per-leg figures must add up: the legs' read counts sum to the
#            prompt (nothing read twice) and the last leg ends at reused + read = the prompt.
#   phase C: the SAME prompt asked uninterrupted on a fresh engine (the server restarts with an empty
#            session dir): the control answer.
# PASS: every cut resumed with M > 0, at least one cut landed between layers, the TTFT of a cut's normal
# request was within FIRST_TOKEN_S, the reads summed to the prompt, and phase L's answer equals phase C's
# (64 tokens, temperature 0 - an uninterrupted read's answer to the same prompt).  No benchmarks, nothing
# destructive.  It waits until the port is free before starting, and on exit kills whatever still holds the
# port (the server's child engine can outlive the launcher).  Needs: curl + a python (the install's .venv).
set -u -o pipefail

OUT=scratch/layer-pause
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
TOKENS=${TOKENS:-60000}
CUTS=${CUTS:-2}                       # random cut moments; each is one short normal request
FIRST_TOKEN_S=${FIRST_TOKEN_S:-1.0}   # the plan's bound: the cut normal request's first token within 1 s
READ_TOL=${READ_TOL:-1024}            # tokens of slack in the "sum of reads = the prompt" check
ANSWER_TOKENS=${ANSWER_TOKENS:-64}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR" || exit 1
DIR=$OUT
SERVE_LOG="$DIR/serve.log"
ELOG="$DIR/engine.log"                # the engine's own stderr, kept inside the out folder (never the install's log)

port_pids() {   # whatever still holds the port (a previous run's engine child outlived its launcher)
    ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE "pid=[0-9]+" | cut -d= -f2
    lsof -nP -t -i "TCP:$PORT" -s TCP:LISTEN 2>/dev/null
    true
}

server_pid() { [ -f "$DIR/serve.pid" ] && cat "$DIR/serve.pid" || true; }

cleanup() {   # on exit the port is ours again: kill the server and whatever still holds it (its engine child).
              # only once WE started one - a failure before that must leave a live server on the port alone
    [ -f "$DIR/serve.pid" ] || return 0
    kill "$(server_pid)" 2>/dev/null
    for p in $(port_pids); do kill "$p" 2>/dev/null; done
    true
}

fail() { echo "FAIL: $*" >&2
         cleanup
         echo "--- last of $SERVE_LOG ---"; tail -40 "$SERVE_LOG" 2>/dev/null
         echo "--- last of $ELOG ---"; tail -20 "$ELOG" 2>/dev/null; exit 1; }

rm -rf "$DIR"; mkdir -p "$DIR" "$DIR/sessions"

# ---- the test copy of the config: same engine, its own session dir, log and no key ---------------------------
CFG_T="$DIR/layer-pause.json"
"$PYTHON" - "$CFG" "$CFG_T" "$DIR/sessions" "$ELOG" <<'PY' || fail "could not write the test config"
import json, sys
cfg = json.load(open(sys.argv[1]))
args = list(cfg.get("args") or [])
if "--session-dir" in args:
    args[args.index("--session-dir") + 1] = sys.argv[3]
else:
    args += ["--session-dir", sys.argv[3]]
cfg["args"] = args
cfg.pop("api_key", None)   # (a test copy: no key, the readiness poll and the requests send none)
cfg["log"] = sys.argv[4]   # the engine's stderr lands in the out folder, not in the install's own log
if __import__("os").environ.get("SAGUARO_TEST_EXE"):   # a build from tools/qfn/build.sh, to test a candidate engine
    cfg["exe"] = __import__("os").environ["SAGUARO_TEST_EXE"]
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY

for _ in $(seq 1 60); do             # start only on a free port: a held one means the old engine is still in there
    [ -z "$(port_pids)" ] && break
    sleep 1
done
[ -z "$(port_pids)" ] || { echo "port $PORT is still held by $(port_pids | tr '\n' ' ') - stop the server on it first"; exit 1; }

# ---- start the server (P125j's own defaults, said out loud) --------------------------------------------------
start_server() {   # (also used for phase C's fresh engine: the same config, an empty session dir)
    SAGUARO_LAYER_PAUSE=1 SAGUARO_PRIORITY_IDLE_S="${IDLE:-0}" nohup "$PYTHON" -m serve.server \
        --config "$CFG_T" --port "$PORT" >>"$SERVE_LOG" 2>&1 &
    echo $! > "$DIR/serve.pid"
}
start_server
trap cleanup EXIT
echo "server pid $(server_pid) on :$PORT, log $SERVE_LOG (loading the model ...)"
wait_ready() {   # $1: the pid the death check runs against
    for _ in $(seq 1 240); do
        curl -fs -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$1" 2>/dev/null || fail "the server died while loading; see $SERVE_LOG"
        sleep 5
    done
    curl -fs -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || fail "no /health after 20 min"
}
wait_ready "$(server_pid)"
MODEL=$("$PYTHON" - "$PORT" <<'PY'
import json, sys, urllib.request
print(json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models"))["data"][0]["id"])
PY
) || fail "no model id on :$PORT"
echo "model: $MODEL"

# ---- prompts: fresh random words, ~4 chars a token -----------------------------------------------------------
prompt() { "$PYTHON" - "$1" "$2" <<'PY'
import random, sys
rng = random.Random(f"p125j-{sys.argv[2]}")
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu analysis bucket canyon engine floor "
         "gamma hinge input junction latent metric needle oracle pixel query radio sector token unity vector "
         "window xenon yield zenith").split()
out, n = [], 0
while n < int(sys.argv[1]):
    w = rng.choice(words)
    out.append(w)
    n += len(w) + 1
print(f"P125j layer-pause probe {sys.argv[2]}: " + " ".join(out))
PY
}
prompt "$((TOKENS * 4))" L > "$DIR/prompt_L.txt" || fail "prompt L"

ask() {   # ask <prompt-file> <tag>: the low request; its answer lands in $DIR/resp_<tag>.json
    "$PYTHON" - "$1" "$MODEL" "$ANSWER_TOKENS" <<'PY' > "$DIR/body_$2.json"
import json, sys
print(json.dumps({"model": sys.argv[2], "priority": "low", "temperature": 0, "max_tokens": int(sys.argv[3]),
                  "messages": [{"role": "user", "content": open(sys.argv[1]).read()}]}))
PY
    curl -s -o "$DIR/resp_$2.json" -m 3600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "content-type: application/json" --data-binary "@$DIR/body_$2.json"
}
short() {   # one short normal request - the one that cuts the low read; prints its time-to-first-token
    "$PYTHON" - "$MODEL" <<'PY' > "$DIR/body_short.json"
import json, sys
print(json.dumps({"model": sys.argv[1], "priority": "normal", "max_tokens": 24, "temperature": 0,
                  "messages": [{"role": "user", "content": "Reply with the word OK."}]}))
PY
    curl -s -o "$DIR/resp_short.json" -m 600 -w "%{time_starttransfer}" -X POST \
        "http://127.0.0.1:$PORT/v1/chat/completions" -H "content-type: application/json" \
        --data-binary "@$DIR/body_short.json"
}
bytes_now() { wc -c < "$1" 2>/dev/null | tr -d ' '; }
lines_from() { tail -c +"$(( $2 + 1 ))" "$1"; }
serve_lines() { lines_from "$SERVE_LOG" "$1"; }      # the server's own lines: [saguaro] progress, pause, continue
engine_lines() { lines_from "$ELOG" "$1"; }          # the engine's stderr: the pause keeps, the reused figures
maxpos() { serve_lines "$1" | tr -d ',' | grep -oE "reading the prompt: [0-9]+ of" | awk '{print $4}' | sort -n | tail -1; }
count() { serve_lines "$1" | grep -cF "$2" || true; }
ecount() { engine_lines "$1" | grep -cF "$2" || true; }

# ---- phase L: the fresh low read, cut at random moments ------------------------------------------------------
echo "== phase L: a fresh ${TOKENS}-token-ish low read, ${CUTS} normal interruptions at random moments =="
MARK_L=$(bytes_now "$SERVE_LOG")
MARK_EL=$(bytes_now "$ELOG")
ask "$DIR/prompt_L.txt" L &
LOW_PID=$!
PAUSE_SEEN=0
TTFT_WORST=0
for i in $(seq 1 "$CUTS"); do
    # a random cut moment between a quarter and three quarters of the (approximate) prompt
    target=$(( TOKENS / 4 + (RANDOM % (TOKENS / 2)) ))
    while :; do
        p=$(maxpos "$MARK_L"); p=${p:-0}
        [ "$p" -ge "$target" ] && break
        kill -0 "$LOW_PID" 2>/dev/null || fail "the low read ended before position $target (cut $i)"
        sleep 1
    done
    echo "-- cut $i at read position ${p} >= ${target} (random): running a short normal request"
    TTFT=$(short) || fail "the short normal request (cut $i) failed"
    TTFT_WORST=$("$PYTHON" -c "print(max($TTFT_WORST, $TTFT))")
    echo "   the normal request's first token: ${TTFT}s"
    for _ in $(seq 1 180); do
        [ "$(count "$MARK_L" 'prompt read paused')" -ge "$i" ] && break
        sleep 1
    done
    [ "$(count "$MARK_L" 'prompt read paused')" -ge "$i" ] || fail "no 'prompt read paused' line for cut $i"
    for _ in $(seq 1 600); do        # grace 0 now: the resume is immediate, the line then carries M
        [ "$(count "$MARK_L" 'prompt read continued')" -ge "$i" ] && break
        sleep 1
    done
    [ "$(count "$MARK_L" 'prompt read continued')" -ge "$i" ] || fail "no 'prompt read continued' line for cut $i"
    if [ "$(ecount "$MARK_EL" 'the read paused at chunk')" -gt "$PAUSE_SEEN" ]; then
        PAUSE_SEEN=$(ecount "$MARK_EL" 'the read paused at chunk')
        echo "   a between-LAYERS pause (engine line 'the read paused at chunk ... layer ...')"
        serve_lines "$MARK_L" | grep -E "prompt read paused at .*inside chunk [0-9-]+ at layer [0-9]+" \
            > "$DIR/paused_lines.txt" || true
        [ -s "$DIR/paused_lines.txt" ] || fail "the engine paused between layers but the server line does not name the layer/chunk"
        echo "   $(tail -1 "$DIR/paused_lines.txt")"
        # the TTFT promise counts for the cuts that actually landed between layers
        ok=$(awk -v a="$TTFT" -v b="$FIRST_TOKEN_S" 'BEGIN { print (a <= b) ? "yes" : "no" }')
        [ "$ok" = yes ] || fail "the normal request that cut a LAYER-paused read waited ${TTFT}s for its first token (> ${FIRST_TOKEN_S}s)"
    else
        echo "   (this cut landed on a chunk end - the plain stop, no layer pause this time)"
    fi
done
wait "$LOW_PID" || fail "the low request failed (see $DIR/resp_L.json): prompt too long for this config?"
grep -qE 'finish' "$DIR/resp_L.json" 2>/dev/null || fail "the low request did not answer"
[ "$PAUSE_SEEN" -ge 1 ] || fail "no cut paused the read between layers (${CUTS} random moments all hit chunk ends? raise CUTS)"
if [ "$(ecount "$MARK_EL" 'on a thread of its own')" -gt 0 ]; then
    echo "   batch lane: the read ran on its own thread (SAGUARO_PIPED_CONCURRENT) and the cuts paused it"
    echo "   there - the leg that silently ran reads to their end until P125m captured the pause on it"
fi
mapfile -t REUSED < <(serve_lines "$MARK_L" \
    | grep -oE "prompt read continued: [0-9,]+ of [0-9,]+ tokens reused" \
    | sed -E 's/^prompt read continued: ([0-9,]+) of .*$/\1/' | tr -d ',')
[ "${#REUSED[@]}" -eq "$CUTS" ] || fail "continue lines without figures: ${REUSED[*]:-none}"
for r in "${REUSED[@]}"; do
    [ "$r" -gt 0 ] || fail "a resumed read reported 0 reused - the pause was dropped (figures: ${REUSED[*]})"
done
# the engine's own per-leg figures (its own lines only: the short requests' small prompts are not legs)
LEG_FIGS=$(engine_lines "$MARK_EL" | grep -oE "prompt [0-9]+ tokens = [0-9]+ reused \+ [0-9]+( of [0-9]+)? read in" \
    | awk '{gsub(/=/,""); if ($2 + 0 >= '"$TOKENS"' / 2) print $2, $4, $7}' || true)
[ -n "$LEG_FIGS" ] || fail "no engine read figures for the low read (the 'prompt N tokens = R reused + K read' lines)"
echo "$LEG_FIGS" | awk '{printf "   leg: prompt %s tokens, %s reused, %s read\n", $1, $2, $3}'
read -r SUM_K LAST_N LAST_R LAST_K <<EOF
$(echo "$LEG_FIGS" | awk '{s += $3; n = $1; r = $2; k = $3} END {print s, n, r, k}')
EOF
over=$(( SUM_K > LAST_N ? SUM_K - LAST_N : LAST_N - SUM_K ))
echo "   legs' reads sum: $SUM_K vs prompt $LAST_N; last leg: $LAST_R reused + $LAST_K read"
[ "$over" -le "$READ_TOL" ] || fail "the legs read $SUM_K tokens against a $LAST_N-token prompt: something was read twice"
[ $(( LAST_R + LAST_K )) -eq "$LAST_N" ] || fail "the last leg does not end at reused + read = the prompt ($LAST_R + $LAST_K != $LAST_N)"
echo "   reused per resume: ${REUSED[*]}"

# ---- phase C: the SAME prompt, uninterrupted, on a fresh engine ----------------------------------------------
# (an uninterrupted read's answer is the check: temperature 0, so the tokens must come out identical. The
# same server with its caches would just reuse the conversation - so: stop it, empty its session dir, start
# again.  Nothing else on the box is touched.)
echo "== phase C: the control - the same prompt, no interruption, fresh engine =="
kill "$(server_pid)" 2>/dev/null
for _ in $(seq 1 60); do [ -z "$(port_pids)" ] && break; sleep 1; done
for p in $(port_pids); do kill "$p" 2>/dev/null; done
sleep 2
rm -rf "$DIR/sessions"; mkdir -p "$DIR/sessions"
: > "$DIR/engine.log"                 # the control's engine lines on their own
MARK_C=$(bytes_now "$SERVE_LOG")
start_server
echo "server pid $(server_pid) again (loading ...)"
wait_ready "$(server_pid)"
ask "$DIR/prompt_L.txt" C || fail "the control request failed (see $DIR/resp_C.json)"
grep -qE 'finish' "$DIR/resp_C.json" 2>/dev/null || fail "the control did not answer"
[ "$(count "$MARK_C" 'prompt read paused')" -eq 0 ] || fail "the control was cut too (a request leaked from L?)"

# ---- the verdict ---------------------------------------------------------------------------------------------
cleanup
"$PYTHON" - "$DIR/resp_L.json" "$DIR/resp_C.json" <<'PY' || fail "the interrupted read did not answer like the uninterrupted one"
import json, sys
ans = lambda f: json.load(open(f, encoding="utf-8"))["choices"][0]["message"]["content"]
a, b = ans(sys.argv[1]), ans(sys.argv[2])
if a == b:
    print(f"answers identical ({len(a)} chars)")
else:
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    sys.exit(f"answers differ at char {i}:\n  cut:  {a[:i+40]!r}\n  plain: {b[:i+40]!r}")
PY
echo "PASS: P125j - the cut read paused between layers (first token of the normal request in ${TTFT_WORST}s," \
     "resumes with M > 0, ${SUM_K} read against a ${LAST_N}-token prompt, answer matches the uninterrupted read)"
