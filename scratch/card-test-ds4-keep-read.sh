#!/usr/bin/env bash
# scratch/card-test-ds4-keep-read.sh - P124k on the card: a ~30K DeepSeek-V4.1 prompt read STOPped mid-way (client
# disconnect: the server then writes STOP to the engine and drains to its DONE) must keep what it read.  The same
# request asked again PASSes when its DONE shows reused > 0.
#
#   CFG=/path/to/<deepseek>.json scratch/card-test-ds4-keep-read.sh
#
# What it does: a test copy of the card's DeepSeek config on port 8092 with its OWN session/slot dirs and its OWN
# engine log, and NO api key (nothing of the live box's files is written, nothing of it is asked to authenticate), then:
#   phase C: a fresh ~30K prompt read, uninterrupted - the control, to learn this card's read ms for the cut timing;
#   phase L: a SECOND fresh ~30K prompt, streamed; after FRAC of the control's read ms the curl is killed (the client
#            goes away): the server's drain ends in the engine's "stopped while reading the prompt" line.  Ask the
#            same request again: the engine's own DONE must say "= K reused" with 0 < K < N - the read resumed on what
#            the stopped read kept (P124k), instead of re-reading the whole prompt from 0 (the pre-P124k shape).
# A cut that lands before the first keep point keeps nothing (K = 0, honest): the script retries the L phase later
# (FRACS, default .50 .65 .80 of the control read) and PASSes on the first attempt that reports a cut.
# SAGUARO_DS4_KEEP_READ=0 on the engine (SAGUARO_DS4_KEEP_READ exported here before the server starts) says READY
# without keepread and drops the read - the same phases then show K = 0: the gate's truth, for A/B.
# One model load, three reads through the real server; nothing destructive.
set -u -o pipefail

PYTHON=${PYTHON:-/opt/llm/saguaro/.venv/bin/python}
PORT=${PORT:-8092}
CFG=${CFG:?the card DeepSeek model config JSON (the one the live server uses)}
TOKENS=${TOKENS:-30000}
FRACS=${FRACS:-".50 .65 .80"}
GRACE=${GRACE:-10}                    # s between the POST and the read (admission, parking the control's conversation)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$HERE")
DIR="$HERE/ds4-keep-read"
LOG="$DIR/serve.log"                   # the server's own log
ELOG="$DIR/ds4.log"                    # the engine's stderr (the config's "log", retargeted): the lines we read
mkdir -p "$DIR" "$DIR/sessions"
rm -f "$LOG" "$ELOG" "$DIR"/resp_*.json "$DIR"/wall_* "$DIR"/body_*.json "$DIR"/prompt_*.txt

fail() { echo "FAIL: $*" >&2
         [ -f "$DIR/serve.pid" ] && kill "$(cat "$DIR/serve.pid")" 2>/dev/null
         echo "--- last of $LOG ---"; tail -30 "$LOG" 2>/dev/null
         echo "--- last of $ELOG ---"; tail -30 "$ELOG" 2>/dev/null; exit 1; }

# ---- the test copy of the config: same engine, own dirs, own engine log, no api key ---------------------------
CFG_T="$DIR/ds4-keep-read.json"
"$PYTHON" - "$CFG" "$CFG_T" "$DIR/sessions" "$ELOG" <<'PY' || fail "could not write the test config"
import json, sys
cfg = json.load(open(sys.argv[1]))
args = list(cfg.get("args") or [])
if "--session-dir" in args:
    args[args.index("--session-dir") + 1] = sys.argv[3]
else:
    args += ["--session-dir", sys.argv[3]]
cfg["args"] = args
env = dict(cfg.get("env") or {})
for k in list(env):                    # the engine's parked-conversation folder: its own, not the live box's
    if k.endswith("SLOT_DIR"):
        env[k] = sys.argv[3]
cfg["env"] = env
cfg["log"] = sys.argv[4]               # the engine's stderr where the test can read it
cfg.pop("api_key", None)               # no api key: nothing to authenticate with on this port
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY

# ---- start the server --------------------------------------------------------------------------------------
cd "$ROOT_DIR" || exit 1
nohup "$PYTHON" -m serve.server --config "$CFG_T" --port "$PORT" >"$LOG" 2>&1 &
echo $! > "$DIR/serve.pid"
echo "server pid $(cat "$DIR/serve.pid") on :$PORT, log $LOG, engine log $ELOG (loading the model ...)"
for _ in $(seq 1 240); do
    curl -fs -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$(cat "$DIR/serve.pid")" 2>/dev/null || fail "the server died while loading; see $LOG"
    sleep 5
done
curl -fs -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || fail "no /health after 20 min"
grep -q "READY.*keepread" "$ELOG" 2>/dev/null || echo "NOTE: the engine's READY line does not say keepread:"
grep -q "READY.*keepread" "$ELOG" 2>/dev/null || echo "      SAGUARO_DS4_KEEP_READ=0 on the engine? the keep is off and this test cannot pass"
MODEL=$("$PYTHON" - "$PORT" <<'PY'
import json, sys, urllib.request
print(json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models"))["data"][0]["id"])
PY
) || fail "no model id on :$PORT"
echo "model: $MODEL"

# ---- prompts: fresh random words, ~4 chars a token, a different seed per phase ------------------------------
prompt() { "$PYTHON" - "$1" "$2" <<'PY'
import random, sys
rng = random.Random(f"p124k-{sys.argv[2]}")
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu analysis bucket canyon engine floor "
         "gamma hinge input junction latent metric needle oracle pixel query radio sector token unity vector "
         "window xenon yield zenith").split()
out, n = [], 0
while n < int(sys.argv[1]):
    w = rng.choice(words)
    out.append(w)
    n += len(w) + 1
print(f"P124k keep-read probe {sys.argv[2]}: " + " ".join(out))
PY
}
prompt "$((TOKENS * 4))" C > "$DIR/prompt_C.txt" || fail "prompt C"
prompt "$((TOKENS * 4))" L > "$DIR/prompt_L.txt" || fail "prompt L"

body() {   # body <prompt-file> <out>: one streaming request body
    "$PYTHON" - "$1" "$MODEL" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[2], "stream": True, "temperature": 0, "max_tokens": 32,
                  "messages": [{"role": "user", "content": open(sys.argv[1]).read()}]}))
PY
}
bytes_now() { wc -c < "$ELOG" | tr -d ' '; }
lines_from() { tail -c +"$(( $1 + 1 ))" "$ELOG"; }

# ---- phase C: the control read (fresh prompt, uninterrupted): this card's read ms ----------------------------
echo "== phase C: a fresh ${TOKENS}-token-ish read, uninterrupted (the cut-timing control) =="
MARK_C=$(bytes_now)
body "$DIR/prompt_C.txt" > "$DIR/body_C.json" || fail "body C"
curl -s -N -o "$DIR/resp_C.json" -w "%{time_total}" -m 7200 -X POST \
    "http://127.0.0.1:$PORT/v1/chat/completions" -H "content-type: application/json" \
    --data-binary "@$DIR/body_C.json" > "$DIR/wall_C" || fail "the control request failed"
CTRL_MS=$(lines_from "$MARK_C" | grep -oE "prompt [0-9]+ tokens = 0 reused \+ [0-9]+ read in [0-9]+ ms" \
          | sed -E 's/.* read in ([0-9]+) ms$/\1/' | tail -1)
[ -n "$CTRL_MS" ] || fail "no engine read figure for the control (engine log?)"
echo "   control read: $CTRL_MS ms"

# ---- phase L: a fresh prompt, cut mid-read by a client disconnect, then asked again --------------------------
for frac in $FRACS; do
    kill_ms=$("$PYTHON" -c "print(int($CTRL_MS * $frac))")
    echo "== phase L (cut at ${frac} of the control read, ~${kill_ms} ms): stream, then go away"
    MARK_L=$(bytes_now)
    body "$DIR/prompt_L.txt" > "$DIR/body_L.json" || fail "body L"
    curl -s -N -o /dev/null -m 7200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "content-type: application/json" --data-binary "@$DIR/body_L.json" &
    CURL_PID=$!
    sleep "$GRACE"
    left=$((kill_ms - GRACE * 1000))
    [ "$left" -gt 0 ] && sleep $((left / 1000))
    if ! kill -0 "$CURL_PID" 2>/dev/null; then
        kill "$(cat "$DIR/serve.pid")" 2>/dev/null
        fail "the ${TOKENS}-token read finished before the earliest cut - too fast to cut mid-read: raise TOKENS"
    fi
    kill -TERM "$CURL_PID" 2>/dev/null
    wait "$CURL_PID" 2>/dev/null
    for _ in $(seq 1 300); do          # the drain: the engine sees STOP and ends the read
        lines_from "$MARK_L" | grep -q "stopped while reading the prompt" && break
        kill -0 "$(cat "$DIR/serve.pid")" 2>/dev/null || fail "the server died during the cut; see $LOG"
        sleep 1
    done
    if ! lines_from "$MARK_L" | grep -q "stopped while reading the prompt"; then
        echo "   no 'stopped while reading' line by the cut point: the read was over already, or the cut missed"
        continue
    fi
    echo "   the engine stopped the read mid-way; the same request again:"
    sleep 2                            # (let the drained request fully close before its resend)
    MARK_R=$(bytes_now)
    curl -s -N -o "$DIR/resp_L.json" -m 7200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "content-type: application/json" --data-binary "@$DIR/body_L.json" || fail "the resumed request failed"
    RLINE=$(lines_from "$MARK_R" | grep -oE "prompt [0-9]+ tokens = [0-9]+ reused" | tail -1)
    [ -n "$RLINE" ] || fail "no DONE figures for the resumed request (engine log?)"
    N=$(echo "$RLINE" | sed -E 's/^prompt ([0-9]+) tokens = .*/\1/')
    K=$(echo "$RLINE" | sed -E 's/.* = ([0-9]+) reused$/\1/')
    KILLED=$(lines_from "$MARK_L" | grep -oE "kept live at [0-9]+ tokens" | sed -E 's/[^0-9]+//g' | tail -1)
    echo "   stopped read kept: ${KILLED:-0} tokens;  resumed DONE: $K reused of $N"
    [ "$K" -gt 0 ] || { echo "   the resumed read reused 0: the cut fell before the first keep point"; continue; }
    [ "$K" -lt "$N" ] || fail "the resumed read reports the whole prompt reused ($K of $N): it was not cut mid-read"
    kill "$(cat "$DIR/serve.pid")" 2>/dev/null
    echo "PASS: P124k - a ${N}-token read cut mid-way kept its conversation; the same request's DONE says $K reused > 0"
    exit 0
done
kill "$(cat "$DIR/serve.pid")" 2>/dev/null
fail "no cut attempt reported reused > 0 (keep off? read too fast for the cut? SAGUARO_DS4_KEEP_READ_STEP?)"
