#!/usr/bin/env bash
# tools/card_tests/glm_pinned_start.sh - PLAN-p125g card test: does the GLM engine's pinned latent cache start
# EMPTY at every --max-context, and does a 20K read run at the same speed with it as without it?
#
# The KV-streaming latent cache used to be ONE cudaHostAlloc sized by --max-context: 0 pinned at 32768 (streaming
# off) but 2.95 GB at 262144, whether or not a conversation ever wrote a byte of it. It is now a granule arena:
# address space reserved at the load, a 64-256 MB granule cudaHostRegister'ed only when a position of it is first
# written. So the load must PIN NOTHING at either context - the same pinned start everywhere - and a conversation
# that reads 20K tokens must read them just as fast with the grow-as-you-write cache as the 32768 run without
# streaming has always done.
#
# What it does, once per context in $CTXS (default "32768 262144"):
#   start a test copy of the card's GLM config on its own port with its own session dir, engine log and config
#   copy (api_key dropped - the readiness poll and the requests send none), wait for /health;
#   read the pinned start from the engine log: "pinned latent cache grows with use: N MiB of pinned RAM at start"
#   (no such line = streaming off at that context = 0 pinned, which is the point of the comparison);
#   one short warm-up request, then one fresh ~TOKENS-token prompt (a different seed word per context, so neither
#   run reuses the other's read) and the wall seconds of that read.
# PASS: every context starts at the SAME pinned figure (and that figure is 0 - nothing pinned at the load), and
# the 262144 read takes within RATIO (default 1.30) of the 32768 read's wall time.
#
# Everything it writes lands under its --out folder (default scratch/glm-pinned-start); the install's own config,
# log and sessions are never touched (the config named by $CFG is only read). It waits until the port is free
# before each start, and kills whatever still holds it between phases and on exit (the server's child engine can
# outlive the launcher).
#
#   CFG=/path/to/saguaro-<glm-model>.json ./tools/card_tests/glm_pinned_start.sh
#   CFG=... SAGUARO_TEST_EXE=scratch/build-p125g/saguaro-glm ./tools/card_tests/glm_pinned_start.sh
#   CFG=... CTXS="32768 262144" TOKENS=20000 PORT=8092 --out scratch/glm-pin-2  # a different folder
#
# Needs: the GLM model set up (its config JSON), curl + a python (the install's .venv when present). Two model
# loads and two 20K reads; nothing destructive, no benchmarks.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=scratch/glm-pinned-start
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT=${2:?--out wants a folder}; shift 2 ;;
        *) echo "unknown argument: $1 (see the header)"; exit 1 ;;
    esac
done
case "$OUT" in ""|.|..) echo "--out wants a dedicated folder, not '$OUT'"; exit 1 ;; esac
PORT=${PORT:-8092}
CTXS=${CTXS:-"32768 262144"}
TOKENS=${TOKENS:-20000}
RATIO=${RATIO:-1.30}
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
CFG=${CFG:?the GLM card config JSON (the one the live server uses)}
DIR=$OUT
SRV_LOG=$DIR/serve.log
rm -rf "$DIR"; mkdir -p "$DIR"

port_pids() {   # whatever still holds the port (a previous run's engine child outlived its launcher)
    ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE "pid=[0-9]+" | cut -d= -f2
    lsof -nP -t -i "TCP:$PORT" -s TCP:LISTEN 2>/dev/null
    true
}
SRV_PID=""
cleanup() {
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    for p in $(port_pids); do kill "$p" 2>/dev/null; done
    true
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2
         cleanup
         echo "--- last of $SRV_LOG ---"; tail -40 "$SRV_LOG" 2>/dev/null
         for f in "$DIR"/engine_*.log; do echo "--- last of $f ---"; tail -30 "$f" 2>/dev/null || true; done
         exit 1; }

# ---- prompts: fresh random words, ~4 chars a token, a different seed per context ----------------------------
prompt() { "$PYTHON" - "$1" "$2" <<'PY'
import random, sys
rng = random.Random(f"p125g-{sys.argv[2]}")
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu analysis bucket canyon engine floor "
         "gamma hinge input junction latent metric needle oracle pixel query radio sector token unity vector "
         "window xenon yield zenith").split()
out, n = [], 0
while n < int(sys.argv[1]):
    w = rng.choice(words)
    out.append(w)
    n += len(w) + 1
print(f"P125g pinned-start probe {sys.argv[2]}: " + " ".join(out))
PY
}

phase() {   # phase <ctx>: starts the test copy at that context, takes the pinned start and the 20K read, stops it
    local CTX=$1 ELOG=$DIR/engine_$1.log CFG_T=$DIR/config_$1.json
    echo
    echo "== phase ctx=$CTX (log $ELOG) =="
    "$PYTHON" - "$CFG" "$CFG_T" "$CTX" "$DIR/sessions_$CTX" "$ELOG" <<'PY' || fail "could not write the test config for $CTX"
import json, os, sys
src, dst, ctx, sess, log = sys.argv[1:6]
cfg = json.load(open(src, encoding="utf-8"))
args = [a for a in cfg.get("args") or []]
if "--max-context" in args:                 # the phase's context, over whatever the card's config carries
    args[args.index("--max-context") + 1] = ctx
else:
    args += ["--max-context", ctx]
if "--session-dir" in args:
    args[args.index("--session-dir") + 1] = sess
else:
    args += ["--session-dir", sess]
cfg["args"] = args
cfg.pop("api_key", None)                    # (a test copy: no key, the readiness poll and the requests send none)
cfg["log"] = log                            # the engine's stderr: the pinned-cache line is read from here
if os.environ.get("SAGUARO_TEST_EXE"):      # a scratch/ build of the engine, to test a candidate binary
    cfg["exe"] = os.environ["SAGUARO_TEST_EXE"]
json.dump(cfg, open(dst, "w", encoding="utf-8"), indent=1)
PY
    for _ in $(seq 1 60); do                # start only on a free port: a held one means the old engine is in there
        [ -z "$(port_pids)" ] && break
        sleep 1
    done
    [ -z "$(port_pids)" ] || fail "port $PORT is still held by $(port_pids | tr '\n' ' ') - stop the server on it first"
    nohup "$PYTHON" -m serve.server --config "$CFG_T" --port "$PORT" >>"$SRV_LOG" 2>&1 &
    SRV_PID=$!
    echo "server pid $SRV_PID on :$PORT at ctx $CTX (loading the model ...)"
    for _ in $(seq 1 240); do
        curl -fs -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        kill -0 "$SRV_PID" 2>/dev/null || fail "the server died while loading at ctx $CTX; see $SRV_LOG"
        sleep 5
    done
    curl -fs -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || fail "no /health after 20 min at ctx $CTX"
    MODEL=$("$PYTHON" - "$PORT" <<'PY'
import json, sys, urllib.request
print(json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models"))["data"][0]["id"])
PY
) || fail "no model id on :$PORT"
    # the pinned start, from the engine's own line (absent = streaming off = 0 MiB pinned)
    PIN=$(grep -oE "pinned latent cache grows with use: [0-9]+ MiB of pinned RAM at start" "$ELOG" 2>/dev/null \
          | grep -oE "[0-9]+ MiB" | grep -oE "[0-9]+" | tail -1)
    PIN=${PIN:-0}
    echo "ctx $CTX: pinned latent cache at the load: $PIN MiB"
    grep -oE "pinned latent cache grows with use: .*" "$ELOG" 2>/dev/null | tail -1 || echo "  (no KV streaming line: streaming off at this context)"
    # warm-up: one short request (the first of a fresh engine pays the kernel graph builds)
    printf '{"model": "%s", "max_tokens": 8, "temperature": 0, "messages": [{"role": "user", "content": "Reply with the word OK."}]}' "$MODEL" > "$DIR/body_warm_$CTX.json"
    curl -s -o /dev/null -m 900 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "content-type: application/json" --data-binary "@$DIR/body_warm_$CTX.json" || fail "warm-up at ctx $CTX"
    # the read: fresh ~TOKENS tokens, max_tokens 40, wall seconds in $DIR/wall_$CTX
    prompt "$((TOKENS * 4))" "ctx$CTX" > "$DIR/prompt_$CTX.txt" || fail "prompt at ctx $CTX"
    "$PYTHON" - "$DIR/prompt_$CTX.txt" "$MODEL" > "$DIR/body_$CTX.json" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[2], "temperature": 0, "max_tokens": 40,
                  "messages": [{"role": "user", "content": open(sys.argv[1], encoding="utf-8").read()}]}))
PY
    curl -s -o "$DIR/resp_$CTX.json" -w "%{time_total}" -m 3600 -X POST \
        "http://127.0.0.1:$PORT/v1/chat/completions" -H "content-type: application/json" \
        --data-binary "@$DIR/body_$CTX.json" > "$DIR/wall_$CTX" || fail "the ${TOKENS}-token read at ctx $CTX"
    echo "ctx $CTX: ${TOKENS}-token read took $(cat "$DIR/wall_$CTX") s wall"
    # and what the cache holds pinned now the conversation has written it (the grown figure, for the report):
    # the re-printed VRAM lines' kv_ram / the INFO kv_ram_bytes, wherever the two logs carry them
    grep -hoE "kv_ram[ _]bytes?=[0-9]+|kv_ram=[0-9]+" "$ELOG" "$SRV_LOG" 2>/dev/null | tail -2 || true
    echo "stopping the test copy (ctx $CTX)"
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
    for _ in $(seq 1 60); do
        [ -z "$(port_pids)" ] && break
        sleep 1
    done
    [ -z "$(port_pids)" ] || fail "port $PORT stayed held after ctx $CTX - an engine child did not go"
    echo "$PIN" > "$DIR/pin_$CTX"
}

for CTX in $CTXS; do
    phase "$CTX"
done

# ---- the verdict ----------------------------------------------------------------------------------------------
FIRST=$(echo "$CTXS" | awk '{print $1}')
PIN_BASE=$(cat "$DIR/pin_$FIRST")
WALL_BASE=$(cat "$DIR/wall_$FIRST")
echo
echo "== verdict =="
RC=0
for CTX in $CTXS; do
    PIN=$(cat "$DIR/pin_$CTX")
    [ "$PIN" = "$PIN_BASE" ] || { echo "FAIL: ctx $CTX starts with $PIN MiB pinned, ctx $FIRST with $PIN_BASE - the pinned start is not the same"; RC=1; }
    [ "$PIN" = "0" ] || echo "CHECK: ctx $CTX pinned $PIN MiB at the load (the granule arena should start at 0 - an eager fallback?)"
done
for CTX in $CTXS; do
    [ "$CTX" = "$FIRST" ] && continue
    W=$(awk -v a="$(cat "$DIR/wall_$CTX")" -v b="$WALL_BASE" 'BEGIN {printf "%.2f", (b > 0 ? a / b : 99)}')
    OK=$(awk -v w="$W" -v r="$RATIO" 'BEGIN {print (w <= r) ? 1 : 0}')
    echo "ctx $CTX read vs ctx $FIRST: ${W}x the wall time (floor ratio $RATIO)"
    [ "$OK" = "1" ] || { echo "FAIL: the $CTX read is ${W}x the $FIRST one - the growing cache slowed the read"; RC=1; }
done
if [ "$RC" = "0" ]; then
    echo "PASS: every context loads with the same pinned start ($PIN_BASE MiB) and the ${TOKENS}-token read runs at the same speed everywhere"
else
    fail "see the lines above"
fi
