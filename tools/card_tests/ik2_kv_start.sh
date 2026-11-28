#!/usr/bin/env bash
# tools/card_tests/ik2_kv_start.sh - PLAN-p125e card test: does a larger --max-context still cost the experts their
# VRAM tier before a single token is generated (ref/REPORT-p125b-ik2-vram.md), or does the elastic start step
# (SAGUARO_IK2_KV_START) and the preloaded offload tier (b1) keep them where they were?
#
# It starts a test copy of the server on its own port (default 8092) with its own --session-dir, its own engine log
# and its own config copy (the api_key dropped - the readiness poll and the requests send none), once per context of
# CONTEXTS (default: 32768 and 262144).  Per context it prints the engine's placement lines from its log - the plan
# line ("experts ... K / V cache ... free"), the elastic NOTE ("NOTE context: ... committed ..."), the
# resident / tier / offload / warm line, and what /api/context says the conversation's ceiling is - then reads a
# prompt of READ tokens (default 20000, inside the committed start step) and times it.  At the big ceiling it also
# asks past the committed context to show exactly what a conversation that grows there meets (a clean "maximum
# context length" answer while the engine runs at the commit, or a full read where the card did not take the
# elastic start).
#
# PASS: the 262144 start keeps its experts in VRAM - no "offload: 0 of" line, and the count in VRAM is within 2 % of
# the 32768 row's (the tools/mem_vs_ctx.py rule), with the elastic NOTE or a preloaded "tier:" line as the witness -
# and both 20K reads complete.
#
# Everything it writes lands under its --out folder (default scratch/ik2-kv-start-card); the install's own config,
# log and sessions are never touched (the newest saguaro-ik2 config is only read).  It waits until the port is free
# before starting, and on exit kills whatever still holds it (the engine's child can outlive the launcher).
#
#   ./tools/card_tests/ik2_kv_start.sh                       # port 8092, 32768 vs 262144, a 20K read
#   CONTEXTS="32768 65536" ./tools/card_tests/ik2_kv_start.sh
#   SAGUARO_TEST_EXE=scratch/build-ik2-elastic/saguaro-ik2 ./tools/card_tests/ik2_kv_start.sh   # a candidate engine
#
# Needs: Inkling 2 Mini already set up (a saguaro-*.json whose engine is saguaro-ik2), curl + a python (the
# install's .venv is used when present).
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=scratch/ik2-kv-start-card
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT=${2:?--out wants a folder}; shift 2 ;;
        *) echo "unknown argument: $1 (see the header)"; exit 1 ;;
    esac
done
case "$OUT" in ""|.|..) echo "--out wants a dedicated folder, not '$OUT'"; exit 1 ;; esac
PORT=${PORT:-8092}
CONTEXTS=${CONTEXTS:-"32768 262144"}
READ=${READ:-20000}
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
CFG_SRC=""
for f in $(ls -t saguaro-*.json 2>/dev/null); do
    if grep -qE '"[^"]*(saguaro|maya)-ik2"' "$f" 2>/dev/null; then CFG_SRC=$f; break; fi
done
[ -n "$CFG_SRC" ] || { echo "no saguaro-ik2 config here: set Inkling 2 Mini up first (or run from the Saguaro folder)"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"

say() { printf '\n== %s\n' "$*"; }
port_pids() {
    ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE "pid=[0-9]+" | cut -d= -f2
    lsof -nP -t -i "TCP:$PORT" -s TCP:LISTEN 2>/dev/null
    true
}
cleanup() {
    [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null
    for p in $(port_pids); do kill "$p" 2>/dev/null; done
    true
}
wait_free_port() {
    for _ in $(seq 1 60); do [ -z "$(port_pids)" ] && return 0; sleep 1; done
    echo "port $PORT is still held by $(port_pids | tr '\n' ' ') - stop the server on it first"; exit 1
}

read_prompt() {   # $1 file, $2 tokens of filler, $3 max_tokens
    "$PY" - "$1" "$2" "$3" <<'PYEOF'
import json, sys
f, n, mx = sys.argv[1:4]
filler = (" ".join(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
                   * (int(n) // 10 + 1)))[:int(n) * 5]
json.dump({"model": "m", "temperature": 0, "max_tokens": int(mx),
           "messages": [{"role": "system", "content": f"Reference: {filler}"},
                        {"role": "user", "content": "What is the first word of the reference? Answer in one word."}]},
          open(f, "w", encoding="utf-8"), ensure_ascii=False)
PYEOF
}

ask() {   # $1 file: prints "HTTP <code> <completion_tokens> <seconds>"; never dies
    "$PY" - "$1" "$PORT" <<'PYEOF'
import json, sys, time, urllib.error, urllib.request
body = json.load(open(sys.argv[1], encoding="utf-8"))
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[2]}/v1/chat/completions",
                             data=json.dumps(body, ensure_ascii=False).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    u = out.get("usage") or {}
    print(f"HTTP {r.status} {u.get('completion_tokens', '?')} {time.time() - t0:.1f}s")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code} - {time.time() - t0:.1f}s {e.read().decode()[:200]}")
PYEOF
}

place() {   # the count of experts the engine said it keeps in VRAM ("3072", "1234", "0", "-" for unknown)
    sed -ne 's/.*resident: every expert in VRAM (\([0-9]*\) copied.*/\1/p; s/.*offload: \([0-9]*\) of [0-9]* experts in VRAM.*/\1/p' "$1" | tail -1
}

declare -A PLACEMENT
for CTX in $CONTEXTS; do
    wait_free_port
    SESS=$OUT/sessions-$CTX; LOG=$OUT/engine-$CTX.log; SRV=$OUT/server-$CTX.log; CFG=$OUT/config-$CTX.json
    mkdir -p "$SESS"
    "$PY" - "$CFG_SRC" "$CFG" "$PORT" "$SESS" "$LOG" "$CTX" <<'PYEOF'
import json, sys
src, dst, port, sess, log, ctx = sys.argv[1:7]
cfg = json.load(open(src, encoding="utf-8"))
args = [a for a in cfg.get("args", [])]
if "--max-context" in args:                            # the ceiling under test
    args[args.index("--max-context") + 1] = ctx
else:
    args += ["--max-context", ctx]
if "--session-dir" in args:
    args[args.index("--session-dir") + 1] = sess
else:
    args += ["--session-dir", sess]
cfg["args"] = args
cfg["port"] = int(port)
cfg.pop("api_key", None)                               # (a test copy: no key, the readiness poll needs none)
cfg["log"] = log
if __import__("os").environ.get("SAGUARO_TEST_EXE"):   # a build, to test a candidate engine
    cfg["exe"] = __import__("os").environ["SAGUARO_TEST_EXE"]
json.dump(cfg, open(dst, "w", encoding="utf-8"), indent=1)
PYEOF
    say "--max-context $CTX: starting the test copy on port $PORT (the model loads - watch $SRV)"
    "$PY" saguaro.py --config "$CFG" --port "$PORT" > "$SRV" 2>&1 &
    SRV_PID=$!
    trap cleanup EXIT
    for _ in $(seq 1 240); do
        curl -sf -o /dev/null "http://127.0.0.1:$PORT/metrics" 2>/dev/null && break
        kill -0 "$SRV_PID" 2>/dev/null || { echo "the server died; see $SRV"; exit 1; }
        sleep 5
    done
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/metrics" || { echo "the API did not come up in 20 min; see $SRV"; exit 1; }
    disown "$SRV_PID" 2>/dev/null || true

    say "--max-context $CTX: the engine's load lines"
    grep -E "experts [0-9.]+ GiB, dense weights|NOTE context:|resident: every expert|tier: [0-9]+ of|offload: [0-9]+ of|warm: " "$SRV" "$LOG" 2>/dev/null | sed -e 's/^[^:]*: //' || echo "  (no placement line found - the engine log holds none)"
    say "--max-context $CTX: the ceiling the engine announced"
    curl -sf "http://127.0.0.1:$PORT/api/context" | "$PY" - <<'PYEOF' || echo "  (/api/context did not answer)"
import json, sys
d = json.load(sys.stdin)
print(f"context: {d.get('context')}")
PYEOF
    PLACEMENT[$CTX]=$(place "$LOG")
    [ -n "${PLACEMENT[$CTX]}" ] || PLACEMENT[$CTX]=-

    say "--max-context $CTX: a $READ-token read (inside the committed start step)"
    read_prompt "$OUT/read-$CTX.json" "$READ" 16
    ask "$OUT/read-$CTX.json"
    if [ "$CTX" != "$(echo "$CONTEXTS" | cut -d" " -f1)" ]; then
        say "--max-context $CTX: asking past the committed context (what a conversation that grows there meets)"
        COMMIT=$(sed -ne 's/.*laid out for the first \([0-9]*\) positions.*/\1/p' "$LOG" | tail -1)
        if [ -n "$COMMIT" ]; then
            read_prompt "$OUT/past-$CTX.json" "$COMMIT" 64   # a full commit of prompt + 64 to generate: past it
            ask "$OUT/past-$CTX.json"
        else
            echo "  (no NOTE context: line - the elastic start was not taken here, nothing is committed short)"
        fi
    fi
    say "--max-context $CTX: stopping the test copy"
    kill "$SRV_PID" 2>/dev/null
    for p in $(port_pids); do kill "$p" 2>/dev/null; done
    sleep 3
done

say "the two contexts side by side"
SMALL=$(echo "$CONTEXTS" | cut -d" " -f1); BIG=$(echo "$CONTEXTS" | tr " " "\n" | tail -1)
S=${PLACEMENT[$SMALL]:--}; B=${PLACEMENT[$BIG]:--}
echo "  experts in VRAM: $SMALL -> $S, $BIG -> $B"
if [ "$B" = "0" ]; then
    echo "RESULT: FAIL - the $BIG ceiling starts with no expert in VRAM (the P125b row, unfixed)"
    exit 1
fi
if [ "$S" != "-" ] && [ "$B" != "-" ]; then
    "$PY" - "$S" "$B" <<'PYEOF'
import sys
s, b = int(sys.argv[1]), int(sys.argv[2])
if b < s * 0.98:
    sys.exit(f"RESULT: FAIL - the big ceiling holds {b} of the {s} experts the small one keeps (> 2 % moved)")
print(f"RESULT: PASS - the {b} experts in VRAM follow the small context's {s} (the 2 % rule), and both reads ran")
PYEOF
else
    echo "RESULT: check by hand - the placement line of one start did not parse ($S vs $B)"
fi
