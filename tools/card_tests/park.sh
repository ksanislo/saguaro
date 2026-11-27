#!/usr/bin/env bash
# tools/card_tests/park.sh - PLAN-p124a card test: does a parked conversation now STAY parked, and what do
# parking and restoring take on the card?
#
# It starts a test copy of the server on its own port (default 8092) with its own --session-dir, its own engine
# log and its own config copy (the api_key dropped - the readiness poll and the requests send none), then for
# conversations of 1K, 30K and 100K tokens, three times each: ask the long conversation A, switch to a
# different short conversation B (that is where A gets parked), wait long enough for one park-pressure tick to
# pass, then ask A again (that is where A is restored - from RAM, or from disk if the RAM was given back).
# After each size it prints every park/restore time from the engine log and whether anything was released for
# pressure.  PASS: parks keep their RAM (no "the host is short of RAM") and restores come from RAM.
#
# Everything it writes lands under its --out folder (default scratch/park-card); the install's own config, log
# and sessions are never touched (the newest saguaro-<model>.json is only read).  It waits until the port is
# free before starting, and on exit kills whatever still holds it (the server's child engine can outlive the
# launcher).
#
#   ./tools/card_tests/park.sh                        # port 8092, sizes 1000 30000 100000, 3 rounds
#   ./tools/card_tests/park.sh --out scratch/park-2   # write under a different folder
#   SIZES="1000 30000" PORT=8093 ./tools/card_tests/park.sh    # a smaller context: drop the 100K case
#   SAGUARO_TEST_EXE=build-hip/qfn/saguaro-qfn ...    # run the built test engine instead of the installed one
#
# Needs: the model already set up (its saguaro-<model>.json in the Saguaro folder), curl + a python (the
# install's .venv is used when present).
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=scratch/park-card
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT=${2:?--out wants a folder}; shift 2 ;;
        *) echo "unknown argument: $1 (see the header)"; exit 1 ;;
    esac
done
case "$OUT" in ""|.|..) echo "--out wants a dedicated folder, not '$OUT'"; exit 1 ;; esac
PORT=${PORT:-8092}
SIZES=${SIZES:-"1000 30000 100000"}
ROUNDS=${ROUNDS:-3}
HOLD_S=${HOLD_S:-5}                 # idle between the switch away and back: longer than one park-pressure tick
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
WORK=$OUT
CFG_SRC=$(ls -t saguaro-*.json 2>/dev/null | head -1 || true)
[ -n "$CFG_SRC" ] || { echo "no saguaro-<model>.json here: set the model up first (or run this from the Saguaro folder)"; exit 1; }
rm -rf "$WORK"; mkdir -p "$WORK/sessions"
LOG=$WORK/engine.log; SRV=$WORK/server.log
CFG=$WORK/config.json

port_pids() {   # whatever still holds the port (a previous run's engine child outlived its launcher)
    ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE "pid=[0-9]+" | cut -d= -f2
    lsof -nP -t -i "TCP:$PORT" -s TCP:LISTEN 2>/dev/null
    true
}

cleanup() {   # on exit the port is ours again: kill the server and whatever still holds it (its engine child)
    kill "$SRV_PID" 2>/dev/null
    for p in $(port_pids); do kill "$p" 2>/dev/null; done
    true
}

for _ in $(seq 1 60); do            # start only on a free port: a held one means the old engine is still in there
    [ -z "$(port_pids)" ] && break
    sleep 1
done
[ -z "$(port_pids)" ] || { echo "port $PORT is still held by $(port_pids | tr '\n' ' ') - stop the server on it first"; exit 1; }

"$PY" - "$CFG_SRC" "$CFG" "$PORT" "$WORK/sessions" "$LOG" <<'PYEOF'
import json, sys
src, dst, port, sess, log = sys.argv[1:6]
cfg = json.load(open(src, encoding="utf-8"))
args = [a for a in cfg.get("args", [])]
if "--session-dir" in args:                          # the test copy parks its conversations in its own folder
    i = args.index("--session-dir"); args[i + 1] = sess
else:
    _env = cfg.setdefault("env", {})            # (GLM/DS4/ik2 name it in the env: *_SLOT_DIR)
    _slot = [k for k in _env if k.endswith("_SLOT_DIR")]
    for _k in _slot:
        _env[_k] = sess
    if not _slot:
        args += ["--session-dir", sess]
cfg["args"] = args
cfg["port"] = int(port)
cfg.pop("api_key", None)                             # (a test copy: no key, the readiness poll needs none)
cfg["log"] = log                                     # the engine's stderr: the park/restore lines are read from here
if __import__("os").environ.get("SAGUARO_TEST_EXE"): # a build from tools/qfn/build.sh, to test a candidate engine
    cfg["exe"] = __import__("os").environ["SAGUARO_TEST_EXE"]
json.dump(cfg, open(dst, "w", encoding="utf-8"), indent=1)
print(f"test config {dst}: port {port}, session dir {sess}, exe {cfg.get('exe')}")
PYEOF

say() { printf '\n== %s\n' "$*"; }
api() { "$PY" - "$1" <<'PYEOF'
import json, sys, urllib.request
port = int(__import__("os").environ.get("PORT", "8092"))
body = json.dumps(json.load(open(sys.argv[1], encoding="utf-8")), ensure_ascii=False).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    out = json.loads(r.read())
print(out["choices"][0]["message"]["content"])
PYEOF
}

say "starting the test copy on port $PORT (the model loads - watch $SRV)"
"$PY" saguaro.py --config "$CFG" --port "$PORT" > "$SRV" 2>&1 &
SRV_PID=$!
trap cleanup EXIT
for _ in $(seq 1 240); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/metrics" 2>/dev/null; then break; fi
    kill -0 $SRV_PID 2>/dev/null || { echo "the server died; see $SRV"; exit 1; }
    sleep 5
done
curl -sf -o /dev/null "http://127.0.0.1:$PORT/metrics" || { echo "the API did not come up in 20 min; see $SRV"; exit 1; }
disown $SRV_PID 2>/dev/null || true

gen_body() {                                 # $1 file, $2 filler tokens, $3 seed word, $4 user text, $5 reply|none
    "$PY" - "$1" "$2" "$3" "$4" "${5:-}" <<'PYEOF'
import json, sys
f, n, seed, text, reply = sys.argv[1:6]
filler = (" " .join(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"] * (int(n) // 10 + 1)))[:int(n) * 5]
msgs = [{"role": "system", "content": f"Reference {seed}: {filler}"}, {"role": "user", "content": text}]
if reply:
    msgs = msgs[:-1] + [{"role": "assistant", "content": reply}, {"role": "user", "content": text}]
json.dump({"model": "m", "messages": msgs, "max_tokens": 16, "temperature": 0},
          open(f, "w", encoding="utf-8"), ensure_ascii=False)
PYEOF
}

for N in $SIZES; do
    START=$(( $(wc -l < "$LOG" 2>/dev/null || echo 0) ))
    say "conversation of $N tokens, $ROUNDS rounds (parking is the switch to the other conversation)"
    for R in $(seq 1 "$ROUNDS"); do
        gen_body "$WORK/a$R.json" "$N" "conv-A-$N-$R" "What does this reference say about the weather? Answer in one word."
        A_REPLY=$(PORT=$PORT api "$WORK/a$R.json" || true)
        gen_body "$WORK/b$R.json" 200 "conv-B" "Answer with the single word: pong"
        PORT=$PORT api "$WORK/b$R.json" > /dev/null        # the switch: this is where A is parked
        sleep "$HOLD_S"                                    # let a park-pressure tick pass
        gen_body "$WORK/a2$R.json" "$N" "conv-A-$N-$R" "And about the rain? One word." "$A_REPLY"
        PORT=$PORT api "$WORK/a2$R.json" > /dev/null       # and back: this is where A is restored
        sleep 2
    done
    echo "--- engine lines for this size (from $LOG) ---"
    tail -n +$((START + 1)) "$LOG" | grep -E "conversation cache: (parked|restored|dropped)" | \
        sed -e 's/^saguaro serve: //' || echo "  (no park lines: was the conversation cache off?)"
    if tail -n +$((START + 1)) "$LOG" | grep -qE "released .* parked in RAM|short of RAM|\(pressure:"; then
        echo "RESULT $N: FAIL - a parked conversation was released for RAM pressure"
    elif tail -n +$((START + 1)) "$LOG" | grep -qE "conversation cache: restored [0-9]+ tokens( \((checkpoint|live)\))?"; then
        # the engine logs the conversation's real token count, not the requested size, and a park that kept its
        # RAM restores as "restored N tokens (checkpoint)" as often as "(live)": match the line, not N
        echo "RESULT $N: PASS - the parks stayed in RAM and restores came back from it"
    else
        echo "RESULT $N: CHECK - no release, but no RAM restore either (parking refused, or the cache full: see the lines)"
    fi
done

say "done. stopping the test copy"
kill $SRV_PID 2>/dev/null || true
wait $SRV_PID 2>/dev/null || true
