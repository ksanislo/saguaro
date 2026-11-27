#!/usr/bin/env bash
# tools/card_tests/mortar_pressure.sh - P124x on the card: the agent-like load that ended the 9070 XT box, run
# against the fixed engine.  RAM and VRAM are sampled every SAMPLE_S seconds for DURATION seconds; the rules
# this branch adds (the resident-budget margin, the 512+512 reserve floor, the swap-voids-the-credit release,
# the drained-#267 recovery) must keep the box out of the shape report 1.1 read thirteen times.
#
#   CFG=/path/to/<model>.json ./tools/card_tests/mortar_pressure.sh
#   CFG=... DURATION=3600 VRAM_FLOOR_MIB=512 ./tools/card_tests/mortar_pressure.sh
#   ./tools/card_tests/mortar_pressure.sh --out scratch/mortar-pressure-2   # write under a different folder
#
# What it does: a test copy of the card model config on port 8092 with its OWN session dir, its OWN engine log
# and the api_key dropped (the readiness poll and the requests send none) - everything it writes lands under
# its --out folder (default scratch/mortar-pressure) and nothing of the live box's config, log or sessions is
# touched (its config is only read).  The load is the plan's: THREE conversations of ~100K / ~175K / ~250K
# tokens, each one reading its prompt and then answering follow-up turns for DURATION s - with --parallel 2
# one always waits, so the turns cycle through parks, turn-save captures, session writes and restores: the
# RAM churn the mortar box died of.  A sampler reads /proc/meminfo (MemAvailable, SwapTotal-SwapFree) and the
# card's free VRAM (rocm-smi, else nvidia-smi) every SAMPLE_S s into samples.csv.
#
# PASS (the plan's line, and the PASS/FAIL verdict lands on stdout):
#   * no abort      - no "the engine had stopped" in the server log.  Every #267 shape fails this: a
#                     not-drained line is fatal, and a drained one (the recovered note, P124x d) ends its
#                     request without a kill but the engine exits after the line and the server reloads it
#                     today (review F-A - the serve loop has no return-to-stdin path for it yet).  The
#                     recovered line is echoed at the verdict so the run can tell the shapes apart;
#   * VRAM never under the floor - every free-VRAM sample >= VRAM_FLOOR_MIB (default 512: the engine's own
#                     #2065 paging line; the fix should keep the card at or above 512 + slack at request start);
#   * no swap growth - swap in use never rises more than SWAP_GROW_MIB above its baseline at load start
#                     (residual swap the box booted with is allowed; a storm is not);
#   * and the release rule said its piece if it had to: the pressure lines the engine prints are listed.
# It waits until the port is free before starting, and on exit kills whatever still holds the port (the
# server's child engine outlived the launcher).  Needs: curl + a python (the install's .venv by default) +
# rocm-smi or nvidia-smi for the VRAM half.
set -u -o pipefail

OUT=scratch/mortar-pressure
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
DURATION=${DURATION:-1800}            # the plan: 30 min of agent-like load
SAMPLE_S=${SAMPLE_S:-10}              # the plan: RAM / VRAM sampled each 10 s
TOKENS_A=${TOKENS_A:-100000}
TOKENS_B=${TOKENS_B:-175000}
TOKENS_C=${TOKENS_C:-250000}
VRAM_FLOOR_MIB=${VRAM_FLOOR_MIB:-512}     # kVramSafeMib (Linux): the engine's own LOW line
SWAP_GROW_MIB=${SWAP_GROW_MIB:-256}       # SAGUARO_PARK_SWAP_MIB: growth past this is a storm
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR" || exit 1
DIR=$OUT
SERVE_LOG="$DIR/serve.log"
ELOG="$DIR/engine.log"               # the engine's own stderr, kept inside the out folder (never the install's log)
CSV="$DIR/samples.csv"

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

fail() { echo "FAIL: $*"
         cleanup
         echo "--- last of $SERVE_LOG ---" >&2; tail -40 "$SERVE_LOG" 2>/dev/null >&2
         echo "--- last of $ELOG ---" >&2; tail -20 "$ELOG" 2>/dev/null >&2; exit 1; }

rm -rf "$DIR"; mkdir -p "$DIR" "$DIR/sessions"

# ---- the test copy of the config: same engine, its own session dir, log and no key ---------------------------
CFG_T="$DIR/mortar-pressure.json"
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
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY

for _ in $(seq 1 60); do             # start only on a free port: a held one means the old engine is still in there
    [ -z "$(port_pids)" ] && break
    sleep 1
done
[ -z "$(port_pids)" ] || { echo "port $PORT is still held by $(port_pids | tr '\n' ' ') - stop the server on it first"; exit 1; }

# ---- start the server --------------------------------------------------------------------------------------
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

# what the rules said at start (informational, and the budget clamp must be honest about the box):
grep -E "resident budget|reserve stays at least|resident RAM|swap in use passes" "$SERVE_LOG" "$ELOG" 2>/dev/null | head

# ---- prompts: fresh random words, ~4 chars a token, a different seed per conversation -----------------------
prompt() { "$PYTHON" - "$1" "$2" <<'PY'
import random, sys
rng = random.Random(f"p124x-{sys.argv[2]}")
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu analysis bucket canyon engine floor "
         "gamma hinge input junction latent metric needle oracle pixel query radio sector token unity vector "
         "window xenon yield zenith memory swap resident park window tier cache margin pool capture").split()
out, n = [], 0
while n < int(sys.argv[1]):
    w = rng.choice(words)
    out.append(w)
    n += len(w) + 1
print(f"P124x pressure probe {sys.argv[2]}: " + " ".join(out))
PY
}
prompt "$((TOKENS_A * 4))" A > "$DIR/prompt_A.txt" || fail "prompt A"
prompt "$((TOKENS_B * 4))" B > "$DIR/prompt_B.txt" || fail "prompt B"
prompt "$((TOKENS_C * 4))" C > "$DIR/prompt_C.txt" || fail "prompt C"

# ---- the sampler: /proc/meminfo and the card, every SAMPLE_S s, until the run ends ---------------------------
vram_cmd=""
if command -v rocm-smi >/dev/null 2>&1; then vram_cmd=rocm-smi
elif command -v nvidia-smi >/dev/null 2>&1; then vram_cmd=nvidia-smi
fi
[ -n "$vram_cmd" ] || fail "neither rocm-smi nor nvidia-smi: the VRAM half of the sampling needs one of them"
"$PYTHON" - "$CSV" "$vram_cmd" "$SAMPLE_S" "$DIR/stop" <<'PY' >"$DIR/sampler.log" 2>&1 &
import csv, re, subprocess, sys, time
csv_path, tool, sample_s, stop = sys.argv[1:5]
def swap_and_avail():                       # /proc/meminfo is in kB
    kv = {}
    for line in open("/proc/meminfo"):
        k, _, rest = line.partition(":")
        kv[k] = int(rest.split()[0])
    return kv.get("MemAvailable"), max(0, kv.get("SwapTotal", 0) - kv.get("SwapFree", 0))
def vram_free_mib():
    try:
        if tool == "rocm-smi":
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                                 capture_output=True, text=True, timeout=15).stdout
            used = total = 0
            for line in out.splitlines():
                f = [x.strip() for x in line.split(",")]
                if len(f) >= 4 and f[2] in ("VRAM Total Memory (B)", "Total Memory (B)"):
                    total = max(total, int(f[3]))
                if len(f) >= 4 and f[2] in ("VRAM Total Used Memory (B)", "Total Used Memory (B)"):
                    used = max(used, int(f[3]))
            return (total - used) // (1 << 20) if total else None
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        vals = [int(x) for x in re.findall(r"\d+", out)]
        return min(vals) if vals else None
    except Exception:
        return None
t_end = time.time() + 24 * 3600            # the run stops the loop by writing the stop file
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["epoch", "mem_available_kib", "swap_used_kib", "vram_free_mib"])
    while not __import__("os").path.exists(stop) and time.time() < t_end:
        avail, swap = swap_and_avail()
        w.writerow([int(time.time()), avail, swap, vram_free_mib()])
        fh.flush()
        time.sleep(float(sample_s))
PY
SAMPLER=$!

# ---- the load: three conversations reading ~100K/175K/250K, then follow-up turns to the deadline ------------
run_conv() {   # run_conv <name> <prompt-file> : one conversation until $DIR/stop appears; rc 0 = clean
    "$PYTHON" - "$1" "$2" "$MODEL" "$PORT" "$DIR/stop" "$DIR/conv_$1.log" <<'PY'
import json, os, sys, urllib.request
name, pf, model, port, stop, logf = sys.argv[1:7]
msgs = [{"role": "user", "content": open(pf).read()}]
turn = 0
def ask():
    body = {"model": model, "temperature": 0, "max_tokens": 160, "messages": msgs}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("usage", {})
try:
    while not os.path.exists(stop):
        text, usage = ask()
        turn += 1
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"Follow-up {turn}: say OK and one sentence about the brief."}]
        with open(logf, "a") as lg:
            lg.write(f"turn {turn}: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}\n")
    with open(logf, "a") as lg:
        lg.write(f"done: {turn} turns\n")
except Exception as e:      # an answer that died mid-run: the watchdog verdict below decides
    with open(logf, "a") as lg:
        lg.write(f"died: {e}\n")
    sys.exit(1)
PY
}
conv() {   # conv <name> <prompt-file>: start it and remember its pid
    run_conv "$1" "$2" &
    echo "$!" > "$DIR/conv_$1.pid"
}
echo "== load: three conversations (~${TOKENS_A}/${TOKENS_B}/${TOKENS_C} tokens), ${DURATION}s, samples every ${SAMPLE_S}s =="
for _ in $(seq 1 30); do [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -ge 2 ] && break; sleep 1; done
BASE_SWAP=$(awk 'NR==2 {print $3}' "$CSV")     # the sampler's first data row: the box's residual swap
[ -n "$BASE_SWAP" ] || fail "the sampler wrote no rows (see $DIR/sampler.log)"
conv A "$DIR/prompt_A.txt"
conv B "$DIR/prompt_B.txt"
conv C "$DIR/prompt_C.txt"

END=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$END" ]; do
    kill -0 "$(cat "$DIR/serve.pid")" 2>/dev/null || fail "the server died during the load; see $SERVE_LOG"
    grep -q "the engine had stopped" "$SERVE_LOG" && fail "abort: the engine had stopped during the load (the plan's no-abort line)"
    sleep 5
done
: > "$DIR/stop"
wait "$(cat "$DIR/conv_A.pid")" 2>/dev/null; RA=$?
wait "$(cat "$DIR/conv_B.pid")" 2>/dev/null; RB=$?
wait "$(cat "$DIR/conv_C.pid")" 2>/dev/null; RC=$?
[ "$SAMPLER" ] && wait "$SAMPLER" 2>/dev/null

# ---- the verdict: no abort, VRAM never under the floor, no swap growth ---------------------------------------
grep -q "the engine had stopped" "$SERVE_LOG" && fail "abort: the engine had stopped during the load"
grep -E "timed out at layer|released and the GPU finished" "$SERVE_LOG" "$ELOG" 2>/dev/null | head -4
RSS=$({ ps -C saguaro-qfn -o rss= 2>/dev/null; } | awk '{s += $1} END {print int(s / 1024)}')
echo "== verdict =="
echo "conversations ended rc $RA/$RB/$RC (0 = every turn answered to the deadline); engine RSS now: ${RSS:-?} MiB"
echo "pressure/release lines the engine said (P124x rule c):"
grep -E "already in swap|pooled park buffers|resident budget" "$SERVE_LOG" "$ELOG" 2>/dev/null | tail -8
"$PYTHON" - "$CSV" "$BASE_SWAP" "$VRAM_FLOOR_MIB" "$SWAP_GROW_MIB" <<'PY' || fail "the samples.csv verdict came out FAIL (above)"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
base = int(sys.argv[2]); floor = int(sys.argv[3]); grow = int(sys.argv[4])
if not rows:
    print("FAIL: the sampler wrote no rows"); sys.exit(1)
vmins = [int(r["vram_free_mib"]) for r in rows if r["vram_free_mib"] not in ("", "None")]
smax = max(int(r["swap_used_kib"]) for r in rows)
amax = min(int(r["mem_available_kib"]) for r in rows if r["mem_available_kib"])
print(f"samples: {len(rows)}, swap used max {(smax - 0) >> 10} MiB (baseline {base >> 10} MiB, growth budget {grow} MiB), "
      f"MemAvailable min {amax >> 10} MiB" + (f", VRAM free min {min(vmins)} MiB (floor {floor})" if vmins else ", no VRAM samples"))
bad = []
if vmins and min(vmins) < floor:
    bad.append(f"VRAM fell to {min(vmins)} MiB, under the {floor} MiB floor")
if (smax - base) > grow * 1024:
    bad.append(f"swap grew {((smax - base) >> 10)} MiB past its baseline (over the {grow} MiB line)")
if bad:
    print("FAIL: " + "; ".join(bad)); sys.exit(1)
print("PASS: no abort, VRAM never under the floor, no swap growth")
PY

echo "artifacts under $DIR (serve.log, engine.log, samples.csv, conv_*.log); the engine stopped with the server."
