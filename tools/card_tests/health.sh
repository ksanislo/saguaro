# tools/card_tests/health.sh - the readiness probe the card tests share (PLAN-p124t2): GET /health as one word.
#
# serve/server.py answers on the port from the first second of every start (its "starting" service, issue #20):
#   while the engine loads   503 {"status": "loading", ...}
#   the model failed to load 503 {"status": "failed", "error": ...}   (LoadState.fail / LoadState.health - the
#       process then STANDS with the page that says what went wrong, so a readiness poll that only takes a 200
#       as ready waits its full 20 min through a dead engine, the cards idle)
#   the model is there       200 (the real Service has taken the port over)
#
#   . "$(dirname "$0")/health.sh"                       # from a script in tools/card_tests/, after its cd to ROOT
#   case "$(health_status "$PORT")" in
#       ready)  ... break ;;                            # the API is up
#       failed) ... exit 1 ;;                           # stop at once: print the server log's last 20 lines
#       loading) ... ;;                                 # keep waiting: still starting, or nothing answers the
#                                                       # port yet - a dead server process is the loop's own
#                                                       # kill -0 check, never a /health answer
#   esac
#
# Needs curl.  Tested without a card or a model by tools/test_card_tools.py (a fake /health answering each state).

health_status() {   # $1: the port -> prints "ready", "loading" or "failed"; nothing else, always exit 0
    local out code body
    out=$(curl -s -m 5 -w '\n%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null) || { echo loading; return 0; }
    code=${out##*$'\n'}
    body=${out%$'\n'*}
    if [ "$code" = "200" ]; then
        echo ready
    elif printf '%s' "$body" | grep -qE '"status"[[:space:]]*:[[:space:]]*"failed"'; then
        echo failed
    else
        echo loading
    fi
}
