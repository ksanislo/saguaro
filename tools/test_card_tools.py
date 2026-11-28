#!/usr/bin/env python3
"""tools/test_card_tools.py - PLAN-p124t2, no card and no model: the card tests' readiness probe
(tools/card_tests/health.sh) must SEE the failed-load state of /health and let the loop stop at once,
instead of waiting the full 20 min while the server stands with the error page (serve/server.py's
LoadState: 503 {"status": "failed", "error": ...}; while loading 503 {"status": "loading", ...}; the
running Service answers 200).  A fake /health on 127.0.0.1 answers each state and health_status, run
under bash with curl, must name it.  This is also the bash -n gate for every tools/card_tests script.

    /opt/llm/saguaro/.venv/bin/python tools/test_card_tools.py     (or python3; needs bash and curl)"""
import http.server
import subprocess
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARD_TESTS = ROOT / "tools" / "card_tests"
HEALTH_SH = CARD_TESTS / "health.sh"


def health_status_for(body, code):
    """The word health_status prints for a /health answering exactly (code, body)."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        return probe("health_status %d" % srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


def probe(command):
    """Run bash with health.sh sourced the way the card tests source it, under set -eu like park.sh."""
    out = subprocess.run(["bash", "-c", 'set -eu; . "%s"; %s' % (HEALTH_SH, command)],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


class ReadinessProbeTest(unittest.TestCase):
    def test_failed_load_is_seen_at_once(self):
        # the failed-load /health, the shape LoadState.fail + LoadState.health send (issue #20): the readiness
        # loop takes this as its stop signal instead of waiting 240 x 5 s for a 200 that will never come
        body = ('{"status": "failed", "model": "qwen3.8-flash-next", "elapsed_s": 41, '
                '"line": "the engine exited before it was ready", "supervisor": null, '
                '"error": "the engine exited before it was ready (see scratch/x/engine.log)"}')
        self.assertEqual(health_status_for(body, 503), "failed")

    def test_still_loading_keeps_the_loop_waiting(self):
        body = ('{"status": "loading", "model": "m", "elapsed_s": 3, "line": "loading the model ...", '
                '"supervisor": null}')
        self.assertEqual(health_status_for(body, 503), "loading")

    def test_the_running_service_is_ready(self):
        self.assertEqual(health_status_for('{"status": "ok", "max_context": 131072, "model": "m"}', 200),
                         "ready")

    def test_a_closed_port_is_loading_not_failed(self):
        # a server that EXITED ends the loop through the script's own kill -0 check, not through /health
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertEqual(probe("health_status %d" % port), "loading")

    def test_no_other_state_reads_as_failed(self):
        # a loading answer whose engine line merely mentions a failed something stays "loading"
        body = ('{"status": "loading", "model": "m", "elapsed_s": 9, "line": "layer 7 failed to warm, '
                'retrying", "supervisor": null}')
        self.assertEqual(health_status_for(body, 503), "loading")

    def test_health_sh_is_bash_clean(self):
        subprocess.run(["bash", "-n", str(HEALTH_SH)], check=True)

    def test_card_test_scripts_are_bash_clean(self):
        for f in sorted(CARD_TESTS.glob("*.sh")):
            subprocess.run(["bash", "-n", str(f)], check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
