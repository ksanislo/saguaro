"""tools/test_mem_vs_ctx.py - P124n's memory-vs-context check, tested without a GPU: the /proc parsing on fixed
inputs (a status file's VmLck/VmPin/VmRSS, the pid tree through a fake /proc whose comm carries spaces and
parens, and the port's owner found through a fake /proc/net/tcp by socket inode), the GPU reader on canned
nvidia-smi / rocm-smi output, the engine-log budget grep, the run config copy (own session dir and log, no
api_key, --max-context set), the table, and the PASS/FAIL rule (less than max(2%, 256 MiB), the biggest
difference named) on fixed figures - plus one real cycle against `python -m serve.server --engine mock`: the
start waits for /health ok, the port's owner is the server, and the stop frees the port again.

    python -m unittest tools.test_mem_vs_ctx -v
"""
from __future__ import annotations

import os
import signal
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import mem_vs_ctx as MV  # noqa: E402

# one /proc/<pid>/status as the kernel writes it (KiB; the tree's two processes look like this)
STATUS = """Name:\tpython
Umask:\t0022
State:\tS (sleeping)
Tgid:\t4242
VmPeak:\t 9999999 kB
VmSize:\t 759440 kB
VmLck:\t 1048576 kB
VmPin:\t   32768 kB
VmRSS:\t 2097152 kB
Threads:\t23
"""


def fake_proc(tmp, pids, net_tcp=None):
    """A /proc-shaped tree: <tmp>/<pid>/stat (ppid = the second figure after the comm), plus a net/tcp."""
    for pid, ppid, comm in pids:
        d = tmp / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194304 1 0 0 0 0 0 0 0 20 0 1 0 99\n",
                                encoding="ascii")
    if net_tcp is not None:
        (tmp / "net").mkdir(parents=True, exist_ok=True)
        (tmp / "net" / "tcp").write_text(net_tcp, encoding="ascii")
    return tmp


def tcp_line(port, inode, state="0A"):
    return (f"   0: 0100007F:{port:04X} 00000000:0000 {state} 00000000:00000000 00:00000000 00000000 "
            f"    0        0 {inode} 1 0000000000000000 100 0 0 10 0\n")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ProcParsingTests(unittest.TestCase):
    def test_status_kib_to_mib_and_absent_is_zero(self):
        m = MV.parse_proc_status(STATUS)
        self.assertEqual(m["VmLck"], 1024.0)          # 1048576 kB
        self.assertEqual(m["VmPin"], 32.0)
        self.assertEqual(m["VmRSS"], 2048.0)

    def test_status_only_those_three(self):
        m = MV.parse_proc_status("VmSize:\t 759440 kB\nVmLck:\t 1 kB\n")   # (kB rounding: 1 kB is 0.001 MiB)
        self.assertAlmostEqual(m["VmLck"], 0.0009765625, places=9)
        self.assertEqual(m["VmPin"], 0.0)
        self.assertEqual(m["VmRSS"], 0.0)

    def test_tree_follows_ppid_over_a_messy_comm(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            fake_proc(tmp, [(100, 1, "python"), (200, 100, "weird (comm) name"), (300, 200, "engine"),
                            (400, 1, "unrelated"), (1000, 300, "grandchild")])
            self.assertEqual(MV.process_tree(100, str(tmp)), [100, 200, 300, 1000])
            self.assertEqual(MV.process_tree(400, str(tmp)), [400])

    def test_tree_mibs_sums_the_tree(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            fake_proc(tmp, [(100, 1, "server"), (200, 100, "engine")])
            (tmp / "100" / "status").write_text(STATUS, encoding="ascii")
            (tmp / "200" / "status").write_text("VmLck:\t 512 kB\nVmPin:\t 0 kB\nVmRSS:\t 1024 kB\n",
                                                encoding="ascii")
            m = MV.tree_mibs(MV.process_tree(100, str(tmp)), str(tmp))
            self.assertEqual(m["VmLck"], 1024.5)      # (1048576 + 512 kB)
            self.assertEqual(m["VmRSS"], 2049.0)

    def test_port_owner_by_inode_tcp_and_tcp6(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            fake_proc(tmp, [(77, 1, "server"), (78, 1, "other")],
                      "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  "
                      "timeout inode\n" + tcp_line(8093, 555))
            fd = tmp / "77" / "fd"
            fd.mkdir()
            os.symlink("socket:[555]", fd / "3")      # (the listen socket: a dangling link reads back fine)
            os.symlink("/some/file", fd / "1")
            (tmp / "tcp6").write_text("  sl  local_address rem_address   st x\n", encoding="ascii")
            self.assertEqual(MV.port_owner(8093, str(tmp)), 77)
            self.assertIsNone(MV.port_owner(8094, str(tmp)))

    def test_port_owner_established_is_not_a_listener(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            fake_proc(tmp, [(77, 1, "client")], "  sl  local_address x\n" + tcp_line(8093, 555, state="01"))
            fd = tmp / "77" / "fd"
            fd.mkdir()
            os.symlink("socket:[555]", fd / "3")
            self.assertIsNone(MV.port_owner(8093, str(tmp)))


class GpuReaderTests(unittest.TestCase):
    NV = "0, 1024\n1, 4096\n2, 7\nnot a row, x\n"
    ROCM = ("GPU[0]\t\t: VRAM Total Memory (B): 17163091968\n"
            "GPU[0]\t\t: VRAM Total Used Memory (B): 1073741824\n"
            "GPU[1]\t\t: VRAM Total Used Memory (B): 536870912\n")

    @staticmethod
    def fake_run(out):
        return lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=out)

    def test_parsers(self):
        self.assertEqual(MV.parse_nvidia_used(self.NV), {0: 1024.0, 1: 4096.0, 2: 7.0})
        self.assertEqual(MV.parse_rocm_used(self.ROCM), {0: 1024.0, 1: 512.0})

    def test_nvidia_first_and_card_sum(self):
        used = MV.gpu_used_mib([0, 2], run=self.fake_run(self.NV), which=lambda n: "/usr/bin/" + n)
        self.assertEqual(used, 1031.0)

    def test_rocm_when_no_nvidia_all_cards_when_none_named(self):
        used = MV.gpu_used_mib([], run=self.fake_run(self.ROCM),
                               which=lambda n: None if n == "nvidia-smi" else "/usr/bin/rocm-smi")
        self.assertEqual(used, 1536.0)

    def test_a_card_the_reader_does_not_list_cannot_say(self):
        self.assertIsNone(MV.gpu_used_mib([5], run=self.fake_run(self.NV), which=lambda n: "/usr/bin/" + n))

    def test_no_reader_is_none(self):
        self.assertIsNone(MV.gpu_used_mib([], run=self.fake_run(self.NV), which=lambda n: None))


class BudgetAndConfigTests(unittest.TestCase):
    def test_budget_lines(self):
        log = ("[saguaro] ready\n"
               "INFO: KV budget 4096 positions in VRAM\n"
               "resident experts: 128 of 384 (32 GiB)\n"
               "KV cache: pinned RAM tier 96 GiB\n"
               "kv streamed lazily - lowercase-only kv is not a budget line\n"
               "loading weights\n")
        self.assertEqual(MV.budget_lines(log), ["INFO: KV budget 4096 positions in VRAM",
                                                "resident experts: 128 of 384 (32 GiB)",
                                                "KV cache: pinned RAM tier 96 GiB"])

    def test_run_config_own_session_dir_log_no_key_context(self):
        cfg = {"exe": "bin/saguaro-ik2", "api_key": "sekret", "host": "0.0.0.0", "idle_unload_s": 300,
               "log": "logs/real.log",
               "args": ["--model", "m.gguf", "--max-context", "131072", "--threads", "8"],
               "env": {"SAGUARO_IK2_SLOT_DIR": "var/x/sessions", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
               "gpu": [2, 3]}
        out = MV.run_config(cfg, Path("/run/32768"), 32768)
        self.assertNotIn("api_key", out)
        self.assertEqual(out["args"], ["--model", "m.gguf", "--max-context", "32768", "--threads", "8"])
        self.assertEqual(out["env"]["SAGUARO_IK2_SLOT_DIR"], "/run/32768/sessions")
        self.assertEqual(out["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")      # (the others stay)
        self.assertEqual(out["log"], "/run/32768/engine.log")
        self.assertEqual(out["host"], "127.0.0.1")
        self.assertEqual(out["idle_unload_s"], 0)
        self.assertEqual(cfg["args"][3], "131072")                           # (the caller's dict untouched)
        self.assertEqual(MV.gpu_cards(cfg), [2, 3])

    def test_run_config_qfn_session_dir_arg_and_missing_context(self):
        out = MV.run_config({"args": ["--session-dir", "/var/sessions", "--gpus", "4"],
                             "env": {}}, Path("/run/1048576"), 1048576)
        self.assertEqual(out["args"], ["--session-dir", "/run/1048576/sessions", "--gpus", "4",
                                       "--max-context", "1048576"])          # (appended when the config has none)
        self.assertEqual(MV.gpu_cards({"gpu": "0,2"}), [0, 2])
        self.assertEqual(MV.gpu_cards({"gpu": 1}), [1])
        self.assertEqual(MV.gpu_cards({}), [])


class TableAndVerdictTests(unittest.TestCase):
    @staticmethod
    def rows(pinned, rss, gpu=(None, None)):
        return [{"context": c, "pinned_mib": p, "rss_mib": r, "gpu_mib": g, "budget": []}
                for c, p, r, g in zip((32768, 262144), pinned, rss, gpu)]

    def test_table_aligns_and_prints_dash_without_a_reader(self):
        table = MV.format_table(self.rows([1000.0, 1010.0], [40000.0, 40500.0]))
        lines = table.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("pinned MiB", lines[0])
        self.assertIn("GPU used MiB", lines[0])
        self.assertIn("-", lines[1])
        self.assertTrue(lines[1].rstrip().endswith("-"))
        self.assertIn("40000.0", lines[1])

    def test_within_the_floor_passes(self):
        # 40000 vs 40500: 500 MiB apart, over the 2% (810 MiB of 40500 -> 500 < 810) and the 256 MiB floor
        ok, why = MV.verdict(self.rows([1000.0, 1010.0], [40000.0, 40500.0]))
        self.assertTrue(ok)
        self.assertTrue(why.startswith("PASS"))

    def test_floor_beats_two_percent(self):
        # 10000 vs 10300: 2% is only 206, the 256 MiB floor holds: 300 >= 256 fails
        ok, why = MV.verdict(self.rows([10000.0, 10300.0], [40000.0, 40500.0]))
        self.assertFalse(ok)
        self.assertIn("pinned RAM", why)
        # 10000 vs 10200: 200 < max(200 ... no: 2% of 10200 = 204, floor 256) -> 200 < 256 passes
        self.assertTrue(MV.verdict(self.rows([10000.0, 10200.0], [40000.0, 40500.0]))[0])

    def test_two_percent_beats_the_floor(self):
        # pinned 1000000 vs 1001000: 1000 apart, limit 2% = 20020 -> passes; 30000 apart: 2% of 1030000 = 20600
        # is the limit there, not the 256 MiB floor -> fails - the percent rules a big figure, the floor a small one
        self.assertTrue(MV.verdict(self.rows([1000000.0, 1001000.0], [40000.0, 40500.0]))[0])
        ok, why = MV.verdict(self.rows([1000000.0, 1030000.0], [40000.0, 40500.0]))
        self.assertFalse(ok)
        self.assertIn("pinned RAM", why)

    def test_fail_names_the_biggest_difference(self):
        # RSS differs by 5000 (>= its limit), pinned by 10000 (>= its 2% = 1020): the biggest named is pinned
        ok, why = MV.verdict(self.rows([100000.0, 110000.0], [50000.0, 55000.0], gpu=(20000.0, 20100.0)))
        self.assertFalse(ok)
        self.assertIn("pinned RAM differs by 10000.0 MiB", why)

    def test_a_figure_the_pc_cannot_read_is_left_out(self):
        # identical RAM, no GPU reader: PASS (the GPU row is not compared)
        ok, _ = MV.verdict(self.rows([1000.0, 1010.0], [40000.0, 40500.0]))
        self.assertTrue(ok)
        # a GPU that jumped is a FAIL naming it
        ok, why = MV.verdict(self.rows([1000.0, 1010.0], [40000.0, 40500.0], gpu=(20000.0, 40000.0)))
        self.assertFalse(ok)
        self.assertIn("GPU used", why)

    def test_one_row_nothing_to_compare(self):
        ok, why = MV.verdict([{"context": 32768, "pinned_mib": 1.0, "rss_mib": 2.0, "gpu_mib": None,
                               "budget": []}])
        self.assertFalse(ok)
        self.assertIn("nothing", why)


class StopServerTests(unittest.TestCase):
    def test_a_stranger_that_took_the_port_gets_no_signal(self):
        # (the run's server (77) and engine (88) are ours; 500 holds the port instead - the pre-check passed,
        #  the server died first, someone else bound it: stop_server must stop OUR tree and leave 500 alone)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            fake_proc(tmp, [(77, 1, "server"), (88, 77, "engine"), (500, 1, "stranger")],
                      "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  "
                      "timeout inode\n" + tcp_line(8093, 555))
            fd = tmp / "500" / "fd"
            fd.mkdir()
            os.symlink("socket:[555]", fd / "4")
            proc = SimpleNamespace(pid=77, poll=lambda: None, wait=lambda **kw: None)
            with mock.patch.object(os, "kill") as killed:
                freed = MV.stop_server(8093, proc, wait_s=0.0, proc_dir=str(tmp))
            self.assertFalse(freed)                                # (the port is still the stranger's)
            targets = {c.args for c in killed.call_args_list}
            self.assertNotIn(500, {p for p, _s in targets})
            self.assertIn((77, signal.SIGTERM), targets)           # (our own server got the TERM)
            self.assertIn((88, signal.SIGKILL), targets)           # (and our own leftover the KILL)


class PlatformTests(unittest.TestCase):
    def test_has_proc_here_yes_in_a_fake_root_no(self):
        self.assertTrue(MV.has_proc())
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(MV.has_proc(str(Path(t) / "no-proc-here")))

    def test_main_refuses_a_box_without_proc(self):
        with mock.patch.object(MV, "has_proc", lambda *a, **k: False):
            self.assertEqual(MV.main(["no-such-config.json", "--contexts", "32768,262144"]), 2)


@unittest.skipUnless(Path("/proc/self/status").is_file(), "the real /proc (VmRSS, the port's owner): Linux only")
class MockEngineCycleTests(unittest.TestCase):
    """One real start/stop through the tool's own functions, against the scripted mock engine - no GPU, no model."""

    def test_start_stop_cycle_frees_the_port(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "mock-run"
            proc = MV.start_server(MV.server_cmd(sys.executable, port), run_dir, cwd=ROOT)
            try:
                health = MV.wait_health(port, 60, proc=proc)
                self.assertEqual(health["status"], "ok")
                self.assertEqual(MV.port_owner(port), proc.pid)
                tree = MV.process_tree(proc.pid)
                self.assertIn(proc.pid, tree)
                mibs = MV.tree_mibs(tree)
                self.assertGreater(mibs["VmRSS"], 0.0)
            finally:
                freed = MV.stop_server(port, proc, wait_s=60)
            self.assertTrue(freed)
            self.assertIsNone(MV.port_owner(port))
            self.assertIsNotNone(proc.poll())
            self.assertTrue((run_dir / "server.log").exists())


if __name__ == "__main__":
    unittest.main()
