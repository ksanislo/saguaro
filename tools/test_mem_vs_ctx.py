"""tools/test_mem_vs_ctx.py - P124n's memory-vs-context check, tested without a GPU: the /proc parsing on fixed
inputs (a status file's VmLck/VmPin/VmRSS, the pid tree through a fake /proc whose comm carries spaces and
parens, and the port's owner found through a fake /proc/net/tcp by socket inode), the host-wide pinned delta
(/proc/meminfo Mlocked + Unevictable + Shmem read before the start and after /health ok - the GLM latent cache
of the 1.2.4 ref run, which no VmLck saw), the GPU reader on canned nvidia-smi / rocm-smi output, the engine-
log budget grep and the expert-placement parse (the engines' lines from ref/mortar-run-20261126.txt: Inkling's
resident-vs-offload flip must FAIL where its GPU totals were equal, GLM must show its pinned latent cache),
the run config copy (own session dir and log, no api_key, --max-context set), the table, and the PASS/FAIL
rules (memory figures under max(2%, 256 MiB), placement figures within 2%, pinned judged on the larger of the
two measures) on fixed figures - plus one real cycle against `python -m serve.server --engine mock`: the start
waits for /health ok, the port's owner is the server, and the stop frees the port again.

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


def fake_proc(tmp, pids, net_tcp=None, net_tcp6=None):
    """A /proc-shaped tree: <tmp>/<pid>/stat (ppid = the second figure after the comm), plus net/tcp and/or
    net/tcp6."""
    for pid, ppid, comm in pids:
        d = tmp / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194304 1 0 0 0 0 0 0 0 20 0 1 0 99\n",
                                encoding="ascii")
    for name, text in (("tcp", net_tcp), ("tcp6", net_tcp6)):
        if text is not None:
            (tmp / "net").mkdir(parents=True, exist_ok=True)
            (tmp / "net" / name).write_text(text, encoding="ascii")
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
                      "timeout inode\n" + tcp_line(8093, 555),
                      "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  "
                      "timeout inode\n" + tcp_line(9093, 777))     # (the v6 row sits in net/tcp6, where the
            fd = tmp / "77" / "fd"                                  #  reader actually looks for it)
            fd.mkdir()
            os.symlink("socket:[555]", fd / "3")      # (the listen socket: a dangling link reads back fine)
            os.symlink("socket:[777]", fd / "5")      # (and its v6 listener)
            os.symlink("/some/file", fd / "1")
            self.assertEqual(MV.port_owner(8093, str(tmp)), 77)
            self.assertEqual(MV.port_owner(9093, str(tmp)), 77)     # (found through net/tcp6)
            self.assertIsNone(MV.port_owner(8094, str(tmp)))

    def test_port_owner_established_is_not_a_listener(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            fake_proc(tmp, [(77, 1, "client")], "  sl  local_address x\n" + tcp_line(8093, 555, state="01"))
            fd = tmp / "77" / "fd"
            fd.mkdir()
            os.symlink("socket:[555]", fd / "3")
            self.assertIsNone(MV.port_owner(8093, str(tmp)))


class MeminfoTests(unittest.TestCase):
    # two fixed reads of /proc/meminfo around a GLM load (the 1.2.4 ref run): the engine page-locked its
    # latent cache the driver never showed in any VmLck, and the kernel's Unevictable rose by exactly it
    BEFORE = """MemTotal:       132435100 kB
MemFree:        42638884 kB
MemAvailable:   81609172 kB
Buffers:          312256 kB
Mlocked:               0 kB
Unevictable:      131072 kB
Shmem:           4260444 kB
HugePages_Total:       0
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
"""
    AFTER = """MemTotal:       132435100 kB
MemFree:        39527724 kB
MemAvailable:   78498012 kB
Buffers:          312256 kB
Mlocked:               0 kB
Unevictable:     3011931 kB
Shmem:           4260444 kB
HugePages_Total:       0
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
"""

    def test_parse_meminfo_mib_and_the_shmem_variants_stay_out(self):
        m = MV.parse_meminfo(self.BEFORE)
        self.assertEqual(m, {"Mlocked": 0.0, "Unevictable": 128.0, "Shmem": 4160.5898437500})
        m2 = MV.parse_meminfo("MemTotal: 100 kB\n")                  # absent keys count 0.0
        self.assertEqual(m2, {"Mlocked": 0.0, "Unevictable": 0.0, "Shmem": 0.0})

    def test_the_glm_latent_cache_shows_host_where_vmlck_read_zero(self):
        # the ref run's "the latent cache in 2.95 GB of pinned RAM": the host-wide delta sees 2.95 GB rise
        # while the tree's VmLck + VmPin stays 0.0 - this is the figure the fixed check now shows
        d = MV.host_pinned_delta(MV.parse_meminfo(self.BEFORE), MV.parse_meminfo(self.AFTER))
        self.assertAlmostEqual(d, 2.95e9 / 2**20, places=2)          # 2813.3 MiB, not 0.0

    def test_delta_sums_the_three_rises_and_clamps_falls(self):
        before = {"Mlocked": 0.0, "Unevictable": 0.0, "Shmem": 1000.0}
        after = {"Mlocked": 68.0 * 1024.0, "Unevictable": 1024.0, "Shmem": 900.0}   # Shmem fell: a neighbour
        # gave shared pages back - the check owns the moment, so a fall is not this run un-pinning anything
        self.assertAlmostEqual(MV.host_pinned_delta(before, after), 68.0 * 1024.0 + 1024.0)
        self.assertIsNone(MV.host_pinned_delta(None, after))
        self.assertIsNone(MV.host_pinned_delta(before, None))

    def test_read_meminfo_here_yes_missing_file_none(self):
        self.assertIsNotNone(MV.read_meminfo())
        self.assertIsNone(MV.read_meminfo("no-such-meminfo-here"))


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

    def test_set_context_keeps_the_one_flag_of_a_ctx_config(self):
        # (serve/models.py's ctx_flag: an ik2 config may name its context --ctx - the copy rewrites that
        #  flag, it never appends --max-context beside it and leaves the winner to the engine's mood)
        self.assertEqual(MV.set_context(["--ctx", "131072", "--threads", "8"], 262144),
                         ["--ctx", "262144", "--threads", "8"])
        self.assertEqual(MV.set_context(["--max-context", "131072", "--ctx", "1"], 262144),
                         ["--max-context", "262144", "--ctx", "1"])   # (max-context holds when both sit there)


class PlacementTests(unittest.TestCase):
    """The engines' expert-placement lines, as the 1.2.4 ref run (ref/mortar-run-20261126.txt) logged them."""

    GLM_32768 = ("glm fast: CUDA0 VRAM before the expert pool: 7.86 of 15.92 GB in use - dense weights 6.86, "
                 "state/KV (32768 ctx) 0.62, activations 0.10, other (context, MTP, scratch, prompt path) 0.28")
    GLM_262144 = ("glm pack: CUDA0 KV streaming: 32768 of 262144 positions of each of 11 DSA layers in VRAM "
                  "(0.37 GB), the latent cache in 2.95 GB of pinned RAM\n"
                  "glm fast: CUDA0 VRAM before the expert pool: 8.17 of 15.92 GB in use - dense weights 6.86, "
                  "state/KV (262144 ctx) 0.92, activations 0.10, other (context, MTP, scratch, prompt path) 0.29")
    INKLING_32768 = ("resident: every expert in VRAM (3072 copied in 7.2 s); tokens take the resident path, "
                     "a CUDA graph each")
    INKLING_262144 = ("offload: 0 of 3072 experts in VRAM (4.5 GiB), the rest fetched by the GPU from pinned "
                      "host memory through 1 staging pools (0.03 GiB); tokens take the resident path's "
                      "launches, a CUDA graph each")
    QFN_262144 = ("saguaro generate: KV streaming: 32768 of 262144 cells per QSA layer in VRAM, the K/V in "
                  "3.09 GiB of pinned RAM\n"
                  "saguaro generate: KV streaming: 3.35 GiB of pinned RAM for the K/V of every session at "
                  "--max-context 262144 (the solo session 3.09 GiB, the batch slots and the drafters 0.26 GiB)\n"
                  "FileExpertSource: mapped pinned cache complement ready: resident 68.00 GiB, pinned 68.00 "
                  "GiB in 19 s; 68.00 GiB page-locked after taking the pages in steps;\n"
                  "saguaro generate: resident RAM mode: 68.00 GiB of experts in RAM (page-locked), 1729 in "
                  "the GPU cache; adaptive swaps exchange them with the GPU cache (no file reads)")
    MM3 = "saguaro mm3: loaded in 171.2 s, 9216 of 9216 experts in VRAM, 0.0 GB pinned pool"
    DS4 = ("saguaro ds4: CUDA0 layers [0, 39]: weights 3.20 GB, experts 28.00 GB (3072 of 4096 in VRAM), "
           "caches 1.10 GB; saguaro ds4: 5916 routed experts on the CPU lane (72.5 GB of RAM, 2 NUMA parts), "
           "16 threads")
    HY4 = "INFO CPU lane: 14678 of 20992 routed experts in VRAM (70%), the rest on 32 CPU threads"

    def test_glm_shows_its_pinned_latent_cache_and_its_expert_pool(self):
        # the figure the VmLck missed shows here too, and the GLM streaming line's "11 DSA layers in VRAM"
        # is positions, not experts: it must not parse as one
        p32, p256 = MV.parse_placement(self.GLM_32768), MV.parse_placement(self.GLM_262144)
        self.assertIsNone(p32["pinned_declared_gib"])
        self.assertEqual(p256["pinned_declared_gib"], 2.95)
        self.assertIsNone(p256["experts_vram"])
        self.assertAlmostEqual(p32["expert_pool_gb"], 15.92 - 7.86, places=9)
        self.assertAlmostEqual(p256["expert_pool_gb"], 15.92 - 8.17, places=9)

    def test_inkling_resident_and_offload_are_two_different_placements(self):
        self.assertEqual(MV.parse_placement(self.INKLING_32768)["experts_vram"], 3072.0)
        self.assertEqual(MV.parse_placement(self.INKLING_262144)["experts_vram"], 0.0)

    def test_qfn_counts_the_gpu_cache_and_the_ram_tier_only(self):
        p = MV.parse_placement(self.QFN_262144)
        self.assertEqual(p["experts_vram"], 1729.0)
        self.assertEqual(p["experts_ram_gib"], 68.0)          # (the FileExpertSource 68.00 GiB lines stay out)
        self.assertEqual(p["pinned_declared_gib"], 3.35)      # (the larger of the two K/V lines)
        self.assertIsNone(p["expert_pool_gb"])

    def test_mm3_ds4_and_hy4_equivalents(self):
        self.assertEqual(MV.parse_placement(self.MM3)["experts_vram"], 9216.0)
        self.assertEqual(MV.parse_placement(self.MM3)["experts_ram_gib"], 0.0)      # (0.0 stated, not absent)
        p = MV.parse_placement(self.DS4)
        self.assertEqual(p["experts_vram"], 3072.0)
        self.assertEqual(p["experts_ram_gib"], 72.5)
        self.assertEqual(MV.parse_placement(self.HY4)["experts_vram"], 14678.0)

    @staticmethod
    def prow(context, placement, pinned=0.0, rss=40000.0, gpu=None):
        return {"context": context, "pinned_tree_mib": pinned, "pinned_host_mib": None, "rss_mib": rss,
                "gpu_mib": gpu, "budget": [], "placement": placement}

    def test_inkling_fails_where_the_equal_gpu_totals_hid_the_change(self):
        # the ref run's numbers: RSS and GPU inside max(2%, 256 MiB) (its PASS line) - the placement rule
        # is what turns that run into the FAIL it should have been
        rows = [self.prow(32768, MV.parse_placement(self.INKLING_32768), rss=13672.6, gpu=15716.8),
                self.prow(262144, MV.parse_placement(self.INKLING_262144), rss=13598.2, gpu=15406.6)]
        ok, why = MV.verdict(rows)
        self.assertFalse(ok)
        self.assertIn("experts in VRAM differs by 3072 between the contexts", why)

    def test_glm_fails_when_the_expert_pool_moved_more_than_two_percent(self):
        # 8.06 GB left for the experts at 32768, 7.75 at 262144: 4%, over the placement rule's 2%
        rows = [self.prow(32768, MV.parse_placement(self.GLM_32768)),
                self.prow(262144, MV.parse_placement(self.GLM_262144))]
        ok, why = MV.verdict(rows)
        self.assertFalse(ok)
        self.assertIn("expert pool (GB) differs by", why)

    def test_placement_within_two_percent_passes_and_a_missing_figure_is_not_a_change(self):
        ok, why = MV.verdict([self.prow(32768, {"experts_vram": 1000.0}),
                              self.prow(262144, {"experts_vram": 1020.0})])      # 20 <= 2% of 1020 = 20.4
        self.assertTrue(ok)
        self.assertIn("PASS", why)
        self.assertFalse(MV.verdict([self.prow(32768, {"experts_vram": 1000.0}),
                                     self.prow(262144, {"experts_vram": 1021.0})])[0])   # 21 > 20.42
        # one context states the figure and the other says nothing: nothing to compare, no FAIL
        self.assertTrue(MV.verdict([self.prow(32768, {"experts_vram": 1000.0}),
                                    self.prow(262144, {})])[0])
        # the placement difference outranks a memory one when it is further past its limit (the FAIL line
        # names the biggest breach): RSS 300 MiB over its 256 MiB floor while the GPU cache lost 1000 of 1024
        ok, why = MV.verdict([self.prow(32768, {"experts_vram": 1024.0}, rss=10000.0),
                              self.prow(262144, {"experts_vram": 24.0}, rss=10300.0)])
        self.assertFalse(ok)
        self.assertIn("experts in VRAM", why)

    def test_experts_in_ram_and_pinned_declared_are_compared_too(self):
        ok, why = MV.verdict([self.prow(32768, {"experts_ram_gib": 68.0}),
                              self.prow(262144, {"experts_ram_gib": 60.0})])
        self.assertFalse(ok)
        self.assertIn("experts in RAM differs by 8", why)
        ok, why = MV.verdict([self.prow(32768, {"pinned_declared_gib": 3.35}),
                              self.prow(262144, {"pinned_declared_gib": 30.0})])
        self.assertFalse(ok)
        self.assertIn("pinned RAM declared", why)

    def test_nothing_measured_or_parsed_anywhere_is_nothing_to_compare(self):
        rows = [{"context": c, "budget": []} for c in (32768, 262144)]     # no memory, no placement
        ok, why = MV.verdict(rows)
        self.assertFalse(ok)
        self.assertIn("nothing", why)

    def test_placement_summary_line_dashes_the_absent_figures(self):
        line = MV.placement_summary(self.prow(32768, MV.parse_placement(self.INKLING_32768)))
        self.assertIn("experts in VRAM 3072", line)
        self.assertIn("expert pool (GB) -", line)
        self.assertIn("pinned RAM declared (GB) -", line)


class TableAndVerdictTests(unittest.TestCase):
    @staticmethod
    def rows(pinned, rss, gpu=(None, None)):
        return [{"context": c, "pinned_tree_mib": p, "pinned_host_mib": None, "rss_mib": r, "gpu_mib": g,
                 "budget": []}
                for c, p, r, g in zip((32768, 262144), pinned, rss, gpu)]

    def test_table_shows_both_pinned_measures_and_dashes_without_a_reader(self):
        rows = self.rows([1000.0, 1010.0], [40000.0, 40500.0])
        for r, h in zip(rows, (3020.8, 3020.8)):
            r["pinned_host_mib"] = h
        table = MV.format_table(rows)
        lines = table.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("pinned host", lines[0])
        self.assertIn("pinned tree", lines[0])
        self.assertIn("GPU used MiB", lines[0])
        self.assertIn("3020.8", lines[1])
        self.assertIn("-", lines[1])
        self.assertTrue(lines[1].rstrip().endswith("-"))
        self.assertIn("40000.0", lines[1])
        # no /proc/meminfo and no GPU reader: both unpinned rows dash, the tree row still prints
        bare = MV.format_table(self.rows([1000.0, 1010.0], [40000.0, 40500.0])).splitlines()
        self.assertTrue(bare[1].split()[-2] == "40000.0" and bare[1].rstrip().endswith("-"))

    def test_pinned_is_judged_on_the_larger_of_the_two_measures(self):
        # the 1.2.4 ref run: every VmLck read 0.0 while the engine page-locked 68 GiB - the host-wide delta
        # sees it, so the pinned row must FAIL on the host figure even though the tree row is flat
        host_only = [{"context": c, "pinned_tree_mib": 0.0, "pinned_host_mib": h, "rss_mib": 40000.0,
                      "gpu_mib": None, "budget": []} for c, h in ((32768, 0.0), (262144, 69632.0))]
        ok, why = MV.verdict(host_only)
        self.assertFalse(ok)
        self.assertIn("pinned RAM differs by 69632.0 MiB", why)
        # the tree saw it and the host-wide read is flat (or absent): the same rule on the tree figure
        tree_only = [{"context": c, "pinned_tree_mib": t, "pinned_host_mib": None, "rss_mib": 40000.0,
                      "gpu_mib": None, "budget": []} for c, t in ((32768, 0.0), (262144, 69632.0))]
        ok, why = MV.verdict(tree_only)
        self.assertFalse(ok)
        self.assertIn("pinned RAM differs by 69632.0 MiB", why)
        # and it judges on the LARGER: the tree's 1000 stands inside the host's 1010
        both = [{"context": c, "pinned_tree_mib": t, "pinned_host_mib": h, "rss_mib": 40000.0,
                 "gpu_mib": None, "budget": []} for c, t, h in ((32768, 1000.0, 0.0), (262144, 0.0, 1010.0))]
        self.assertTrue(MV.verdict(both)[0])

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

    def test_a_difference_exactly_at_the_limit_fails(self):
        # (the report's "the boundary equal fails - less than", now actually on the boundary:
        #  10000 vs 10256 is exactly the 256 MiB floor (2% is 205.12), 39690 vs 40500 exactly the 2%)
        ok, why = MV.verdict(self.rows([10000.0, 10256.0], [40000.0, 40500.0]))
        self.assertFalse(ok)
        self.assertIn("pinned RAM differs by 256.0 MiB", why)
        ok, why = MV.verdict(self.rows([1000.0, 1010.0], [39690.0, 40500.0]))
        self.assertFalse(ok)
        self.assertIn("RSS differs by 810.0 MiB", why)
        self.assertTrue(MV.verdict(self.rows([10000.0, 10255.0], [40000.0, 40500.0]))[0])   # (one under: ok)

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
