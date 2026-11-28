"""P125q: the elastic expert tail's mlock.  (1) The systemd service docs' unit carries LimitMEMLOCK=infinity and the
section says why.  (2) saguaro.sh raises the soft memory-lock limit to the hard one at start and says ONCE, with the
one-line fix, when even the hard limit is under 8 GiB (qfn_fit's smallest tier) - the logic runs here through fake
uname/ulimit shells, and a non-Linux start touches nothing.

    python -m unittest tools.test_setup_memlock -v
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SETUP = (ROOT / "docs" / "SETUP.md").read_text(encoding="utf-8")
SH = (ROOT / "saguaro.sh").read_text(encoding="utf-8")
RAISE = re.search(r"raise_memlock\(\) \{.*?\n\}", SH, re.S)


class UnitText(unittest.TestCase):
    def section(self):
        m = re.search(r"## Running Saguaro as a systemd service\n(.*?)(?=\n## )", SETUP, re.S)
        self.assertIsNotNone(m, "the systemd service section of docs/SETUP.md")
        return m.group(1)

    def test_the_unit_line_is_in_the_service_section_of_the_unit(self):
        sec = self.section()
        unit = re.search(r"```ini\n(.*?)```", sec, re.S).group(1)
        svc = unit[unit.index("[Service]"):unit.index("[Install]")]
        self.assertIn("LimitMEMLOCK=infinity", svc)          # (the fix itself, inside [Service])
        self.assertIn("saguaro.sh", svc)                     # (the unit runs saguaro.sh - the start step 2 wraps)

    def test_the_section_says_why(self):
        sec = self.section()
        self.assertIn("Memory-lock limit.", sec)             # (the docs say so, not only the unit)
        self.assertIn("mlock failed (raise ulimit -l)", sec)  # (the line the user sees without it)
        self.assertGreaterEqual(sec.count("LimitMEMLOCK=infinity"), 2)   # (the unit AND the bullet)


class SaguaroShRaises(unittest.TestCase):
    """raise_memlock's logic, run through a scripted `sh`: uname/ulimit are fake shell functions; every
    `ulimit -Sl VALUE` the function makes is logged, and its whole stdout is the warning (if any)."""

    def run_raise(self, os_name, soft, hard):
        self.assertIsNotNone(RAISE, "raise_memlock() { ... } in saguaro.sh")
        with tempfile.TemporaryDirectory() as t:
            log = Path(t, "calls")
            script = (
                "uname() { echo " + os_name + "; }\n"
                "ulimit() {\n"
                "  case \"$1\" in\n"
                "    -Sl) if [ \"$#\" -gt 1 ]; then echo \"set $2\" >> '" + str(log) + "'; else echo " + soft + "; fi ;;\n"
                "    -Hl) echo " + hard + " ;;\n"
                "  esac\n"
                "}\n" + RAISE.group(0) + "\nraise_memlock\n")
            p = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30)
            calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout, calls

    def test_the_script_calls_it_at_start(self):
        after = SH.split(RAISE.group(0), 1)[1].split("\n", 2)[1]   # the line right after the definition
        self.assertTrue(after.startswith("raise_memlock"), "saguaro.sh calls raise_memlock beside the definition")

    def test_a_non_linux_start_touches_nothing(self):
        out, calls = self.run_raise("Darwin", "64", "8192")
        self.assertEqual(calls, [])
        self.assertEqual(out, "")

    def test_a_soft_limit_under_the_hard_one_is_raised_to_it(self):
        out, calls = self.run_raise("Linux", "64", "unlimited")    # (the defaults'-era 64 KiB soft, unlimited hard)
        self.assertEqual(calls, ["set unlimited"])                 # (soft -> hard, what prlimit wraps do)
        self.assertEqual(out, "")                                  # (unlimited hard: never too low, nothing said)

    def test_a_hard_limit_under_the_smallest_tier_warns_once_with_the_fix(self):
        out, calls = self.run_raise("Linux", "64", "8192")         # (new systemd's 8 MiB hard: unraisable)
        self.assertEqual(calls, ["set 8192"])
        self.assertIn("8192 KiB", out)                             # (the hard limit, as ulimit -l shows it)
        self.assertIn("ulimit -l", out)
        self.assertIn("memlock unlimited", out)                    # (the one-line fix: a limits.conf line...)
        self.assertIn("/etc/security/limits.conf, then log out and in", out)
        self.assertIn("LimitMEMLOCK=infinity", out)                # (... and the systemd one)
        self.assertEqual(out.count("hard memory-lock limit"), 1)   # (said ONCE)

    def test_a_hard_limit_at_or_over_the_smallest_tier_says_nothing(self):
        for hard in ("8388608", "74448896"):                       # (8 GiB - qfn_fit's max(8, RAM - spare) - up)
            out, calls = self.run_raise("Linux", "64", hard)
            self.assertEqual(calls, ["set " + hard])               # (still raised to the hard limit)
            self.assertEqual(out, "")

    def test_just_under_the_smallest_tier_warns(self):
        out, _ = self.run_raise("Linux", "8388607", "8388607")     # (8 GiB minus 1 KiB, soft == hard)
        self.assertIn("8388607 KiB", out)

    def test_an_unlimited_soft_and_hard_says_and_does_nothing(self):
        out, calls = self.run_raise("Linux", "unlimited", "unlimited")
        self.assertEqual(calls, [])
        self.assertEqual(out, "")


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
