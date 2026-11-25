"""tools/test_rename_upstream.py - the rename layer's one command (tools/saguaro/rename_upstream.py) and its shared
table (rename_rules.py): every family with a positive and a negative example, idempotence over a fixture tree,
the M1 hand-rename cases reproducing from upstream v1.1.78's text to our committed text, and the exception list
untouched.  No GPU, no network, no writes: the tool runs over in-memory files (plan_tree) and its --check over
the real tree.

    python -m unittest tools.test_rename_upstream -v
"""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "saguaro")]
import rename_rules as T          # noqa: E402
import rename_env as renv         # noqa: E402
import rename_upstream as ru      # noqa: E402


def F(rel, text, explicit=False):
    """one in-memory tree file, in collect()'s shape (the path need not exist: only the suffix is read)"""
    p = Path(rel)
    return rel, [p, renv.family_of(p), text, False, explicit]


def run(*files):
    """plan every family over the given files; returns (texts, changed)"""
    texts, changed, _ = ru.plan_tree(dict(files))
    return texts, changed


def upstream(rel):
    """v1.1.78's own text for a path (the M1 intake's input; the tag is local, git only)"""
    got = subprocess.run(["git", "show", f"v1.1.78:{rel}"], cwd=ROOT, capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    return got.stdout


def one(text, rel="x/y.md"):
    return run(F(rel, text))[0][rel]


class Settings(unittest.TestCase):
    def test_new_setting_caught_by_pattern(self):
        cpp = 'static const char* v = std::getenv("MAYA_SOMETHING_NEW");   // MAYA_SOMETHING_NEW in the comment\n'
        out = run(F("src/core/n.cu", cpp))[0]["src/core/n.cu"]
        self.assertIn('getenv("SAGUARO_SOMETHING_NEW")', out)
        self.assertIn("// SAGUARO_SOMETHING_NEW in the comment", out)     # the proven mention follows

    def test_strata_prefix_merges_into_the_one_name(self):
        cpp = 'const char* v = std::getenv("STRATA_OLD_THING");  // and STRATA_OLD_THING again\n'
        out = run(F("src/a.cu", cpp))[0]["src/a.cu"]
        self.assertIn('getenv("SAGUARO_OLD_THING")', out)
        self.assertNotIn("STRATA_OLD_THING", out)

    def test_config_key_renamed(self):
        out = run(F("configs/c.json", '{"env": {"MAYA_BENCH_LOCK_OK": "1"}}'))[0]["configs/c.json"]
        self.assertIn('"SAGUARO_BENCH_LOCK_OK"', out)

    def test_words_and_build_options_stay(self):
        cpp = '#define MAYA_EXPORT_API x\nint MAYA_HELPER_FN = 1;   // a bare identifier\n'
        cmake = 'option(MAYA_VISION_FULL "lean or full" OFF)\ntarget_compile_definitions(m PRIVATE MAYA_XOR_BPERMUTE)\n'
        sh = 'cmake --build build -DMAYA_BUILD_TESTS=ON\n'
        out = run(F("src/a.cpp", cpp), F("CMakeLists.txt", cmake), F("build.sh", sh))[0]
        self.assertEqual(out["src/a.cpp"], cpp)
        self.assertEqual(out["CMakeLists.txt"], cmake)      # build options and defines stay MAYA_*
        self.assertEqual(out["build.sh"], sh)               # a -DNAME= build flag stays
        self.assertEqual(one("MAYA_METAL example in prose\n", "docs/d.md"), "MAYA_METAL example in prose\n")


class Internal(unittest.TestCase):
    def test_internal_names_renamed(self):
        cpp = ('#include "strata/core.h"\nnamespace strata {\n'
               'void f() { strata::g(); }\nvoid* p = include/"strata";\n')
        out = run(F("src/a.cpp", cpp))[0]["src/a.cpp"]
        self.assertIn('#include "maya/core.h"', out)
        self.assertIn("namespace maya", out)
        self.assertIn("maya::g()", out)

    def test_pack_format_and_maya_line_kept(self):
        cpp = ('int strata_pack_size(); int strata_fmt(); int strata_kq();   // the Strata pack format stays\n'
               '#include "strata_fmt/x.h"\n')
        out = run(F("src/a.cpp", cpp))[0]["src/a.cpp"]
        self.assertEqual(out, cpp)

    def test_build_option_not_renamed_beside_the_maya_spelling(self):
        line = '#ifdef STRATA_USE_HIP   // this line also names MAYA_USE_HIP: the old spelling beside the new\n'
        out = run(F("src/a.cu", line))[0]["src/a.cu"]
        self.assertEqual(out, line)


class LogPrefixes(unittest.TestCase):
    def test_engine_prefixes(self):
        cpp = ('fprintf(stderr, "maya generate: hi\\n");\n'
               'fprintf(stderr, "ds4: no expert profile (%s)\\n", p);\n'
               'fprintf(stderr, "maya-vision: encoding\\n");\n'
               'fprintf(stderr, "maya-ds4: the lane\\n");\n')
        out = run(F("src/ds4/m.cu", cpp))[0]["src/ds4/m.cu"]
        self.assertIn('"saguaro generate: hi\\n"', out)
        self.assertIn('"saguaro ds4: no expert profile', out)
        self.assertIn('"saguaro vision: encoding\\n"', out)
        self.assertIn('"saguaro ds4: the lane\\n"', out)

    def test_not_prefixes_stay(self):
        cpp = ('fprintf(stderr, "strata:requests_total %f\\n", v);   // metric: no space after the colon\n'
               'const char* k = "strata.ple.format";\nfprintf(stderr, "ds4.txt: no such file\\n");\n'
               'usage(stderr, "maya generate --pack f ...\\n");\nfprintf(stderr, "bare strata here\\n");\n')
        import rename_log_prefixes as rlp
        self.assertEqual(rlp.rewrite(cpp)[0], cpp)            # the prefix rules alone leave every line
        out = run(F("src/b.cpp", cpp))[0]["src/b.cpp"]        # the other families leave them too
        self.assertEqual(out, cpp)

    def test_own_banner_outside_the_engine_sources_stays(self):
        cpp = 'std::puts("ds4 path policy: passed");\n'      # a test's own line, not an engine log
        out = run(F("tests/ds4_path_policy_test.cpp", cpp))[0]["tests/ds4_path_policy_test.cpp"]
        self.assertEqual(out, cpp)


class Names(unittest.TestCase):
    def test_program_and_launcher_names(self):
        md = ("run `python maya.py --bench` or `./maya.sh --bench`, see START-MAYA.bat;\n"
              "the files are maya-nightly.json, maya-nightly-status.txt and maya-nightly-glm-5.3.tokens.json;\n"
              'temp dirs: prefix="maya-vision-http-", PICTURE_DIR_PREFIXES ("maya-vision-", "maya-ds4-pictures-",'
              ' "maya-hy4-pictures-", "maya-mm3-pictures-")\n')
        out = one(md, "docs/d.md")
        self.assertIn("python saguaro.py", out)
        self.assertIn("./saguaro.sh", out)
        self.assertIn("START-SAGUARO.bat", out)
        self.assertIn("saguaro-nightly.json", out)
        self.assertIn("saguaro-nightly-glm-5.3.tokens.json", out)
        self.assertIn('"saguaro-vision-http-"', out)
        self.assertIn('"saguaro-ds4-pictures-"', out)
        self.assertIn('"saguaro-hy4-pictures-"', out)
        self.assertIn('"saguaro-mm3-pictures-"', out)

    def test_c122_config_name_placeholders(self):
        """C122: the generic config-name placeholder follows the saguaro- family; the concrete maya-*.json file
        names a running install points at stay (config_names finds both families)."""
        md = ("start it with `saguaro.py --config maya-<model>.json`; its run-maya-<model>.sh (run-maya-<model>.bat\n"
              "on Windows) does the same; the configs maya-ds4.json and maya-mm3.json of older installs stay\n")
        out = one(md, "docs/d.md")
        self.assertIn("--config saguaro-<model>.json", out)
        self.assertIn("run-saguaro-<model>.sh", out)
        self.assertIn("run-saguaro-<model>.bat", out)
        self.assertIn("maya-ds4.json", out)
        self.assertIn("maya-mm3.json", out)

    def test_server_log_tag_and_launcher_switch(self):
        out = run(F("serve/s.py", 'print(f"[maya] the GPUs freed")\n'))[0]["serve/s.py"]
        self.assertIn("[saguaro]", out)
        out = run(F("src/g.cu", '// maya --speed-calibration times it again\n'))[0]["src/g.cu"]
        self.assertIn("saguaro.py --speed-calibration", out)

    def test_names_that_stay(self):
        text = ('exe glob "maya-*.json": maya-ds4.json, maya-mm3.json, maya-hy4-flash-maya-s.json, .maya-switch.json;\n'
                "cmake --build build-vision --target maya-vision; targets maya_kernels, maya_mmq; build/MAYA-BUILD.json;\n"
                'localStorage.getItem("maya." + k); serve --engine maya; engine_kind=ds4; maya_vision.cpp;\n'
                "./serve-maya.sh status; tools/maya_nightly.py and maya_golden.py keep their names; `maya.shard`-like"
                " words and myamaya.py too\n")
        self.assertEqual(one(text, "docs/d.md"), text)

    def test_exceptions_are_reported_not_rewritten(self):
        texts, changed = run(F("src/hy4/hy4_main.cpp", "// HY4_FETCH_MEASURE=1 or maya --speed-calibration measures\n"))
        self.assertEqual(texts["src/hy4/hy4_main.cpp"].count("maya --speed-calibration"), 1)
        self.assertEqual(changed, {})


class Brand(unittest.TestCase):
    def test_user_visible_docs(self):
        out = one("Maya checks the Installer's path. Put the folder on PATH and restart Maya.\n", "docs/SETUP.md")
        self.assertIn("Saguaro checks the Installer", out)
        self.assertIn("restart Saguaro.", out)

    def test_credits_versions_and_quants_stay(self):
        text = ("based on Project Maya (github.com/mw00/project-maya), tested on Maya v1.1.78 and stock Maya v1.1.5;\n"
                "the Maya quant Maya-S24 beside Maya-M, Maya-L; Saguaro starts when you start Saguaro.\n")
        self.assertEqual(one(text, "docs/d.md"), text)

    def test_brand_is_docs_only(self):
        out = run(F("saguaro.py", 'warn("fix it, then start Maya again")\n'))[0]["saguaro.py"]
        self.assertIn("start Maya again", out)          # product strings wait for a product pass (notes.md C)


class FixtureTree(unittest.TestCase):
    def fixture(self):
        return (
            F("src/ds4/model.cu",
              'const char* v = std::getenv("MAYA_DS4_SLOT_KEEP");   // MAYA_DS4_SLOT_KEEP here too\n'
              'fprintf(stderr, "ds4: slot dir %s\\n", v);\n'),
            F("serve/server.py", 'print("[maya] starting the engine")\n'
                                 'P = ("maya-vision-", "maya-ds4-pictures-")   # see maya.py\n'),
            F("docs/R.md", "All three sit on `./maya.sh --bench`; the file is maya-nightly.json.\n"
                           "restart Maya to apply it. Project Maya and Maya v1.1.78 stay; Maya-S too.\n"),
            F("CMakeLists.txt", "option(STRATA_VISION_FULL \"\" OFF)\nadd_executable(maya src/program/generate.cpp)\n"
                                "#include-dir strata comment: strata:: and include/strata/x.h\n"),
            F("src/a.cpp", '#include "strata/core.h"\nnamespace strata { int strata_pack(); }\n'),
            F("configs/maya-x.json", '{"env": {"MAYA_BENCH_LOCK_OK": "1"}}\n'),
            F("serve/web/lib.js", 'localStorage.setItem("maya." + k, v);\nprint("[maya] in js");\n'))

    def test_every_family_fires_where_it_should(self):
        texts, changed = run(*self.fixture())
        self.assertIn('getenv("SAGUARO_DS4_SLOT_KEEP")', texts["src/ds4/model.cu"])
        self.assertIn("// SAGUARO_DS4_SLOT_KEEP here too", texts["src/ds4/model.cu"])
        self.assertIn('"saguaro ds4: slot dir', texts["src/ds4/model.cu"])
        self.assertIn('print("[saguaro] starting the engine")', texts["serve/server.py"])
        self.assertIn('("saguaro-vision-", "saguaro-ds4-pictures-")', texts["serve/server.py"])
        self.assertIn("# see saguaro.py", texts["serve/server.py"])
        self.assertIn("`./saguaro.sh --bench`", texts["docs/R.md"])
        self.assertIn("saguaro-nightly.json", texts["docs/R.md"])
        self.assertIn("restart Saguaro", texts["docs/R.md"])
        self.assertIn("Project Maya and Maya v1.1.78 stay; Maya-S too", texts["docs/R.md"])
        self.assertIn("option(MAYA_VISION_FULL", texts["CMakeLists.txt"])       # STRATA_ build option -> MAYA_
        self.assertIn("add_executable(maya ", texts["CMakeLists.txt"])          # the engine target keeps its name
        self.assertIn('#include "maya/core.h"', texts["src/a.cpp"])
        self.assertIn("namespace maya", texts["src/a.cpp"])
        self.assertIn("int strata_pack()", texts["src/a.cpp"])                  # the pack format keeps its name
        self.assertIn('"SAGUARO_BENCH_LOCK_OK"', texts["configs/maya-x.json"])  # the key renames, the file name stays
        self.assertIn('"maya." + k', texts["serve/web/lib.js"])                 # the browser key stays
        self.assertIn('print("[saguaro] in js")', texts["serve/web/lib.js"])
        for rel in ("src/ds4/model.cu", "serve/server.py", "docs/R.md", "CMakeLists.txt", "src/a.cpp",
                    "configs/maya-x.json", "serve/web/lib.js"):
            self.assertIn(rel, changed)

    def test_idempotent_over_the_fixture_tree(self):
        texts, _ = run(*self.fixture())
        again = {rel: [ent[0], ent[1], texts[rel], False, False] for rel, ent in dict(self.fixture()).items()}
        texts2, changed2, _ = ru.plan_tree(again)
        self.assertEqual(changed2, {})
        self.assertEqual(texts2, texts)


class M1FromV1178(unittest.TestCase):
    """the M1 hand renames, reproduced by the tool alone: upstream v1.1.78's text in, our committed text out"""

    FILES = ["src/core/vram_late.hpp", "serve/test_vision_cache.py", "serve/server.py", "docs/SETUP.md",
             "docs/RELEASE-CHECKS.md", "src/ds4/ds4_model.cu", "src/core/glm_fast_path.cu", "src/core/glm_window.cu",
             "src/kernels/cuda/glm_fast.cu"]     # the last one proves MAYA_GLM_DSA_SCORE2 the setting it is (02c95f94a)

    @classmethod
    def setUpClass(cls):
        texts, _ = run(*(F(rel, upstream(rel)) for rel in cls.FILES))
        cls.texts = texts

    def line(self, rel, needle, text):
        return next((l for l in text.splitlines() if needle in l), None)

    def test_whole_files_that_renamed_cleanly(self):
        for rel in ("src/core/vram_late.hpp",        # b2e85aa52: the settings lines (SAGUARO_VRAM_LATE_* et al.)
                    "serve/test_vision_cache.py"):   # fe22cd559: the maya-vision-http- temp prefix
            with self.subTest(rel):
                self.assertEqual(self.texts[rel], (ROOT / rel).read_text())

    def test_settings_lines_match_our_commit(self):
        for rel, needle in (("src/core/vram_late.hpp", "SAGUARO_VRAM_LATE_MB=0 turns this off"),
                            ("src/ds4/ds4_model.cu", 'std::getenv("SAGUARO_DS4_PROMPT_SKIP")'),
                            ("src/core/glm_window.cu", "SAGUARO_GLM_DSA_SCORE2's kernel")):   # the 02c95f94a comment
            with self.subTest(rel, needle=needle):
                self.assertEqual(self.line(rel, needle, self.texts[rel]), self.line(rel, needle, (ROOT / rel).read_text()))

    def test_log_prefix_lines_match_our_commit(self):
        for rel, needle in (("src/ds4/ds4_model.cu", "saguaro ds4: expert profile %s ignored"),        # 02c95f94a
                            ("serve/server.py", "[saguaro] starting the engine")):                     # ef27c2650
            with self.subTest(rel, needle=needle):
                self.assertEqual(self.line(rel, needle, self.texts[rel]), self.line(rel, needle, (ROOT / rel).read_text()))

    def test_name_and_brand_lines_match_our_commit(self):
        for rel, needle in (("docs/RELEASE-CHECKS.md", "sit on `./saguaro.sh --bench`"),               # b2e85aa52
                            ("docs/RELEASE-CHECKS.md", "`saguaro-nightly.json` in the checkout"),
                            ("docs/SETUP.md", "Saguaro checks the Installer"),                          # 2d4246a26
                            ("docs/SETUP.md", "restart Saguaro"),
                            ("src/core/glm_fast_path.cu", "saguaro.py --speed-calibration marks")):    # a4a45bd71
            with self.subTest(rel, needle=needle):
                self.assertEqual(self.line(rel, needle, self.texts[rel]), self.line(rel, needle, (ROOT / rel).read_text()))

    def test_idempotent_on_the_upstream_text(self):
        again = {rel: [Path(rel), renv.family_of(Path(rel)), self.texts[rel], False, False] for rel in self.FILES}
        texts2, changed2, _ = ru.plan_tree(again)
        self.assertEqual(changed2, {})
        self.assertEqual(texts2, self.texts)


class RealTree(unittest.TestCase):
    def test_check_passes_over_the_worktree(self):
        got = subprocess.run([sys.executable, "tools/saguaro/rename_upstream.py", "--check"],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(got.returncode, 0, got.stdout[-2000:] + got.stderr[-2000:])
        self.assertIn("CHECK OK", got.stdout)


if __name__ == "__main__":
    unittest.main()
