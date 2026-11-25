# REVIEW-RESULT - v1.2.1-D setup fixes (diff d708f396..ca5a159e)

MERGE-WITH-NITS - all six steps do what the plan says; the OFF path is byte-identical (verified by a
base-vs-HEAD byte compare, below), the expert count is read not assumed, the merged-GGUF check is as strict as
the plan's rule, the stop condition matches the engine's own requirement exactly, and no High/Medium finding
turned up. Six nits and three informational notes follow.

## Findings

Low

1. `--context 0` says the wrong thing. The scale-down line gates on `getattr(a, "context", None) is not None`
   (saguaro.py:8097) but `choose_context` returns the given context only `if a.context` (saguaro.py:1871-1872).
   With `--context 0` on a slow layout the setup drops the "a 32K context instead of 64K" entry and claims
   "the context you named with --context 0: kept" (saguaro.py:8098-8102) while `--max-context` is written as
   the scaled-down default. Nonsense input; a wrong line, not a wrong config for any sane value. Make the
   gate `if getattr(a, "context", None)` to match `choose_context`.
2. The `ExpertProfileShapes` fixture never swaps the shipped profile. `QFN_PROFILE` is bound at import
   (saguaro.py:715), so patching `maya.ROOT` (tools/test_setup_qfn.py:264) leaves it on the real
   `data/expert-profile-qfn.bin`; the file the test writes at tools/test_setup_qfn.py:260-261 is never read and
   the `shipped=` parameter is dead. The tests pass because the real file is 48x512 - so the branch of the stop
   message at saguaro.py:7916-7917 ("...and the shipped expert-profile-qfn.bin is 48 x 512" vs "is not here",
   and a shipped file of a third shape) is never exercised. Patch `maya.QFN_PROFILE` in the test.
3. `MergedGguf.test_one_flipped_sample_tensor_refuses_it` (tools/test_setup_qfn.py:449-458) mutates the shared
   class table: `ts = [list(x) for x in self.TS]` is a shallow copy, so `ts[2][2][0] = 99.0` rewrites `TS[2][2]`
   for the rest of the run. Harmless today (the flipped tensor goes into both sides of every later test, and
   alphabetical order runs this one last), but one reorder or one new later test turns it into a phantom pass or
   fail. Deep-copy the value lists.

Nit

4. Dead `import hashlib` in `qfn_merged_gguf` (saguaro.py:7442) - the hashing lives in `gguf_tensor_slice`
   (saguaro.py:7423).
5. `comfortable`'s docstring cites "UD-Q4_K_XL's 111 GiB" (tools/qfn_fit.py:284); the manifest sums 103.7 GiB
   (the report's own 104 is right). Comment only - the waiver compares against `MIN_RAM_GB`, not the prose.
6. The two `--yes` warnings say "--yes answers the download question no" (saguaro.py:7597-7598, 7743-7744) even
   when the operator answered the prompt no interactively (`offer_download` returns False either way). The flag
   gets named, which is what item 4 asked for; a "(or you answered no)" phrasing would cover both.

Informational

7. The headless-Pascal 1024 is general, not QFN-only: `gpu_reserve_mib` also feeds the GLM HIP env
   (saguaro.py:3887), the Maya paths (saguaro.py:3957, 6599, 7191) and line 9196's runtime fallback, so a
   headless all-Pascal PC now writes 1024 (1 GiB less expert pool) for every family. The plan's wording is
   general ("On all-sm_6x cards use 1024 when headless") and the measurement is a property of the card, so this
   reads as intended - flagging it because it changes configs of the other engines on such a PC too.
8. Without the index on disk the merged file is accepted on its own self-consistency only (names, shapes,
   types, total bytes via `incomplete`, saguaro.py:7477-7486) - the file's quant identity is then whatever it
   declares. That is the plan's explicit weaker rule, said with "not sha256-checked" in the ok line.
9. The same "vision tower was not downloaded" wording exists in the DS4 and Inkling steps (saguaro.py:6030,
   8746) and was left as it was - item 4's two warnings are the QFN ones, so out of scope, noted for the next
   pass.

## The brief's targeted checks

Byte-identical OFF path - verified with a byte compare, not by reading. I extracted the base versions of the
three changed code files (saguaro.py, tools/qfn_fit.py, tools/ik2_fit.py) into ./scratch/, loaded base-saguaro
with base-qfn_fit (pre-seeded into `sys.modules` so the calibrate_* path inserts cannot pull the new fit in)
and HEAD in two processes, and byte-compared the configs `write_config_qfn` writes (paths and `written`
timestamps normalised) plus `QF.plan` over a matrix (6 card sets x RAM {0,31,60,88,96,220} x 5 quants). Result
(scratch/cmp/{base,head}): identical for ud-q4_k_xl and iq3_s at 220 GB RAM, ud-q4_k_xl at 31/60 (big quants
stay slow), the coder pack whose index does not read (the pre-(f) rule), headless T4 splits, Pascal with a
display (3072), mixed Pascal+T4 headless (512), and `--vram-reserve-mib 2048` on a P40 (the flag still wins).
The only differences are exactly the four intended changes: the coder's `--expert-profile`/`--expert-profile-save`
pair and the var/ copy drop (220 GB) or the setup stops before writing (31/60 GB tier); iq3_s/coder at 31/60 GB
lose the scale-down (32768->65536 context, 4096->32768 kv-resident, drafting not off); P40 headless moves
512->1024 MiB. In the plan matrix, plan output differs in 20 cells, all of them RAM {31,60} x {gsq-iq3_s,
gsq-coder-iq1_m} x five card sets - every small-card cell and every big-quant cell is byte-identical, i.e. the
v1.2.1-B matrix's other points stand as the plan requires.

Expert count read, not assumed - yes. `qfn_profile_shape` parses the header the engine itself parses
(saguaro.py:7802-7813 vs engines/qfn/src/core/expert_cache.cpp:89,111-112 - "STRP", u32 version/layers/experts,
refused as "a profile for a different artifact"); `qfn_expert_shape` takes the quant's width from the pack's
index.txt field 8 = ne1 (saguaro.py:7816-7838), which is exactly the column tools/iq_pack.py:245,257 writes and
the rule tools/iq_pack.py:430 states ("router rows = experts kept"); layers from the max `blk.N.`. The Coder's
256 comes from the file; nothing hard-codes it.

Stop rule vs the engine - the proxy is exact, not broad: `--resident-budget-gib` itself sets
`resident_cpu_experts` (engines/qfn/src/program/generate.cpp:3173-3177), and that option is refused without a
static `--expert-profile` (generate.cpp:3450-3453) - so a written tier is precisely the case where a config
without the profile cannot start, and saguaro.py:7910 stops with a plain line naming the file, both shapes and
the make_profile command (saguaro.py:7912-7920; verified live in the byte-compare run: "is not there, and the
shipped expert-profile-qfn.bin is 48 x 512 ... make a matching profile first: ... --n-expert 256 --out ...").
A stale 48x512 copy is left in place and unread (saguaro.py:7908).

Merged-GGUF strictness - with the pinned shards on the PC (in --gguf-dir, the old config's files_dir, or the
model folder, saguaro.py:7460-7474) the merged file must match tensor for tensor on names, shape AND type_name
(saguaro.py:7487-7497), so a different quant - which differs per expert tensor by block type - cannot pass even
with copied names; then the first, last and 8 spread sampled tensors' bytes must hash equal to the shards'
(saguaro.py:7498-7507), which catches a same-type file with different weights. Without the index nothing can be
compared and the plan's weaker rule applies, disclosed in the ok line with "not sha256-checked"
(saguaro.py:7477-7486). Zero or several candidates keep the old refusal word for word (saguaro.py:7439-7441;
tools/test_setup_qfn.py test_two_unknown_ggufs), a single-file-pinned quant never enters the path
(saguaro.py:7436-7437), and the accepted file skips the pinned-sha256 loop by returning before it
(saguaro.py:7537-7545, asserted via a `sha256_checked` that throws in the tests).

Reserve rule, display cards - unchanged, verified byte-identical: Pascal with a display keeps 3072, a mixed
headless list keeps 512, the APU branch keeps its precedence (saguaro.py:1044-1049), and
`--vram-reserve-mib` is read before the rule (saguaro.py:7936-7938). `gpu_sm6x` reads AMD arch strings as
not-Pascal and an unread arch as no (saguaro.py:1026-1035).

Generated file - the whole diff touches no file under engines/ or serve/ (serve/qfn_engine.py absent from
`git diff --stat`), so no extract_qfn_engine.py run was owed. The plan's Don'ts hold: vram_elastic / pictures
logic and RAM_SPARE_GB are untouched (no diff line).

Commits - six code commits, one per numbered step, each touching only its step's files; zero
Co-Authored-By/Claude-Session/Assisted-by lines; REPORT-*.md force-added as in the repo's long-standing
convention (git log --all -- REPORT-*.md).

Tests really exercise the change - the new `MergedGguf` cases build real GGUF v3 files (fake_gguf,
tools/test_setup_qfn.py:361-378) so header, alignment, tensor directory and data offsets are all parsed by the
real gguf_reader; the profile tests write real STRP headers and real index.txt rows; the reserve and fit tests
call the real functions. I re-ran the six named modules, one at a time, with
/opt/llm/saguaro/.venv/bin/python -m unittest: tools.test_qfn_fit 44 OK; tools.test_setup_qfn 98 OK 1 skip;
tools.test_setup_ds4 81 OK; tools.test_setup_amd 35 OK; tools.test_setup_ik2 54 OK; tools.test_maya_quants 25
OK 1 skip - exactly the report's counts. The report's baseline claim for test_exl3_on_amd holds structurally
(the old test asserted `said == []` while the old code warned, tools/test_setup_ds4.py diff).

Locks, threads, stdout - the diff adds no threading and no locks (nothing to nest); everything runs on the
setup's single thread, and the new output (the profile ok line saguaro.py:7924-7927, the merged-file ok line
saguaro.py:7540-7541, the reserve why at :7942-7944) is plain one-shot say/ok calls - no interleaving surface.
`qfn_merged_gguf`'s sampling reads at most 10 tensors with no output in between.

Paths and secrets - no path outside the repo is written (the config lands under the patched/real ROOT, the
merged check only reads user-named folders); no secret or token appears anywhere in the diff.
