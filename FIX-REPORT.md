# FIX-REPORT: P124a review (REVIEW-RESULT.md) - finding by finding

Both FIX-FIRST findings and every cheap item landed, one commit per finding; the two items that are
not fixable in this environment are declined with their reasons below.

| finding | where |
| --- | --- |
| F1 HIGH - the credited floor was dead: no release, unsaved ones included, could ever fire | `e91547e17` |
| F2 MEDIUM - the card test could never print PASS | fixed in `scratch/card-test-park.sh` - no commit: `scratch/` is gitignored (.gitignore:20) and the plan left the script untracked |
| F3 MEDIUM - the ~3 s are measured, not yet explained-and-removed | unchanged - declined here, it is the manager's card run (recorded below) |
| F4 LOW - the batched wait removes the per-copy watchdog beats | unchanged - declined with the reason below |
| F5 LOW - the save reported the LAST copy failure, not the first | `7f105758f` |
| F6 LOW - build proof covers sm_75 only | unchanged - no toolchain here can build the others (recorded below) |
| F7 LOW - the restore line kept the arithmetic lie the park line was fixed for | `dd6f00328` |
| F8 INFO - the new drop text was unreachable on the box it was written for | fixed by `e91547e17` - it is reachable again, no separate change |
| F9 INFO - `own_pinned` credited an unready tier | `4d81a7bc0` |
| F10 NIT - REPORT cited `...20261225.log` | `22ed0dd63` |

## HIGH - F1. The hard floor is live again: the credit stops at `severe_bytes`.

`pressure_need` now credits `own_pinned` only while the RAW reading is at or above the hard floor
(park_policy.hpp:88, 96-103): under `severe_bytes` the raw figure is asked back, exactly the pre-credit
reading. So a raw 1 GiB with 68 GiB pinned asks 17 GiB, `pressed` goes true, the tick passes its
gate, `plan()` reaches its `hard && why == pressure` branch and unsaved entries go - and the pooled
trim, gated on `pressed`, lives too. `severe()` moved above `pressure_need` (it is now shared). The
review's own probe re-run says it:

    deep (raw 1 GiB, own 68 GiB): need=18253611008 severe=1 actions=3     (before the fix: need=0 actions=0)
    same raw reading, no credit:   need=18253611008 severe=1 actions=3

Above the floor the credit works exactly as the mortar case needs: 8 GiB raw + 68 GiB own is still
no need and `plan()` still releases nothing; the new check adds 3 GiB raw - one GiB over the hard
floor - and says the same. The tick gate (generate.cpp:18221-18226) weighs `pk::severe` outright as
well, so the gate says in code what the rule now is. The deep test asserts what the review asked:
`plan()` at raw 1 GiB yields `1d:pressure 3d:pressure 2u:unsaved` - the message "unsaved entries
still go" is finally behind a call to `plan()`. The header's and REPORT §3's guarantee ("a host that
is genuinely full still releases") is now what the code does. `SAGUARO_PARK_FREE_MIB=0` still
returns 0 at the first guard, before any credit - the feature-off path is byte-identical.

## MEDIUM - F2. The PASS grep.

`scratch/card-test-park.sh:108` is `grep -qE "restored [0-9]+ tokens"` with a comment: the engine
logs the conversation's real token count, the script's N only approximates it, so the line - not the
number - is the criterion. The FAIL grep (:106) was verified to match, unchanged. This is the only
finding with no commit: the file lives under `scratch/`, which is gitignored - the plan and the
review both keep the card test untracked there. `bash -n` says the edited script parses.

## MEDIUM - F3. The remaining 2-3 s ride on the card, as the review itself concludes.

The cause is named and its ours-part removed (batched wait, one round-trip instead of ~200); the
seconds themselves are contention behind the cut request's writer traffic, which cannot be measured
without a GPU and is what the new `checkpoints`/`wait` fields will quantify on the manager's first
card run (REPORT §5). Not papering over it with a speculative second fix (the double park+turn-save
share is a separate design change the REPORT leaves open on purpose).

## LOW - F4. Declined for this round: a beat inside the batched wait would beat blind.

#2049's beats fire when bytes actually move; a stream sync exposes no progress to beat on, so a
periodic `progress_beat()` in that wait would keep the watchdog alive through a genuinely stalled
stream - the one thing the beats exist to let show - and an event-poll loop adds CUDA/HIP-dependent
spin logic to a path only the card can validate. The wait is now a single *measured* segment: if the
card shows it long enough to trip the 60 s limit, the poll-wait belongs in then, with evidence. The
per-copy beats of the non-queued paths are untouched.

## LOW - F5. The first failing copy is the one named.

`conversation_checkpoint_save` collects the copy errors into a local and hands the FIRST one to the
caller (conversation_state.cpp:194-209); the loop still runs on and the stream is still waited for
before every return, so the "caller may free `c` on failure" contract holds. The wait-failure
message can no longer overwrite a copy message and vice versa.

## LOW - F6. Unprovable here, stated plainly.

This box has one CUDA toolchain - 13.1 - and `nvcc --list-gpu-arch` starts at `compute_75`
(`-arch=sm_61`: "Unsupported gpu architecture"); there is no `hipcc`. The P40 and ROCm paths build
on the manager's older toolchains, where this environment cannot reach them. The risk the review
already rates low is unchanged by this round: every new line is host-only C++ (a `?:` in a header,
two arithmetic/temporary locals, one `std::string`) using nothing the changed files did not already
use. What IS proven: the sm_75 CUDA build is green after all five code commits - full incremental
`cmake --build scratch/build-p124a`: 444/444, `saguaro-qfn` linked (the one warning is the
pre-existing unused `w` at generate.cpp:15580, untouched by this diff).

## LOW - F7. The restore line times its whole restore too.

The switch section measures `restore_ms` once, uses it for both the INFO line and the log line, and
passes it to `conversation_snapshot_take_split(restore_ms)` - the restore line ends with
`wait N ms` and its split can no longer exceed the total printed beside it. The split reset also
moved from the switch section's start to the incoming restore's own start (before its clock), so a
borrow's or a slot copy's records cannot ride in the restore's split - the same lie, opposite
direction, closed.

## INFO - F8. Reachable again.

The branch fires precisely when F1 made `pressed` possible-with-credit: a raw reading under the hard
floor releases while `own_pinned > 0`, and a mortar-class box now logs `pressure: 1.0 GiB free,
short even crediting the engine's own 68.0 GiB pinned`. No change of its own.

## INFO - F9. Only a ready tier is credited.

`park_tick`'s credit (generate.cpp:18213-18215) is `src.complement_ready() ? src.resident_bytes() :
0` for the file source - the guard the prefill host plan uses - so during a pin or after a failed
one the credit is what is actually pinned, not the planned size. The arena branch already reported
its held bytes.

## NIT - F10. REPORT:4 cites `ref/mortar-park-lines-20261125.log`, as the file and the plan say.

## Runs and touches

* `park_policy_test` - the one test module the plan names - run twice, one module at a time:
  standalone (g++ -std=c++20) and as the `park_policy_test` target of the scratch build: **34 checks
  passed** both times (32 before, +2 from the deep case and the 3 GiB case). Nothing was run on a GPU.
* Files in the five commits (`git diff --name-only c5f8b8c84..HEAD`): park_policy.hpp,
  park_policy_test.cpp, generate.cpp, conversation_state.cpp, REPORT-p124a-park.md. No commit
  touches `serve/`, `serve/qfn_engine.py` or `tools/saguaro/extract_qfn_engine.py` - none of the
  fixes is server-side, so no re-extraction is owed. No Co-Authored-By / Claude-Session /
  Assisted-by lines. Scratch files (`park_policy_test` binaries, the probe) live under `scratch/`
  only and nothing there is committed.
