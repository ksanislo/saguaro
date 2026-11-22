# saguaro/hy4-up: upstream #1122 / #1100 / #1084 / #1026 for saguaro-hy4 (2026-11-22)

Branch saguaro/hy4-up off saguaro/qfn-vram 95fb5638. Engine: build-hy4up/saguaro-hy4 (CUDA 13.1, sm_75).
Card tests are HELD (four-card Hy4, Ken). Lines below are for the queue: one line each, SAGUARO_PCIE_SAFE=1 on every
arm, 131K only (never 262K+).

## What changed for a run
- Token windows under --tp (#1122): int8 rows written by the kernel before each GEMV (HY4_Q8_FUSE, on; our HY4_FUSE
  only acts with HY4_Q8_FUSE=0), host sums streamed as parts come (HY4_TP_SUM_STREAM), publish / pull on 3 blocks,
  shared-expert part added into the routed partial before it goes out, new Q8_0 GEMV forms (SAGUARO_IQ_*), FFN fork.
  The L2 read-ahead (HY4_PREFETCH) is probed at the start with 8/16/24 MB ranges: a T4's L2 is 4 MB, so expect
  "INFO tensor parallel: no weights read ahead" - the part of upstream's +30% that came from it does not apply here.
  lat_pre (int8 rows from the experts' sum) is off while the CPU lane runs (our layout). So expect less than +30%.
- #1026: with --reserve-gb given, the engine measures what it allocates after the plan and keeps it free from the next
  start on, in <pack>/vram-late.txt (here: build-hy4/pack-check). That changes the plan from start to start, so every
  arm below fixes it: SAGUARO_HY4_LATE_MB=0 (old build ignores it). T1 measures it into a log file instead.
- #1100: under --tp no peer access is turned on any more (it was enabled between neighbours though unused).

## Held test lines (from /data/llm/saguaro-hy4port/build-hy4/bench)
    cd /data/llm/saguaro-hy4port/build-hy4/bench; export HY4_NO_CLAIM=1
    N=/data/llm/saguaro-hy4-up/build-hy4up/saguaro-hy4; O=/data/llm/saguaro-hy4port/build-hy4c/saguaro-hy4
    A="--max-context 131072 --tp 4 --chunk 2048 --reserve-gb 0.5"; P="--env SAGUARO_PCIE_SAFE=1"

T1 (new build: reads with the switch, PCIe per card, the #1026 figure; the serve line "streamed:" must NOT appear):

    cd /data/llm/saguaro-hy4port/build-hy4/bench && export HY4_NO_CLAIM=1 && (nvidia-smi dmon -s tu -o T -d 1 > logs/pcie-up-safe.dmon 2>&1 & echo $! > logs/pcie-up-safe.pid) && timeout 1800 ./b.sh up-safe-1 10k:50,10k:50,60k:50,short:300 --exe /data/llm/saguaro-hy4-up/build-hy4up/saguaro-hy4 --env SAGUARO_PCIE_SAFE=1 --env SAGUARO_HY4_LATE_FILE=/data/llm/saguaro-hy4port/build-hy4/bench/logs/vram-late-hy4up.txt -- --max-context 131072 --tp 4 --chunk 2048 --reserve-gb 0.5 > logs/nq-up-safe-1.out 2>&1; kill $(cat logs/pcie-up-safe.pid); echo done > logs/nq-up-safe.done

   Check: grep -c 'streamed:' logs/up-safe-1.log = 0; rxpci in the dmon; "hy4: CUDA" late lines; read 10K/60K tok/s.

T2 (ABAB old vs new, the same tokens and the speed-up; order new, old, old, new):

    cd /data/llm/saguaro-hy4port/build-hy4/bench && export HY4_NO_CLAIM=1 && for a in up-n1:/data/llm/saguaro-hy4-up/build-hy4up/saguaro-hy4 up-o1:/data/llm/saguaro-hy4port/build-hy4c/saguaro-hy4 up-o2:/data/llm/saguaro-hy4port/build-hy4c/saguaro-hy4 up-n2:/data/llm/saguaro-hy4-up/build-hy4up/saguaro-hy4; do n=${a%%:*}; x=${a#*:}; timeout 1800 ./b.sh $n short:200,10k:200,v1:200 --exe $x --env SAGUARO_PCIE_SAFE=1 --env SAGUARO_HY4_LATE_MB=0 -- --max-context 131072 --tp 4 --chunk 2048 --reserve-gb 0.5 > logs/nq-$n.out 2>&1; done; echo done > logs/nq-up-abab.done

   Read: decode tok/s short / 10K / v1 and the 10K read per arm (results.jsonl), md5 per request: new == old means
   the same tokens. If they differ, compare the "experts in VRAM" counts of the two builds' logs first (new buffers
   can move experts to the CPU lane, which changes rounding - the head-split case); equal counts + different md5 = a
   real difference, then rerun one pair with HY4_Q8_FUSE=0 HY4_TP_SUM_STREAM=0 to bisect.

T3 (optional, after T2: our HY4_FUSE under the new code, only meaningful with upstream's fusion off):

    cd /data/llm/saguaro-hy4port/build-hy4/bench && export HY4_NO_CLAIM=1 && timeout 1800 ./b.sh up-q8off short:200,10k:200 --exe /data/llm/saguaro-hy4-up/build-hy4up/saguaro-hy4 --env SAGUARO_PCIE_SAFE=1 --env SAGUARO_HY4_LATE_MB=0 --env HY4_Q8_FUSE=0 -- --max-context 131072 --tp 4 --chunk 2048 --reserve-gb 0.5 > logs/nq-up-q8off.out 2>&1; timeout 1800 ./b.sh up-q8off-fz short:200,10k:200 --exe /data/llm/saguaro-hy4-up/build-hy4up/saguaro-hy4 --env SAGUARO_PCIE_SAFE=1 --env SAGUARO_HY4_LATE_MB=0 --env HY4_Q8_FUSE=0 --env HY4_FUSE=1 -- --max-context 131072 --tp 4 --chunk 2048 --reserve-gb 0.5 > logs/nq-up-q8off-fz.out 2>&1; echo done > logs/nq-up-fz.done

## Before installing
- The config should get SAGUARO_HY4_LATE_MB only if the measured figure (T1) is unwanted; otherwise the pack's
  vram-late.txt is written by the engine into the live model folder on the first start.
- The server's restart on "INFO reserve_short=" is upstream b94be5d72 (serve/server.py, shared, DS4's #1026): not on
  this branch (saguaro/upstream-u2 brings it). Without it the engine only reports the shortfall.
