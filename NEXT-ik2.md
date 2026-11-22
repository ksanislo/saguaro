# saguaro-ik2 (Inkling 2 Mini on T4s): what remains, in order

State: branch saguaro/ik2, worktree /data/llm/saguaro-ik2port, build dir build-ik2 (binary = commit 80422cc0's
engine). Baseline binary of before the token-path work: build-ik2/basebin/saguaro-ik2. Results: build-ik2/bench/*.out.
Python for every tool: /opt/llm/saguaro/.venv/bin/python -B (system python3 lacks 'regex').
GPU rules: every card run and every build under one line in /tmp/gpu-status (tools/ik2/bench.py and serve_check.py
claim by themselves; builds: tools/ik2/claim.sh build-x nice ninja -C build-ik2 -j20 saguaro-ik2). No build while timing.

Measured so far (greedy, 256-token answers, clocks NOT pinned, cards 2 / 2,3):
  1 card, ctx 32768, --chunk 1024        before (cache path)   now, default pins     SAGUARO_IK2_PIN=0 (all cached)
    decode, ~1K deep                      37.9 tok/s            40.7 (fast=0: 33.1)    42.1 (fast=0: 36.3)
    decode, ~8-9K deep                    31.8                  37.4 (fast=0: 29.6)    39.0 (fast=0: 32.0)
    decode, 30K deep                      25.8                  not measured           not measured
    read 8K / 30K                         419-466 / 406-412     492 / -                428 / -
    first token, 52-token prompt          0.87-1.68 s           0.95 s                 1.63 s (first request of a load)
  2 cards, ctx 131072                     before (t4_check)     now (by layers)
    decode, ~1K deep                      37-39 (7 tokens!)     47.9 (fast=0: 38.4)
    decode, ~8-9K deep                    -                     44.7 (fast=0: 33.9)
    read 8K                               820 (2K prompt)       844
  Load 8-10 s. VRAM: 1 card 14.94 GiB; 2 cards 15.44 + 8.13 GiB (GPU 0 is nearly full at 131072: see 2).
  fast=1 vs fast=0 greedy tokens part after 41-185 tokens (upstream's tensor-core experts / attention are not bit
  equal to the scalar path: its docs give KL 5.7e-5); fast=0 twice is 256 of 256.

1. Finish the 1-card layout choice (the variants queue was cut short; only pin0 ran):
     P="/opt/llm/saguaro/.venv/bin/python -B tools/ik2/bench.py"
     for e in SAGUARO_IK2_PIN=core SAGUARO_IK2_PIN_KEEP=0.5 SAGUARO_IK2_PIN_KEEP=0.7 SAGUARO_IK2_GRAPH=0 SAGUARO_IK2_EXPERT_MMA=0; do
       $P --gpus 2 --ctx 32768 --depths 0,8000 --ab 2 --arg=--chunk --arg=1024 --env $e --label v1-$e --out build-ik2/bench/fast.jsonl; done
   Compare "ab depth ~" lines (fast=1 column) and the "plain rep 0" read tok/s against build-ik2/bench/fast1a.out
   (default pins) and v1-pin0.out. pin0 decodes faster (42.1 / 39.0 vs 40.7 / 37.4) but reads slower (428 vs 492) and
   one load each is inside the box's ~11% drift: repeat each arm 2-3 times interleaved before changing the default.
   The default is set in the pin loop of main(), src/ik2/ik2_main.cu:4703 (keep 0.3; order = core then smaller layers).
2. Two cards at depth: 131072 was only loaded, never filled. GPU 0 had ~150 MiB free with the 256 MiB room, so
   configs/Inkling-2-Mini-UD-Q4_K_XL.json now sets SAGUARO_IK2_ROOM_MB=1024 (UNTESTED with this value). Run
     $P --gpus 2,3 --ctx 131072 --depths 0,8000,30000,125000 --think --env SAGUARO_IK2_ROOM_MB=1024 --label d2
   and the 1-card depth run:  $P --gpus 2 --ctx 32768 --depths 0,8000,30000 --think --arg=--chunk --arg=1024 --label d1
   If it holds, make the room rule the code default (src/ik2/ik2_main.cu:4685: GPU 0 of two -> 1024 MiB).
3. Clock pinning: not measured. Same two commands as 2 with --lock added, interleaved with unlocked runs.
4. Read speed: a read is bound by decoding every expert of every visit to FP16 once a chunk, so larger chunks should
   read faster (2 cards at --chunk 2048 read 844 tok/s, 1 card at 1024 reads 492). Measure
     for ch in 2048 4096; do $P --gpus 2 --ctx 32768 --depths 8000 --gen 128 --arg=--chunk --arg=$ch --label r1-c$ch; done
     for ch in 4096 8192; do $P --gpus 2,3 --ctx 65536 --depths 16000 --gen 128 --arg=--chunk --arg=$ch --label r2-c$ch; done
   then set --chunk in the two configs. For where a read's time goes add a per-request print of g_pf under
   SAGUARO_IK2_PROF in the serve loop (next to "decode stats", src/ik2/ik2_main.cu:5344) and run with
   --env SAGUARO_IK2_PROF=1: that also answers the cp.async question (the attention's share of a read on sm_75).
5. Serve end to end with the real engine (never run): tools/ik2/serve_check.py --gpus 2
   (starts serve/server.py on port 18977 with a config under build-ik2/serve-check/, makes the tokenizer folder there,
   checks /health, thinking chat, thinking off, streaming, tool call + result turn, stop, a picture through
   build-ik2-vision/build/bin/vision-glm, and saves /health /metrics /props /slots /v1/status as JSON there).
   Then compare those JSON files' fields with what serve/web/*.js reads for the Monitor (the engine prints no STAT
   lines: see how src/mm3/mm3_main.cpp prints "INFO ... experts_vram= experts=" and STAT, serve/engine_stats.py).
6. The hot-set file (SAGUARO_IK2_HOT=FILE, src/ik2/ik2_main.cu:307): built, never run. Test: two loads with
   --env SAGUARO_IK2_HOT=build-ik2/bench/hot.bin, compare the first request's "decode stats" misses of load 2 vs load 1.
7. Left out on purpose / ideas, none started:
   - 1 card still copies 6-14 experts a token (33-70 MB): a CPU lane for the token path's cached layers (NVFP4 rows
     exist in src/ik2/ik2_cpu.cpp; Q8_0 rows do not), or a smaller quant of layers 20-23 (see 8).
   - a short prompt (18-54 tokens) takes 0.4-1 s to read on 1 card: profile it (4).
   - several conversations at once and saved conversations: the engine has one conversation (hist + 8 snapshots in
     the serve loop, src/ik2/ik2_main.cu:4905); it would need per-conversation K/V + conv histories (Visit state,
     src/ik2/ik2_main.cu:242) and a slot table in the serve loop; saving = those buffers to a file.
   - Web: the per-chat Turbo chip still shows for ik2 (serve/web/chat-view.js); the Models page cannot switch TO an
     ik2 config (serve/models.py missing() wants --pack).
8. Quant proposal (sizes only, nothing built). One T4 needs experts <= ~11.7 GiB to keep all resident at 32K
   (15.4 free - dense 1.1 - small 0.2 - K/V 1.3 - work 0.8 - room 0.25); the file's are 14.71 GiB.
     layers 20-23 Q8_0 -> NVFP4-size (4.5 bpw): 4.03 -> 2.13 GiB, experts 12.8 GiB: still 1.1 over
     layers 20-23 -> IQ4_XS (4.25 bpw) + layers 0-5,18,19 -> IQ3_XXS (3.06 bpw): experts ~11.3 GiB: fits, ~0.4 GiB spare
     (IQ3_XXS / IQ4_XS / Q5_K / Q6_K experts are types the engine already reads; a KL check against the file is needed.)
