# saguaro-hy4 on 4x T4: install and run (Hy4-Flash-Maya-S-IQ2_S, the fast config)

Numbers and their runs: bench/hy4/RESULTS.md. What is still open: NEXT-hy4.md.

## 1. Files
- Engine: `build-hy4*/saguaro-hy4` (target `maya-hy4`) -> `/opt/llm/saguaro/bin/saguaro-hy4`.
- Server: this branch's `serve/` and `tools/` (hy4_vision.py, the hy4 template / tokenizer support) -> `/opt/llm/saguaro/`.
- Expert placement scores: `data/hy4-expert-hot-Hy4-Flash-Maya-S.bin` -> `/opt/llm/saguaro/data/` (the config's HY4_HOT_FILE).
- Config: `configs/Hy4-Flash-Maya-S-IQ2_S.json` -> `/opt/llm/saguaro/configs/`.
- Model (read in place): `/opt/llm/models/Hy4-Flash-Maya-S-IQ2_S/{Maya-S,Maya-XS,mtp}/`.

## 2. One-time steps
Pack (an index of the GGUF files + the tokenizer, 5 MB, seconds, no GPU):

    M=/opt/llm/models/Hy4-Flash-Maya-S-IQ2_S
    cd /opt/llm/saguaro && .venv/bin/python tools/hy4/pack_gguf.py --model $M --out $M/pack \
      --files Maya-S/Hy4-Flash-Maya-S-IQ2_S-00001-of-00002.gguf Maya-XS/Hy4-Flash-Maya-XS-IQ2_XXS-00002-of-00002.gguf \
      --mmproj Maya-XS/mmproj-Hy4-Flash-Maya-XS.gguf --mtp mtp/Hy4-Flash-mtp-BF16.gguf_file \
      --drafter Maya-S/Hy4-Flash-Maya-S-drafter.dfly2
    mkdir -p /opt/llm/var/Hy4-Flash-Maya-S-IQ2_S/sessions

(The pack lists the MTP block and the DFly2 drafter; neither is loaded by the config - see RESULTS.md for why.)

## 3. The config's choices
- `--tp 4`: tensor parallel over the four cards (each holds a share of every layer and a whole copy of the cache;
  the host adds the partial sums). Decode 32 tok/s against 18 for the layer split; reads about 2x.
- `--chunk 2048 --reserve-gb 0.5`: 85% of the routed experts in VRAM; 0.3 failed once at warm-up with a larger chunk.
- `HY4_HOT=1` + `HY4_HOT_FILE`: tensor parallel places the experts by these scores at the start (without HY4_HOT=1 it
  ignores the file and places by index: 21 tok/s instead of 30+). The file is frozen; relearn with a layer-split start
  (`--tp 0`) on a NEW file path over real traffic, then copy it.
- `--kv auto`: takes 8-bit cache rows by itself under tensor parallel with the CPU lane.
- `HY4_Q4X=0`: the CPU lane's 4-bit expert copies need AVX2.
- `HY4_SLOT_DIR` (engines from the saved-conversations commit on): conversations set aside when another chat comes
  and at unload, read back when a request continues one. `HY4_SLOT_GB` caps the folder (the box's NVMe is tight).
- No drafter: `--mtp` gives no decode gain under tensor parallel here and costs reads; `--dfly2` cannot load on this layout.
- THE CLOCK LOCK IS REQUIRED: without `llm-gpu-clocks lock` decode is 7.6 tok/s instead of 32.

## 4. Proposed llama-swap profile (modelled on the MiniMax block; add the name to the same `groups:` member list)

      # Hy4-Flash (Tencent Hy License, LICENSE.hy4-flash), Project Maya's Maya-S GGUF quant, on saguaro-hy4.
      "Hy4-Flash-Maya-S-IQ2_S":
        cmd: >
          /bin/sh -c "sudo -n /usr/local/sbin/llm-gpu-clocks lock >/dev/null 2>&1; sudo -n prlimit --pid $$ --memlock=unlimited:unlimited; cd /opt/llm/saguaro && exec .venv/bin/python serve/server.py --config /opt/llm/saguaro/configs/Hy4-Flash-Maya-S-IQ2_S.json --host 127.0.0.1 --port ${PORT}"
        cmdStop: /bin/sh -c "sudo -n /usr/local/sbin/llm-gpu-clocks unlock >/dev/null 2>&1; kill -TERM ${PID}"
        ttl: 900
        unloadTimeout: 120
        env:
          - "SAGUARO_ENGINE_WRAP_S=0"

Load to READY is about 20 s warm (the 49 GB trunk shard in the page cache), minutes cold. The first start with the
config's env also times the prompt route on ~57K tokens of docs before READY unless `HY4_MEASURE=0` /
an empty `HY4_MEASURE_IDS` is set: check llama-swap's health-check timeout, or set `HY4_MEASURE=0` in the config's env
for the first start.

## 5. By hand from the worktree (nothing under /opt written)

    cd /data/llm/saguaro-hy4port
    sudo -n /usr/local/sbin/llm-gpu-clocks lock; sudo -n prlimit --pid $$ --memlock=unlimited:unlimited
    /opt/llm/saguaro/.venv/bin/python serve/server.py --config build-hy4/fast.json --host 127.0.0.1 --port 8099
    curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "model":"Hy4-Flash-Maya-S-IQ2_S","max_tokens":200,"messages":[{"role":"user","content":"Say hello in five words."}]}'

`build-hy4/fast.json` is the config on worktree paths (pack: build-hy4/pack-check; state: build-hy4/var/).
Engine alone, timed: `build-hy4/bench/b.sh LABEL short:200,10k:200 -- --max-context 131072 --tp 4 --chunk 2048 --reserve-gb 0.5`.

Success lines: `INFO tensor parallel: every layer over GPUs 0,1,2,3 ...`, `INFO HY4_HOT: tensor parallel places the
experts in VRAM at the start (by .../hy4-expert-hot-Hy4-Flash-Maya-S.bin)`, `INFO CPU lane: N of 20992 routed experts
in VRAM (85%)`, `INFO saguaro-hy4: Hy4-flash, 42 layers ... loaded in N s`, then `READY 131072 stop`.

## 6. Limits to know
- One conversation computes at a time (no --parallel in this engine); others wait in the server's queue.
- Thinking: on (high) or off; `reasoning_effort: "none"` turns it off.
- Tensor parallel's answers differ from the layer split's in the last bits, and this 2-bit quant amplifies that to a KL
  of ~0.057 between any two layouts / chunk sizes (RESULTS.md's KL table); each layout repeats its own answers.
- Pictures: the vision tower runs BF16 / FP32 GEMMs without tensor cores on sm_75 (src/hy4/hy4_vision.cu:350, :652).
