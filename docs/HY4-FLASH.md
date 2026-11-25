# Hy4-Flash

**[Hy4-Flash](https://huggingface.co/tencent/Hy4-flash)** (Tencent; 220 B parameters, a mixture of experts with 512
experts a layer of which 16 run a token, and an n-gram table the model looks up for every token) runs on its own
engine, `saguaro-hy4`, from [Project Maya's GGUF quants](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF). It is
made for **four cards**. The model is under the **Tencent Hy License** ([LICENSE.hy4-flash](../LICENSE.hy4-flash):
derived from Apache 2.0, with added terms for large Model-as-a-Service businesses).

Back to [the models](MODELS.md) and the [README](../README.md).

## The files

| Size · setup id | Trunk | Experts | VRAM + RAM that hold it |
| --- | ---: | --- | ---: |
| Maya-XS · `hy4-flash-maya-xs` | 44.8 GB | 2 bits | 52 GB |
| **Maya-S** · `hy4-flash-maya-s` | 53.0 GB | 2 bits on layers 1-23, 3 bits on 24-41 | 64 GB |
| Maya-M · `hy4-flash-maya-m` | 68.4 GB | 3 bits, 4 on 7 layers | 76 GB |
| Maya-L · `hy4-flash-maya-l` | 89.5 GB | 4 bits, 5 on 5 layers | 100 GB |

Every size also needs the **n-gram table** (63.4 GB, one file for all four: a second size links to it and downloads
nothing twice) and takes the vision tower (0.8 GB). The table stays on the SSD: the engine reads its rows page by page
while the model answers. Maya-S is the size measured here; the others install the same way and have not been run on
this project's hardware.

## What it needs

| | |
| --- | --- |
| **GPU** | NVIDIA, Linux. Four equal 16 GB cards (four Tesla T4s) are what it was measured on. Everything that is not a routed expert - attention, the head, the cache: about 10 GB at a 32K context, 11.5 GB at 131K - has to fit in VRAM; the experts fill the rest and the others run on the CPU. |
| **RAM** | Enough for VRAM + RAM to reach the size's figure above (Maya-S: 64 GB), plus the system's own. |
| **Disk** | An NVMe SSD with 117 GB for Maya-S (trunk, table, vision tower). The table is read on every token: no spinning disk. |

On one or two cards it starts, with most of its experts on the CPU; that was not measured, and
[Inkling 2 Mini](INKLING2-MINI.md) is the model for such a PC.

## Install

```sh
git clone https://github.com/ksanislo/saguaro.git && cd saguaro
./setup.sh --model hy4-flash-maya-s
```

`--model hy4-flash` asks which size (Maya-S recommended where the memory holds it). The setup:

1. checks the PC and takes all the cards (`--gpus 0,1,2,3` names them yourself);
2. compiles `build/saguaro-hy4`;
3. shows the download (117 GB for Maya-S from `peasantsmith/Hy4-Flash-Maya-GGUF`, pinned to a commit) and starts it
   after you answer `y` (`--download-model` skips the question). It resumes where it stopped and checks each file's
   sha256;
4. builds the pack - the engine's index of the model files, a few MB, no weights copied - the same as
   `python tools/hy4/pack_gguf.py --model <folder> --files <trunk> <table> --mmproj <mmproj> --mtp none --drafter none`;
5. writes `saguaro-hy4-flash-maya-s.json` and starts the dashboard on `http://127.0.0.1:8080`.

On four equal 16 GB cards the config is the measured one: `--tp 4 --chunk 2048 --reserve-gb 0.5`, a 131K context and
`HY4_HOT=1`, which places the experts in VRAM by the scores shipped in `data/` for Maya-S, and `HY4_VISION_F16=1` (the
vision tower's GEMMs on FP16 tensor cores: a T4 has no BF16 ones). Any other set of cards gets
the engine's own choices (`--tp auto`, `--chunk auto`) at a 32K context.

**Tesla T4s: lock the GPU clocks before you start.** While a card waits its turn it idles its clock down; with the
layers split across the four cards the model wrote 7.6 tok/s unlocked and 15.4 locked here, and every figure below is
with them locked:

```sh
sudo nvidia-smi -lgc 1590,1590     # undo: sudo nvidia-smi -rgc
```

The cards' power cap still applies, so this cannot overheat them.

From the web app: **Models › Download a model › Hy4-Flash · Maya-S**. The row says what it needs and how it fits this
PC's cards.

Files you already have go in with `./setup.sh --model hy4-flash-maya-s --model-dir /path/to/folder`: under their
published names in that folder, or in the repo's own subfolders (`Maya-S/`, `Maya-XS/`) as a plain download of the
repo leaves them.

## Speed

Maya-S on four Tesla T4s (16 GB, sm_75) with two Xeon E5-2660 v2, greedy 200-token answers, a 131K context, GPU clocks
locked (`bench/hy4/RESULTS.md` has every row):

| | 4x Tesla T4 |
| --- | ---: |
| Writing an answer | about 32 tok/s (31 at 60K tokens deep) |
| Reading a prompt | about 590 tok/s (10K tokens), 460 (60K) |
| Load | about 20 s once the files are in the page cache; the first load reads 49 GB from the SSD |
| Experts in VRAM | 85% (placed by the learned scores: the ones the answers route to most) |

One run each, with other work on the box at times: differences under about 3% are noise. How it got there, from the
first load's 7.6 tok/s: locked clocks (15.4), tensor parallel over the four cards (20.8), the learned placement (30.3),
a smaller prompt chunk that leaves more VRAM to the experts (32.5).

## What it does

- **Thinking:** High (the default) or Off - the two levels the model is made for.
- **Tool calls**, the OpenAI- and Anthropic-compatible API, streaming and stop, as for the other models.
- **Pictures:** the vision tower runs in the engine and takes no VRAM of its own. On four Tesla T4s a 621-token picture
  prompt is read in 2.7 s with `HY4_VISION_F16=1` (3.1 s on the BF16 path), and the answers name its shapes, colours
  and text correctly; a picture already in the conversation is not read again.
- **Engine settings on the Models page:** context, GPUs, tensor parallel (Auto, Off, or the group's size), the VRAM
  kept free, the prompt chunk, CPU threads, the cache's rows (Auto, 8-bit, F16) and pictures. The engine stops at a
  flag it does not know, so the page writes only these.
- **The Monitor** shows how many experts are in VRAM and how many the CPU computes, the experts a token took from the
  CPU and the n-gram table's read time.

## Limits

- **One conversation at a time.** A second request waits for the first.
- **Saved conversations are off unless you name a folder:** add `"HY4_SLOT_DIR": "<folder>"` (and `"HY4_SLOT_GB"`, 64
  by default) to the config's `env`. A conversation set aside by another, or at a restart or model switch, is then
  read back instead of read again: 10K tokens saved in 0.3 s (0.36 GB) and read back in 0.5-1.0 s on four T4s.
  Conversations with pictures are not saved.
- **No drafter is installed, and no Turbo.** On four Tesla T4s the model's MTP block was no faster (32-34 tok/s against
  32-33) and read prompts 35% slower; the DFly2 drafter does not load under tensor parallel. Both remain by hand
  (`--mtp`, `--dfly2`; the repo has the files).
- **Quality of this layout:** against the layer split with F16 cache rows, tensor parallel with 8-bit rows has a KL
  divergence of 0.06 and picks the same next token 88% of the time. Any change of rounding order moves this 2-3 bit
  quant by that much - a different chunk size alone does (0.059) - so it is the quant's floor, not a fault of the
  layout.
- **`SAGUARO_PCIE_SAFE=1`** (off by default): for a PC that resets when all its cards stream weights from host
  memory at the same time. Under tensor parallel a prompt read normally streams the experts that are not in VRAM to
  all four cards at once; with the switch they run on the CPU instead. It costs prompt-read speed (not yet measured;
  estimated up to a quarter), nothing while writing. Add it to the config's `env` only if your PC needs it.
- **Switches of upstream's tensor-parallel token windows (#1122, all on; `0` turns one off, the same tokens either
  way):** `HY4_PREFETCH` (the next kernels' dense weights read into each GPU's cache while its partial is out for the
  sum; measured per GPU at the start, a number forces the bits; AMD: on by default with gfx1100, off with other AMD
  GPUs, `-1` measures everywhere), `HY4_SUM_PER` (2048: the fewest values of a sum a host thread takes; AMD builds with
  the CPU lane leave it unset and run a token window's sums on one thread - its helpers would spin on the lane's
  cores), `HY4_PF_BLOCKS`, `HY4_PF_ATTN_EARLY`, `HY4_PF_SMALL`,
  `HY4_PULL_BLOCKS` / `HY4_PUB_BLOCKS` (3), `HY4_TP_SUM_STREAM`, `HY4_Q8_FUSE` (kernels write the next GEMV's int8
  rows; our `HY4_FUSE` applies only with it off), `HY4_FFN_FORK`, `HY4_EG_EARLY`, `HY4_GEMV56`, `HY4_GEMV_AHEAD`,
  `HY4_GROUP_PAR2`, `HY4_RMS_AHEAD`, `SAGUARO_IQ_MV_REG` (+ `_MIN`, `_THREADS`), `SAGUARO_IQ_MV56`,
  `SAGUARO_IQ_NE56`, `SAGUARO_IQ_DN_RW4`, `SAGUARO_IQ_Q8_WIDE`,
  `SAGUARO_IQ_MV_RW` (on with AMD gfx1100: the attention heads' short Q8_0 / Q6_K rows packed 2 or 4 to a warp, the
  same sums; `0`: one row a warp, `1`: on with any GPU); off: `HY4_MTP_Q4K`, `HY4_LOGIT_HASH` (debugging).
- **VRAM reserve held for the whole run (#1026):** with `--reserve-gb` given, what a start and its requests allocate
  after the plan is measured and kept free beside the reserve from the next start on (the pack's `vram-late.txt`);
  `SAGUARO_HY4_LATE_MB=<MiB>` fixes the figure (`0`: the reserve alone), `SAGUARO_HY4_LATE_FILE` names another file.
- **Hand-offs between GPUs (layer split, #1100):** a peer copy only where it is offered, the IOMMU does not translate
  and a test copy arrives right, else through pinned host memory (`HY4_HOST_STAGE=1` always, `HY4_STAGE_MB` its
  pieces). Tensor parallel turns no peer access on.
- AMD: compiles for HIP (gfx1201 checked) but has not run on an AMD GPU yet, so the setup does not offer it there ([AMD_SAGUARO.md](AMD_SAGUARO.md#what-runs-on-amd)).
  Windows and Mac: not built.

## Running it by hand or under a proxy

The setup's start script does this; by hand, from the Saguaro folder (`<config>` = the setup's
`saguaro-hy4-flash-maya-s.json`, or `configs/Hy4-Flash-Maya-S-IQ2_S.json`, the measured 4x T4 config with its paths
relative to the Saguaro folder - the model in `models/Hy4-Flash-Maya-S-IQ2_S/`, its state in `var/`):

```sh
sudo nvidia-smi -lgc 1590,1590           # Tesla T4s (above)
ulimit -l unlimited                      # or a memlock limit large enough for the pinned expert copies
.venv/bin/python serve/server.py --config <config> --host 127.0.0.1 --port 8099
```

The pack, if you built the model folder yourself (an index of the GGUF files and the tokenizer, a few MB, no GPU):

```sh
M=<models folder>/Hy4-Flash-Maya-S-IQ2_S
.venv/bin/python tools/hy4/pack_gguf.py --model $M --out $M/pack \
  --files Maya-S/Hy4-Flash-Maya-S-IQ2_S-00001-of-00002.gguf Maya-XS/Hy4-Flash-Maya-XS-IQ2_XXS-00002-of-00002.gguf \
  --mmproj Maya-XS/mmproj-Hy4-Flash-Maya-XS.gguf --mtp none --drafter none
```

Under a proxy that starts models on demand (llama-swap and the like), run the same command with the proxy's port and
give it a stop timeout of a minute or two: the server saves open conversations when it stops. The first start times the
prompt route on ~57K tokens before it is ready unless `"HY4_MEASURE": "0"` is in the config's `env` - allow for that in
the proxy's health-check timeout. A good start logs `tensor parallel: every layer over GPUs 0,1,2,3`, `HY4_HOT: tensor
parallel places the experts in VRAM at the start`, `CPU lane: N of 20992 routed experts in VRAM`, then `READY`.

`HY4_HOT_FILE`'s scores are fixed; to learn them for your own traffic, run a layer-split start (`--tp 0`) with a new
file path over real use, then point the config at it. `HY4_Q4X=0` keeps the CPU lane on the GGUF's own experts (its
4-bit copies need AVX2).
