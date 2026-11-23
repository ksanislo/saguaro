# Inkling 2 Mini

**[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** (Thinking Machines; 26 B parameters, a
looped mixture of experts: 24 layers of which 12 run three times, 8 of 128 experts a token) is the model for a PC with
**one or two 16 GB cards**. It runs on its own engine, `saguaro-ik2` (upstream Project Maya v1.1.74's, tuned here on
Tesla T4s), from
[unsloth's UD-Q4_K_XL GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) or one of
[Strata's packs](#the-files-unsloths-gguf-or-stratas-packs), read in place. The model, the GGUF and the packs are
**Apache 2.0**.

Back to [the models](MODELS.md) and the [README](../README.md). How the engine works, and the measurements behind it:
[INKLING2.md](INKLING2.md).

## What it needs

| | |
| --- | --- |
| **GPU** | One or two NVIDIA cards of 8 GB or more, V100 / Tesla T4 / RTX 20 (compute capability 7.0) and newer; or one AMD RDNA 3 / 3.5 / 4 GPU of 16 GB or more (checked on an RX 9070 XT). Linux. No Mac or Windows engine here. |
| **RAM** | The experts the card does not hold, pinned, beside the system: about 6 GiB with one 16 GB card, about 13 GiB with an 8 GB one; RAM and VRAM together about 24 GB (no SSD tier: the setup refuses less). |
| **Disk** | 18.3 GB: the GGUF (17.6 GB) and the vision tower (0.6 GB); 0.6 GB more for the drafter where drafting runs (every expert and the drafter in VRAM: RTX 30 / A100 and newer, an AMD GPU; not on a 16 GB card at 32K). An NVMe SSD loads it in about ten seconds. |

| Cards | Layout the setup writes | Context | Experts |
| --- | --- | ---: | --- |
| One 8-16 GB card (one Tesla T4) | `--gpus 1 --chunk 2048`, the seed expert profile (`SAGUARO_IK2_WARM`) | 32K | the offload path: the most-used in VRAM (2,303 of 3,072 on a T4), the rest in pinned RAM, fetched by the GPU |
| Two 16 GB cards (two Tesla T4s) | `--gpus 2 --chunk 8192` | 131K | the split: all in VRAM across the two cards |
| One card of 24 GB or more | `--gpus 1 --chunk 2048` | up to 256K (FP16 cache: 128K) | all in VRAM; drafting on RTX 30 / A100 and newer |
| One 16 GB AMD RDNA 3 / 4 GPU (an RX 9070 XT) | `--gpus 1 --chunk 2048`, the seed profile, `SAGUARO_IK2_RESERVE_MB=1024` | 32K | the offload path as on NVIDIA (the one-T4 quant: all in VRAM, `--chunk 1024`; each GGUF has its own config; the Models page refuses a Prompt chunk past 1024 there) |
| One AMD GPU of 24 GB or more | `--gpus 1 --chunk 1024` | up to 128K | all in VRAM |

The two T4 rows and the RX 9070 XT row were measured here. The others are upstream Project Maya's layouts (measured
there on RTX 3090s, V100s and a Radeon 8065S, see [INKLING2.md](INKLING2.md) section 8). Drafting is off on Volta / Turing, on the offload path and on two cards; where the drafter's
room does not fit beside the model (the one-T4 quant on a 16 GB card at 32K) the setup does not fetch it and the prompt
lookup drafts alone.

## The files: unsloth's GGUF or Strata's packs

The setup offers four sets of the same model and picks the one for your cards (`--ik2-quant` or `--model
inkling-2-mini-sq-15` chooses): unsloth's UD-Q4_K_XL GGUF, or one of
[Strata's packs](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata) (`.strata`, Apache 2.0), which
`saguaro-ik2` reads as they are - every weight exactly as Strata's own decode gives it (checked value by value on the
CPU, and on a T4 and an RX 9070 XT through the engine's kernels). A pack's folder also holds the repo's
`tokenizer.json` (the GGUF's tokenizer, table for table); pictures and drafting use the same mmproj and drafter.

| Files | Download | Experts in the engine | KL from the original (here / Strata's) | Same top token | 1x Tesla T4 (32K) | 1x RX 9070 XT (32K) | Picked for |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Strata SQ-20 (`inkling-2-mini-sq-20`) | 19.3 GB | 14.95 GiB | 0 (the original weights) / 0 | 100 % | 15.0 / 18.3 tok/s, read 519 (offload) | 41.0 / 42.8 tok/s, read 1329 (offload) | 24 GB cards and two 16 GB cards (every expert in VRAM) |
| unsloth UD-Q4_K_XL (`inkling-2-mini`) | 17.6 GB | 14.7 GiB | 0.0041 / - | 97.1 % | 26.3 / 29.4, read 546 (offload) | 58-59, read 1718 (offload) | one 16 GB card (the offload path) |
| Strata SQ-15 (`inkling-2-mini-sq-15`) | 14.3 GB | 13.0 GiB | 0.0171 / 0.0089 | 93.5 % | 24.7 / 25.6, read 540 (offload) | 56.9 / 53.6, read 1277 (offload) | - (a smaller download than UD at nearly its speed) |
| Strata SQ-10 (`inkling-2-mini-sq-10`) | 10.5 GB | 10.45 GiB | 0.0795 / 0.0502 | 87.1 % | 33.9 / 31.5, read 544 (every expert in VRAM) | 67.8 / 60.9, read 1247 (every expert in VRAM) | 8-12 GB cards |

KL here: against SQ-20 (the original weights), 2,000 tokens of text through `tools/ik2/kl.py` (the same on a T4 and
the RX 9070 XT to within 0.0004);
Strata's figures are on their own test prompts. Speeds: decode at ~50 / 8K tokens deep, 8K read, one request,
GPU clocks not pinned, measured 2026-11-22 (`runs/sf4`, `runs/sfm5`, `runs/sfm6`; the RX 9070 XT's SQ-10 / SQ-15 with
the faster S_K row dots, `runs/sf5` on the T4). The packs keep each expert in its own codec,
and the engine still gives every expert of a layer the slot of its widest one - which is why SQ-15 (14.3 GB) does not
yet stay whole in a 16 GB card's VRAM; per-expert slot sizes are the next step.

## Install

```sh
git clone https://github.com/ksanislo/saguaro.git && cd saguaro
./setup.sh --model inkling-2-mini
```

On a PC with one or two small cards and no model set up yet, `./setup.sh` alone recommends Inkling 2 Mini in its model
list. The setup:

1. checks the PC and picks the cards: one where it holds every expert, else two where there are two (asked first when
   both are possible; the larger first), else one on the offload path. `--gpu 2` or `--gpus 2,3` names them yourself;
2. compiles `build/saguaro-ik2` for your cards;
3. shows the download (18.3 GB from `unsloth/Inkling-2-Mini-GGUF`, pinned to a commit) and starts it after you answer
   `y` (`--download-model` skips the question). It resumes where it stopped and checks each file's sha256;
4. writes the tokenizer folder with Thinking Machines' own chat template (the GGUF carries another one), the same as
   `python tools/strata_tokenizer.py --gguf <the GGUF> --out <folder> --chat-template tools/ik2/chat_template.jinja --check`;
5. fetches the drafter where drafting runs (`--drafting off` skips it), and builds the image encoder (`--no-vision`
   skips it and the 0.6 GB tower);
6. writes `maya-inkling-2-mini.json` with the layout above and starts the dashboard on `http://127.0.0.1:8080`.

From the web app: **Models › Download a model › Inkling 2 Mini**. The row says how it fits this PC's cards before you
start; the download is the same setup, run for you.

Files you already have go in with `./setup.sh --model inkling-2-mini --model-dir /path/to/folder` (the GGUF and
`mmproj-BF16.gguf` under their published names) or `./saguaro.sh --import-model FOLDER`. A folder with another
Inkling-2-Mini GGUF and no published one (a quant of your own) is taken as it is, not sha256-checked, and its experts'
size is read from the file for the fit (e.g. the one-T4 quant on a 16 GB AMD card).

## Speed

Measured here 2026-11-22, greedy 256-300-token answers, GPU clocks not pinned. Tesla T4s (16 GB, sm_75, PCIe 3)
with two Xeon E5-2660 v2: two loads of each, interleaved with the previous engine (our v1.1.72 port); an RX 9070 XT
(16 GB, gfx1201, PCIe 5) with a Ryzen 7 9800X3D: one load each.

| Decode at ~50 / 8K / 30K tokens deep, read of an 8K / 30K prompt | this engine | previous engine (our v1.1.72 port) |
| --- | ---: | ---: |
| 1x T4, UD-Q4_K_XL, 32K (offload path, dynamic GPU cache) | 40.0 / 39.3 / 35.6 tok/s, read 837 / 791 | 42.8 / 38.5 / 34.9, read 633 / 616 |
| 1x T4, one-T4 quant, 32K (every expert in VRAM) | 44.9 / 41.5 / 37.0, read 515 / 429 (before the Turing prompt attention) | 44.5 / 41.2 / 36.9, read 569 / 550 |
| 2x T4, UD-Q4_K_XL, 131K (the split; decode ~50 / 16K, read 16K) | 51.3 / 39.9, read 1102 | 50.4 / 41.6, read 1137 |
| 1x RX 9070 XT, UD-Q4_K_XL, 32K (offload path, dynamic GPU cache) | 71.2 / 65.2 / 60.5, read 1668 / 1282 | - |
| 1x RX 9070 XT, one-T4 quant, 32K (every expert in VRAM) | 83.3 / 73.3 / 66.7, read 1148 / 950 | - |

One T4 writes within a few percent of the previous engine and reads prompts about 30 % faster; the first short prompt
of a load answers in 0.8 s instead of 1.6 s. What makes it: the offload path's VRAM tier is a cache on the GPU
(upstream PR #1118's Clock-LFU slots, the same bits as a fixed tier - KL 0), the tail layers' pool a fifth of the
budget, and Turing's fused prompt attention (one K / V tile in shared memory; against cuBLAS's prompt attention KL
3.5e-3 / the same top token 97 % over 500 tokens after an 8K prompt). Two T4s write and read within 4 % of the
previous engine; the first short prompt is ~0.5 s slower. With the one-T4 quant two answers at once (`--parallel 2`,
16K context) give 47.7 tok/s together against 41.9 alone. Two loads of each, interleaved; this box drifts by about a
tenth between runs. The one-T4 quant rows are from before these changes. In short: 1x Tesla T4 about 40 tok/s
writing and about 840 tok/s reading; 2x Tesla T4 about 51 tok/s writing and about 1100 tok/s reading; 1x RX 9070 XT
about 71 tok/s writing and about 1670 tok/s reading (the one-T4 quant: about 83 tok/s writing, about 1150 tok/s
reading) - what the setup quotes for these layouts.

## What it does

- **Thinking levels:** Off, Low, Medium, High (the default) and Max - the template's effort from 0 to 0.99. The model
  has no switch for its thinking; Off opens the answer's text at once.
- **Tool calls**, the OpenAI- and Anthropic-compatible API, streaming and stop, as for the other models.
- **Pictures:** the model's own vision tower through the image encoder, which starts when a picture arrives and keeps
  no VRAM between pictures.
- **Engine settings on the Models page:** context, the GPUs (one or two), the prompt chunk, CPU threads, the KV
  cache's format (FP16, or FP8 for half the bytes), pictures, drafting (where it runs) and answers at once
  (`--parallel`, where one GPU holds every expert). The engine stops at a flag it does not know, so the page writes
  only these.

## Tuning on your hardware

`saguaro-ik2` runs on NVIDIA cards from Volta on (compute capability 7.0: V100, Tesla T4, RTX 20 and newer), Linux. A
card of 24 GB or more keeps the whole model in VRAM and needs none of this; on 16 GB cards these settings matter. Each
is a flag or an `env` entry in the config (`maya-inkling-2-mini.json`, Models page or by hand); restart the model after
a change. Tuned on Tesla T4s, two loads of each setting, interleaved (decode about 60 / 8K tokens deep, an 8-16K read).
The `--chunk` figures are the previous engine's (its one-card cache); on this engine one T4 runs the offload path and
the one-card `--chunk` trade is not measured again yet:

| Setting | What it trades | One T4 | Two T4s |
| --- | --- | --- | --- |
| `--chunk N` (prompt chunk) | A larger chunk reads prompts faster but its work buffers take VRAM from the experts' tier: on one card, fewer experts stay on the card and answers write slower. | (previous engine) 1024 -> 2048: read 497 -> 712 tok/s, decode 34 -> 31 tok/s at 8K | (previous engine) 2048 -> 8192: read 844 -> 1136 tok/s, decode 44 -> 46 (every expert stays in VRAM) |
| `SAGUARO_IK2_ROOM_MB` (the offload path's room) | The VRAM the offload path keeps for prompts and its staging slots; the rest of the card is the experts' tier. More room lets the picture encoder run on the GPU (0.8-1.8 s a new picture instead of 2-6 s on the CPU) at the cost of tier: on the previous engine 1536 on one T4 made every answer 9-11 % slower. | 1024 (default) | 1024 on the first card (default) |
| `SAGUARO_IK2_OFF_DYNAMIC` (the offload path's GPU cache) | On by default on one GPU: missed experts take the coldest slots on the GPU (Clock-LFU, inside the token's CUDA graph), the same answers to the bit. `0`: the fixed tier from the profile. | default on: 19.6 -> 40 tok/s | off (the split) |
| `SAGUARO_IK2_TAIL_SLOTS` | The dynamic cache's pool for the tail layers (20-23, Q8_0, 8.1 MiB an expert): a fifth of the budget by default (about 260 on a T4); 96 measured 12-16 % slower, 384 no faster. | default | - |
| `SAGUARO_IK2_WARM` (the tier's start) | The experts the tier starts with: the setup writes Saguaro's profile (`data/expert-profile-ik2mini.bin`); a profile of your own use (`SAGUARO_IK2_POP_OUT=FILE` writes one) fits your prompts better. | the setup's default | the setup's default |
| The one-T4 quant (`Inkling-2-Mini-T4-NVFP4-IQ3_XXS`, experimental) | Smaller experts (11.45 GiB: IQ3_XXS gate/up on 8 layers, layers 20-23 re-quantised) so every expert fits on one 16 GB card - the resident path (and `--parallel`), at a measurable quality cost: KL 0.012 and the same top token 95.0 % of the time against UD-Q4_K_XL, whose own run-to-run floor is 0.0019 / 97.8 %. Not downloadable: `--model-dir` with its file. Needs `--chunk 1024` and `SAGUARO_IK2_RESERVE_MB=1024` to stay resident at 32K. | decode 24 -> 45 tok/s (24 -> 37 at 30K), read 706 -> 515 tok/s | (not needed) |
| `SAGUARO_IK2_RESERVE_MB` | What the plan keeps free beside every expert (1536 by default); 1024 lets a file that just misses stay resident. | 1024 with the one-T4 quant | - |

Pinning the GPU clocks measured no gain. To try a setting on other cards: change one value, restart, and compare a
long answer's tok/s and a long prompt's read speed against the default, a few times each in turn - a box's speed drifts
by several percent between runs. On the offload path the engine prints at load how much of the experts its tier holds
and, after an answer, how many picks it fetched; the closer the tier is to all of them, the less these settings matter.

## Limits

- **One conversation at a time** on one or two 16 GB cards with UD-Q4_K_XL: a second request waits for the first.
  Answers at once (`--parallel N`) run only where one GPU holds every expert (a 24 GB card, the one-T4 quant on one
  T4).
- **Conversations kept:** up to four are set aside in RAM and come back without their prompt read again. They are
  saved across a restart only when the config's env names a folder (`SAGUARO_IK2_SLOT_DIR`); the setup sets none yet.
- **Drafting** is off on Volta / Turing (a T4), on the offload path and on two cards; elsewhere the setup fetches the
  drafter and `--drafting auto` measures once whether it pays. No Turbo.
- The Monitor shows its speeds, context and each card's memory; the experts card counts the expert-cache hits after
  each answer when the config's env has `SAGUARO_IK2_STATS=1`, and otherwise says where the experts are.
- **Out of GPU memory in a request** (a prompt chunk too big for what the card has left): that request fails with an
  error saying so, and the engine starts again for the next one (a second or two).
- Two cards were filled to 125K tokens of a 131K context (at `--chunk 2048`; `--chunk 8192` to 16K); the config leaves
  1 GiB of the first card free for that (`SAGUARO_IK2_ROOM_MB`).
