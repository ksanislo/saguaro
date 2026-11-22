# Inkling 2 Mini

**[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** (Thinking Machines; 26 B parameters, a
looped mixture of experts: 24 layers of which 12 run three times, 8 of 128 experts a token) is the model for a PC with
**one or two 16 GB cards**. It runs on its own engine, `saguaro-ik2`, from
[unsloth's UD-Q4_K_XL GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) read in place. The model and the GGUF
are **Apache 2.0**.

Back to [the models](MODELS.md) and the [README](../README.md). How the engine works, and the measurements behind it:
[INKLING2.md](INKLING2.md).

## What it needs

| | |
| --- | --- |
| **GPU** | One or two NVIDIA cards of 16 GB or more, Tesla T4 / RTX 20 (compute capability 7.5) and newer. Linux. No AMD, Mac or Windows engine yet. |
| **RAM** | Room for the experts beside the system: the engine keeps all 14.7 GiB of them in pinned RAM, and on one 16 GB card streams about a quarter of them to the card as they are routed. |
| **Disk** | 18.3 GB: the GGUF (17.6 GB) and the vision tower (0.6 GB). An NVMe SSD loads it in about ten seconds. |

| Cards | Layout the setup writes | Context | Experts |
| --- | --- | ---: | --- |
| One 16 GB card (one Tesla T4) | `--gpus 1 --chunk 2048` and `SAGUARO_IK2_PIN=0` | 32K | about three quarters in VRAM, the rest streamed from RAM |
| Two 16 GB cards (two Tesla T4s) | `--gpus 2 --chunk 8192` | 131K | all in VRAM |
| One card of 24 GB or more | `--gpus 1 --chunk 2048` | up to 131K (FP16 cache) | all in VRAM |

The first two were measured here on Tesla T4s. The third is upstream Project Maya's layout (measured there on an
RTX 3090, see [INKLING2.md](INKLING2.md) section 8); it has not been run on this project's hardware.

## Install

```sh
git clone https://github.com/ksanislo/saguaro.git && cd saguaro
./setup.sh --model inkling-2-mini
```

On a PC with one or two small cards and no model set up yet, `./setup.sh` alone recommends Inkling 2 Mini in its model
list. The setup:

1. checks the PC and picks the cards: two 16 GB cards where there are two (asked first when both are possible), else
   one. `--gpu 2` or `--gpus 2,3` names them yourself;
2. compiles `build/saguaro-ik2` for your cards;
3. shows the download (18.3 GB from `unsloth/Inkling-2-Mini-GGUF`, pinned to a commit) and starts it after you answer
   `y` (`--download-model` skips the question). It resumes where it stopped and checks each file's sha256;
4. writes the tokenizer folder with Thinking Machines' own chat template (the GGUF carries another one), the same as
   `python tools/strata_tokenizer.py --gguf <the GGUF> --out <folder> --chat-template tools/ik2/chat_template.jinja --check`;
5. builds the image encoder (`--no-vision` skips it and the 0.6 GB tower);
6. writes `maya-inkling-2-mini.json` with the layout above and starts the dashboard on `http://127.0.0.1:8080`.

From the web app: **Models › Download a model › Inkling 2 Mini**. The row says how it fits this PC's cards before you
start; the download is the same setup, run for you.

Files you already have go in with `./setup.sh --model inkling-2-mini --model-dir /path/to/folder` (the GGUF and
`mmproj-BF16.gguf` under their published names) or `./saguaro.sh --import-model FOLDER`.

## Speed

Measured on Tesla T4s (16 GB, sm_75) with two Xeon E5-2660 v2, greedy 128-256-token answers about 1K tokens deep,
prompts of 8-16K tokens, GPU clocks not pinned (pinning them measured no gain):

| | 1x Tesla T4, 32K context | 2x Tesla T4, 131K context |
| --- | ---: | ---: |
| Writing an answer | about 42 tok/s | about 52 tok/s |
| Reading a prompt | about 640 tok/s | about 1130 tok/s |
| Load | 8-10 s | 8-10 s |
| VRAM | 14.9 GiB | 14.9 + 14.3 GiB |

Answers get slower deeper into a context: one card 39 tok/s at 8K; two cards 46 at 16K, 40 at 30K and 28 at 125K.
Two loads of each, interleaved; this box drifts by about a tenth between runs.

## What it does

- **Thinking levels:** Off, Low, Medium, High (the default) and Max - the template's effort from 0 to 0.99. The model
  has no switch for its thinking; Off opens the answer's text at once.
- **Tool calls**, the OpenAI- and Anthropic-compatible API, streaming and stop, as for the other models.
- **Pictures:** the model's own vision tower through the image encoder, which starts when a picture arrives and keeps
  no VRAM between pictures.
- **Engine settings on the Models page:** context, the GPUs (one or two), the prompt chunk, CPU threads, the KV
  cache's format (FP16, or FP8 for half the bytes) and pictures. The engine stops at a flag it does not know, so the
  page writes only these.

## Tuning on your hardware

`saguaro-ik2` runs on NVIDIA cards from Turing on (compute capability 7.5: Tesla T4, RTX 20 and newer), Linux. A card
of 24 GB or more keeps the whole model in VRAM and needs none of this; on 16 GB cards four settings matter. Each is a
flag or an `env` entry in the config (`maya-inkling-2-mini.json`, Models page or by hand); restart the model after a
change. Measured on Tesla T4s, two loads of each setting, interleaved (decode about 60 / 8K tokens deep, an 8-16K read):

| Setting | What it trades | One T4 | Two T4s |
| --- | --- | --- | --- |
| `--chunk N` (prompt chunk) | A larger chunk reads prompts faster but its work buffers take VRAM from the expert cache: on one card, fewer experts stay on the card and answers write slower. | 1024 -> 2048: read 497 -> 712 tok/s, decode 34 -> 31 tok/s at 8K (default pins) | 2048 -> 8192: read 844 -> 1136 tok/s, decode 44 -> 46 (every expert stays in VRAM) |
| `SAGUARO_IK2_PIN` (pinned layers; one card) | Default: the core layers' experts are pinned on the card and run as one CUDA graph; `0`: one shared cache for every layer. `0` writes faster and, with chunk 2048, reads faster too; a load's first short prompt reads ~0.6 s slower. `core` / `SAGUARO_IK2_PIN_KEEP` measured no better. | default 39 / 34 tok/s, read 497 -> PIN=0 + chunk 2048: 42 / 39 tok/s, read 639 (the setup's default) | (not used: every expert is in VRAM) |
| `SAGUARO_IK2_ROOM_MB` (VRAM left free) | The rest of the card goes to experts. More room lets the picture encoder run on the GPU (0.8-1.8 s a new picture instead of 2-6 s on the CPU) at the cost of expert slots: 1536 on one T4 made every answer 9-11 % slower. | 1024 (default) | 1024 on the first card (default) |
| The one-T4 quant (`Inkling-2-Mini-T4-NVFP4-IQ3_XXS`, experimental) | Smaller experts (11.45 GiB: IQ3_XXS gate/up on 8 layers, layers 20-23 re-quantised) so every expert fits on one 16 GB card - faster, at a measurable quality cost: KL 0.012 and the same top token 95.0 % of the time against UD-Q4_K_XL, whose own run-to-run floor is 0.0019 / 97.8 %. Not a setup choice yet. | decode 39 -> 45 tok/s (31 -> 37 at 30K), read 502 -> 572 tok/s | (not needed) |

Pinning the GPU clocks measured no gain. To try a setting on other cards: change one value, restart, and compare a
long answer's tok/s and a long prompt's read speed against the default, a few times each in turn - a box's speed drifts
by several percent between runs. On a card between 16 and 24 GB, the engine prints at load how many experts stay in VRAM
("expert cache ... slots", "resident"); the closer that is to all of them, the less these settings matter.

## Limits

- **One conversation at a time.** A second request waits for the first.
- **Conversations kept:** up to four are set aside in RAM and come back without their prompt read again. They are
  saved across a restart only when the config's env names a folder (`SAGUARO_IK2_SLOT_DIR`); the setup sets none yet.
- **No drafter** is set up (the engine's `--draft` is by hand: [INKLING2.md](INKLING2.md)), and there is no Turbo.
- The Monitor shows its speeds, context and each card's memory; the experts card counts the expert-cache hits after
  each answer when the config's env has `SAGUARO_IK2_STATS=1`, and otherwise says where the experts are.
- Two cards were filled to 125K tokens of a 131K context (at `--chunk 2048`; `--chunk 8192` to 16K); the config leaves
  1 GiB of the first card free for that (`SAGUARO_IK2_ROOM_MB`).
