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
| One 16 GB card (one Tesla T4) | `--gpus 1 --chunk 1024` | 32K | about three quarters in VRAM, the rest streamed from RAM |
| Two 16 GB cards (two Tesla T4s) | `--gpus 2 --chunk 2048` | 131K | all in VRAM |
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

Measured on Tesla T4s (16 GB, sm_75) with two Xeon E5-2660 v2, greedy 256-token answers about 1K tokens deep, GPU
clocks not pinned:

| | 1x Tesla T4, 32K context | 2x Tesla T4, 131K context |
| --- | ---: | ---: |
| Writing an answer | about 40 tok/s | about 48 tok/s |
| Reading a prompt | about 490 tok/s | about 840 tok/s |
| Load | 8-10 s | 8-10 s |
| VRAM | 14.9 GiB | 15.4 + 8.1 GiB |

Answers get slower deeper into a context (one card: 37 tok/s at 8-9K). One load each; this box drifts by about a
tenth between runs.

## What it does

- **Thinking levels:** Off, Low, Medium, High (the default) and Max - the template's effort from 0 to 0.99. The model
  has no switch for its thinking; Off opens the answer's text at once.
- **Tool calls**, the OpenAI- and Anthropic-compatible API, streaming and stop, as for the other models.
- **Pictures:** the model's own vision tower through the image encoder, which starts when a picture arrives and keeps
  no VRAM between pictures.
- **Engine settings on the Models page:** context, the GPUs (one or two), the prompt chunk, CPU threads, the KV
  cache's format (FP16, or FP8 for half the bytes) and pictures. The engine stops at a flag it does not know, so the
  page writes only these.

## Limits

- **One conversation at a time.** A second request waits for the first.
- **No saved conversations yet:** a restart or a model switch reads the conversation again.
- **No drafter** is set up (the engine's `--draft` is by hand: [INKLING2.md](INKLING2.md)), and there is no Turbo.
- The Monitor shows its speeds, context and each card's memory; the engine reports no per-token expert counts, so the
  experts card says where they are instead of counting them.
- Two cards were loaded at a 131K context, not filled to it; the config leaves 1 GiB of the first card free for that
  (`SAGUARO_IK2_ROOM_MB`).
