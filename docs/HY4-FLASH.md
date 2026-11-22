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
5. writes `maya-hy4-flash-maya-s.json` and starts the dashboard on `http://127.0.0.1:8080`.

On four equal 16 GB cards the config is the measured one: `--tp 4 --chunk 2048 --reserve-gb 0.5`, a 131K context and
`HY4_HOT=1`, which places the experts in VRAM by the scores shipped in `data/` for Maya-S. Any other set of cards gets
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
- **Pictures:** the vision tower runs in the engine and takes no VRAM of its own. **Not measured on Tesla T4s yet.**
- **Engine settings on the Models page:** context, GPUs, tensor parallel (Auto, Off, or the group's size), the VRAM
  kept free, the prompt chunk, CPU threads, the cache's rows (Auto, 8-bit, F16) and pictures. The engine stops at a
  flag it does not know, so the page writes only these.
- **The Monitor** shows how many experts are in VRAM and how many the CPU computes, the experts a token took from the
  CPU and the n-gram table's read time.

## Limits

- **One conversation at a time.** A second request waits for the first.
- **No saved conversations yet:** a restart or a model switch reads the conversation again.
- **No drafter is installed, and no Turbo.** On four Tesla T4s the model's MTP block was no faster (32-34 tok/s against
  32-33) and read prompts 35% slower; the DFly2 drafter does not load under tensor parallel. Both remain by hand
  (`--mtp`, `--dfly2`; the repo has the files).
- **Quality of this layout:** against the layer split with F16 cache rows, tensor parallel with 8-bit rows has a KL
  divergence of 0.06 and picks the same next token 88% of the time. Any change of rounding order moves this 2-3 bit
  quant by that much - a different chunk size alone does (0.059) - so it is the quant's floor, not a fault of the
  layout.
- AMD, Windows and Mac: not built here.
