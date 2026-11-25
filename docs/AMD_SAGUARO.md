# Saguaro on RX 7900 XT / XTX, R9700 / RX 9070 and Strix Halo / Gorgon Halo

## What runs on AMD

The state of this version (keep this table, the setup's `NO_ENGINE` list in `saguaro.py` and the README in step):

| Model | Engine (CMake target) | On AMD (HIP) |
| --- | --- | --- |
| GLM-5.3-Flash | `saguaro-glm` (`maya`) | builds and runs; its GPU self-tests pass on an RX 9070 XT (gfx1201, ROCm 7.1). The quant that fits a 16 GB card + 96 GB of RAM is Maya-XS (73 GB); the setup recommends the largest one that fits VRAM + RAM |
| Qwen3.8-Flash-Next (unsloth's UD-Q4_K_XL / Q8_0; Blackfrost-AI's CYBER-FROST fine-tune as a quant of its own) | `saguaro-qfn` (`engines/qfn`) | builds and runs on an RX 9070 XT (gfx1201, ROCm 7.1) with 96 GB of RAM (27-41 tok/s answers, 520-1,080 tok/s 8K-16K prompt reads); the setup offers it (`--model qwen3.8-flash-next`, `--model cyber-frost` for the fine-tune) for one RDNA GPU (the layout it was measured on: 16 GB or more with ~96 GB of RAM; smaller cards and less RAM run it too, slowly - only a PC that cannot hold the model at all is refused): the experts the card does not hold in a pinned RAM tier and the rest read from the SSD - [below](#qwen38-flash-next-on-the-saguaro-qfn-engine) |
| DeepSeek-V4.1-Flash-1017 | `saguaro-ds4` (`maya-ds4`) | compiles for HIP (gfx1201 checked); not yet run on an AMD GPU |
| MiniMax-M3.1-Flash | `saguaro-mm3` (`maya-mm3`) | compiles for HIP (gfx1201 checked); not yet run on an AMD GPU |
| Hy4-Flash | `saguaro-hy4` (`maya-hy4`) | not run on an AMD GPU yet: the setup does not offer it on AMD |
| Inkling 2 Mini | `saguaro-ik2` | builds and runs on an RX 9070 XT (gfx1201, ROCm 7.1): UD-Q4_K_XL on the offload path about 71 tok/s, every serve check incl. pictures passes; the setup recommends it for one 16-23 GB RDNA 3 / 4 card |

`./saguaro.sh --check` on an AMD PC lists the same. Report how a model runs on your card in an
[issue](https://github.com/ksanislo/saguaro/issues).

## GLM-5.3-Flash on AMD

This is a Linux and Windows HIP build and installer path for Saguaro's GLM-5.3-Flash
engine on `gfx1100`, `gfx1201` and `gfx1151` (Strix Halo, Gorgon Halo). It uses one GPU, or two supported GPUs that split the
layers (see [Two GPUs](#two-gpus)), and reads pictures too: setup builds the image encoder for the GPU
(ggml's HIP backend), started only while a picture is encoded, as on NVIDIA; `--no-vision` leaves it out. The
[CPU image encoder](#cpu-image-encoder-on-linux) stays an option on Linux.
Windows: see [Windows](#windows).

Use a system ROCm 7 installation with its HIP compiler and hipBLAS, Python
3.10+, CMake 3.24+, and a C++20 compiler. `ROCM_PATH` selects an installation
outside `/opt/rocm`. Versioned LLVM slots, as Gentoo's
`<root>/lib/llvm/22/bin/clang++`, are accepted too. Everything Saguaro downloads
goes into the project or its model-data folder; setup does not install system
packages.

A distribution's own ROCm packages under `/usr` work too, including Debian's and Ubuntu's multiarch layout
(`/usr/lib/x86_64-linux-gnu/libhipblas.so`, the compiler in `/usr/lib/llvm-<N>/bin`, the device bitcode in
`/usr/lib/llvm-<N>/lib/clang/<N>/amdgcn/bitcode`). A ROCm older than 7 (Debian's 6.4, hipBLAS 2.x) is untested and
refused by default; `SAGUARO_HIP_ALLOW_ROCM6=1` lets setup go on with a warning. The build then turns on hipBLAS 2's
`HIPBLAS_V2` API and skips `hipblasSetWorkspace` (absent before hipBLAS 3). Measured that way on Debian sid
(ROCm 6.4.3, hipBLAS 2.4) with an MI50 + MI100 layer split and a CPU without AVX2 - see the pull request.

```sh
./saguaro.sh --backend hip --gpu 0 --check
./saguaro.sh --backend hip --gpu 0 --context 8192 --no-vision --yes \
  --download-model --no-start --port 8099 --env SAGUARO_GLM_RAM_GB=60
./saguaro.sh --backend hip --port 8099
```

The first setup downloads about 96.5 GB of weights and verifies the SHA-256
hash of each shard. `--gguf-dir DIR` uses existing files instead. GPU numbers
are the kernel KFD topology order shown by `--check`; on the test machine the
7900 XT is GPU 0, an R9700 is GPU 1, and the integrated GPU is GPU 2. This
installer accepts `gfx1100`, `gfx1201` and `gfx1151` and compiles the binary for the
architectures of the chosen cards only (`gfx1100;gfx1201` for that 7900 XT + R9700).

RDNA 2 (`gfx1030`: RX 6800 / 6800 XT / 6900 XT / 6950 XT, Radeon PRO V620 / W6800) is accepted too: it has been run
on a Radeon PRO V620 with ROCm 7, and not on another RDNA 2 card yet - please report how it goes on other RDNA 2 cards in an issue. These
cards have no matrix cores (WMMA), so the engine uses its portable kernels there (the prompt attention on the F32
kernel), and a build for RDNA 2 cards alone needs no rocWMMA headers (ROCm 7 no longer ships them). The ROCm
installation needs its library kernels (hipBLAS) for `gfx1030`.

Configs are named `maya-<quant>-hip.json` and
select the AMD device through `HIP_VISIBLE_DEVICES`.

`--bench` and `--report` work on HIP: `./saguaro.sh --backend hip --bench` uses
`build-hip/saguaro-glm` and the config's GPU order and environment; run it with Saguaro
stopped. `./saguaro.sh --backend hip --report` includes AMD GPU details from
`rocm-smi` or `amd-smi` when available (KFD topology otherwise), the ROCm path
and version, and the engine's speed lines. Add `--config /path/to/maya-hip.json`
to either command to select a particular installed config.

The discrete-GPU HIP config starts with an 8K context when requested above, 3 GiB of GPU
headroom, and 16 GiB of system-RAM headroom. The engine sizes the prompt chunk
from its prompt-memory budget. The example additionally caps the pinned expert
cache at 60 GiB. Change those settings with `--env KEY=VALUE` during setup.
Available RAM, rather than installed RAM, determines how much can be cached.
The engine reduces its RAM-tier allocation if ROCm cannot pin the requested
amount.

## CPU image encoder on Linux

HIP setup builds a GPU image encoder (above); instead, the server can use a CPU-only
`vision-glm` helper alongside the HIP text engine. This is an explicit manual
setup. Stop Saguaro before building or changing the config.

After HIP setup has fetched its pinned `third_party/llama.cpp` checkout, run
these commands from the Saguaro project root. If you used `--llama-dir`, use that
same directory for both the patch and `LLAMA_DIR` below. Use a separate build
directory to keep any existing CUDA image encoder intact.

```sh
.venv/bin/python tools/vision/glm5next_patch.py third_party/llama.cpp
cmake -S tools/vision -B build-vision-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR="$PWD/third_party/llama.cpp" -DMAYA_VISION_CUDA=OFF
cmake --build build-vision-cpu --target maya-vision -j 2
```

The idempotent patch above adds GLM-5.3-Flash support to the pinned llama.cpp
image library. It is the same patch used by Saguaro's CUDA vision installer.
Download the matching vision files into your model-data directory (about
1.14 GB). This example uses the artifact revision tested with Maya v1.1.5:

```sh
mkdir -p data/vision
vision_url=https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/resolve/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da/vision
curl -fL --retry 5 -C - -o data/vision/mmproj-GLM-5.3-Flash-F16.gguf \
  "$vision_url/mmproj-GLM-5.3-Flash-F16.gguf"
curl -fL --retry 5 -C - -o data/vision/GLM-5.3-Flash-vocab.gguf \
  "$vision_url/GLM-5.3-Flash-vocab.gguf"
printf '%s\n' \
  '3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882  data/vision/mmproj-GLM-5.3-Flash-F16.gguf' \
  '8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912  data/vision/GLM-5.3-Flash-vocab.gguf' \
  | sha256sum -c -
```

Add this `vision` member to the HIP config, replacing `/absolute/path/to/saguaro`
with your project path. If your model-data directory differs, use its paths:

```json
"vision": {
  "exe": "/absolute/path/to/saguaro/build-vision-cpu/bin/vision-glm",
  "mmproj": "/absolute/path/to/saguaro/data/vision/mmproj-GLM-5.3-Flash-F16.gguf",
  "model": "/absolute/path/to/saguaro/data/vision/GLM-5.3-Flash-vocab.gguf",
  "gpu": false
}
```

Start with `./saguaro.sh --config /path/to/your-hip-config.json --port 8099`.
Check that `/health` reports `images: true`, then send an actual image through
`/v1/chat/completions`. Keep the manually edited config if running setup again;
the HIP installer still writes a text-only config.

Verified on Linux, Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`), 128 GB unified
RAM, ROCm 7.14.1, stock Maya v1.1.5 (`5403c222`), Maya-S24, and pinned llama.cpp
`3cf03257`: a real image request correctly identified a green circle on a white
background. Two checks took 3.38 and 3.12 seconds, including generation.
These are small functional checks, not broad vision-quality or performance
claims. CPU image encoding uses system RAM and adds CPU work; other AMD devices,
Windows, large-image limits, and multiple-image workloads were not tested here.

## Two GPUs

`--gpus 0,1` splits the layers across two supported cards, as on NVIDIA: each card caches the experts of its own
half, and the second half also runs the model's MTP block, which drafts the next token so both cards stay busy. Setup
puts the card with more VRAM first (it takes the bigger first half), leaves the split point to the engine, and writes
one hipBLASLt table per architecture (`SAGUARO_HIPBLASLT_TUNING` takes a path list, `:`-separated on Linux and `;` on Windows; each card uses the table
for its own architecture). A discrete GPU paired with a unified-memory APU puts the discrete GPU first and loads the
parts sequentially (`SAGUARO_GLM_PARALLEL_LOAD=0`), so the APU sizes its pool after the first part pins its RAM tier.
The two cards may be different architectures. The split passes one small host-memory hop
per token, so no peer-to-peer access between the cards is needed.

Measured on an AI PRO R9700 (32 GB, first) + RX 7900 XT (20 GB), 192 GB RAM, Maya-S, 8K context, 90 GB RAM tier,
v1.0.11, greedy, 256-token answers (median of 10 requests):

| | one RX 7900 XT | R9700 + RX 7900 XT |
|---|---|---|
| answer speed | ~15 tok/s | ~33 tok/s (drafts accepted ~76%) |
| prompt speed, 4K-token prompt | ~414 tok/s | ~490 tok/s |

39 back-to-back requests (1.8-4K-token prompts) ran without an error. The default split (the midpoint + 2) measured
best: moving two or four more layers to the first card did not speed up answers.

### Expert helper

`SAGUARO_GLM_EXPERT_HELPER=1` is an opt-in alternative to a layer split: visible GPU 0 keeps every trunk layer,
attention, state and MTP; visible GPU 1 owns a complementary expert cache. A unified-memory helper reserves slots
in the existing portable pinned RAM tier (no second weight copy); a discrete helper uses its own VRAM, automatically
sized by aligned expert bytes with 512 MiB left free. Primary admissions, prefetch and background promotions exclude
helper-owned experts. At idle boundaries, colder helper experts can be replaced by hotter unowned experts of the same
layer.
The helper quantizes activations and reduces weighted expert outputs on its GPU before returning one vector per token.
CPU quantization runs only for actual CPU fallback; uncovered experts retain the existing CPU/PCIe/disk paths.
Batched prefill retains its primary-GPU staging and CPU pool, and may read helper-owned UMA slots without evicting them.
All of this is part of the single helper switch; there is no separate optimization flag.
Native GGUF experts only; EXL3 is not supported.

For physical device 0 = 8060S and 1 = R9700, run the engine with `HIP_VISIBLE_DEVICES=1,0`,
`SAGUARO_GLM_EXPERT_HELPER=1` and `SAGUARO_GLM_SPLIT=0`. The indices are process-local; the system order is unchanged.
Keep the normal pack, context, headroom and MTP settings. Unset `SAGUARO_GLM_EXPERT_HELPER` to restore the existing
behavior. This path does not claim general multi-GPU support or a speedup; measure it on the target
hardware.

#### The helper beside a layer split (two cards and the APU)

With `SAGUARO_GLM_SPLIT=auto` (or pinned boundaries) the helper's device is left out of the split and every part of the
split opens its own helper there over its own layers: a part's misses go to the APU, which computes them inside that
part's pinned RAM tier at the APU's memory speed instead of the part's CPU lane. For physical device 0 = 8060S and
1, 2 = two discrete cards, run the engine with `HIP_VISIBLE_DEVICES=1,2,0`, `SAGUARO_GLM_EXPERT_HELPER=2` and
`SAGUARO_GLM_SPLIT=auto` (the split's devices are then 0 and 1; `SAGUARO_GLM_DEVS=0,1` says so explicitly). The helper's
device may not be one of the split's. The engine logs one `glm helper:` line per part. Measured on this layout: see the
speed table; the CPU lane still takes the misses the helper does not own.

## Strix Halo

Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) uses unified memory. Use the same
HIP setup command above, selecting its GPU number and omitting the
`--env SAGUARO_GLM_RAM_GB=60` override. Setup detects the APU even if sysfs
reports only a small VRAM carve-out. It writes a 1 GiB reserve, 16 GiB of
system headroom, `SAGUARO_GLM_PREFILL_SUB=1024`, and a 6144 MiB prompt budget
when system RAM is at least 96 GiB (4096 MiB otherwise), plus the existing
`gfx1151` hipBLASLt table. Use `--gpu N` for the APU alone, or `--gpus D,N` to pair it with a supported discrete GPU
(see [Two GPUs](#two-gpus)). It leaves `POOL_GB` and `RAM_GB` unset.

For integrated HIP devices the engine sizes the pool from `/proc/meminfo`'s
`MemAvailable` **after** dense weights, KV and prompt staging are allocated:
`min(expert bytes, max(0, MemAvailable - RAM_HEADROOM_GB - RESERVE_MB))`.
It does not treat HIP's reported GTT capacity (112 GiB on the 128 GB box) as
free physical memory. For example, 110 GiB available leaves 93 GiB after the
default headroom/reserve, capped at Maya-S's ~80.17 GiB expert pool. When every
expert fits, warm-up fills every pool slot without reserving idle spares;
the RAM tier keeps only its small per-class staging floor, and
the prompt disk landing ring defaults to 12 slots. Prompt buffers borrow the
pool's tail; they do not require a second 6 GiB allocation. Explicit `--env`
settings still override defaults; pool/VRAM caps still limit the pool.
Discrete HIP GPUs and CUDA keep their existing sizing behavior.

Measured on a 128 GB Strix Halo box with ROCm 7.2.2 and manually tuned settings:
~210 tok/s for 3-4K-token prompts and ~17.8 tok/s answers, with every expert
in the GPU pool (288 slots/layer, 12,096 total, ~99.9% hits). These measurements
precede the automatic sizing change; verify its startup log and repeated
prompt/answer rounds on the real box. See Project Maya's [tracking issue #6](https://github.com/mw00/project-maya/issues/6).

**Answers and prompts (v1.1.10).** Where every expert sits in the GPU pool, the APUs answer faster with the same
text (`SAGUARO_GLM_MTP_DECODE`, on by default on `gfx115x` only). On the Gorgon Halo of [Windows](#windows) (Radeon
8065S, Maya-S, 32K context), 256-token answers run at 28.4-29.6 tokens/s to short prompts and at 23.2-23.5 after
3K- and 8K-token prompts, against 18.4-19.7 without it; 3K- and 8K-token prompts read at 271 and 278 tokens/s
(the fused prompt experts and the `wmma2` attention, both on by default on `gfx1151`).

## Inkling 2 Mini

`./saguaro.sh --setup --backend hip --model inkling-2-mini` (Linux). Its engine (`saguaro-ik2`, upstream Maya v1.1.74's
with Saguaro's additions) runs on an RDNA 3 / 3.5 / 4 GPU (gfx11 / gfx12) of 12 GB or more: every expert in VRAM where
the file fits, else the offload path as on NVIDIA (the most-used experts in VRAM, the others pinned in RAM and fetched
by the GPU). Checked on an RX 9070 XT (16 GB, gfx1201, ROCm 7.1, PCIe 5) with a Ryzen 7 9800X3D, 32K context:

| | decode at ~50 / 8K / 30K tokens deep | 8K prompt read |
| --- | ---: | ---: |
| unsloth's UD-Q4_K_XL (offload path, the dynamic GPU cache) | 71.2 / 65.2 / 60.5 tok/s | 1,668 tok/s |
| our one-T4 quant (every expert in VRAM; lower quality: KL 0.012 vs UD-Q4_K_XL) | 83.3 / 73.3 / 66.7 tok/s | 1,148 tok/s |

Chat, thinking, tools, stops, pictures, the Monitor and kept conversations pass on both. Not checked on AMD yet:
drafting, two AMD GPUs, a Strix / Gorgon Halo APU. [INKLING2.md](INKLING2.md) has the engine's AMD paths.

## Qwen3.8-Flash-Next on the saguaro-qfn engine

Measured on one RX 9070 XT (gfx1201) with 96 GB of RAM (figures below). This is Qwen3.8-Flash-Next as Qwen released it,
in unsloth's GGUFs, on its own engine (`saguaro-qfn`, `engines/qfn`, from Strata); Blackfrost-AI's CYBER-FROST
fine-tune is a quant of the same engine (`--model cyber-frost`, [SETTINGS.md](SETTINGS.md#cyber-frost-38-ps-the-cyber-frost-quant)).
The setup offers it for one AMD RDNA 3 / 4 GPU of 16 GB or more and about 96 GB of RAM (Linux; an APU not yet), and
refuses with the reason below that. The same model sets up on NVIDIA too - one card of 16 GB or more, or up to four
equal ones taking the 48 layers in turn - on Linux only: [SETUP.md](SETUP.md#what-the-first-run-does). Its settings are
in [SETTINGS.md](SETTINGS.md#qwen38-flash-next-saguaro-qfn). The two setup commands:

```sh
./saguaro.sh --setup --backend hip --gpu 0 --model qwen3.8-flash-next            # UD-Q4_K_XL, 111 GB (or ...-q8_0)
./saguaro.sh --setup --backend hip --gpu 0 --model cyber-frost                   # CYBER-FROST-3.8, 112.5 GB (its NextN block is the drafter)
./saguaro.sh --setup --backend hip --gpu 0 --model qwen3.8-flash-next --gguf-dir /path/to/UD-Q4_K_XL   # files you have
```

It builds `build-hip/qfn/saguaro-qfn` for the card, downloads the files pinned and sha256-checked (or checks yours in
place), indexes them (the pack, about a minute), downloads the Lamina MTP draft layer (0.8 GB) and writes
`maya-qwen3.8-flash-next-ud-q4_k_xl-hip.json` with the settings measured on the RX 9070 XT: the experts the card does not
hold, the hottest by the shipped expert profile, in a page-locked RAM tier of the RAM less 20 GB
(`--resident-budget-gib`), the rest read from the SSD; MTP drafting; INT8 K/V, 32K of it in VRAM; the K-quant prompt
kernels (`SAGUARO_MMQ_KQ=1`). The RDNA4 prompt defaults (GDN, control-vector fusion, chunks up to 16K, matrix-core prompt
attention) switch on by themselves on gfx12.

The same run also fetches the picture tower (the model's mmproj, 0.9 GB, pinned and sha256-checked) and builds its
reader for the card (`vision-qfn-hip`), which starts when the first picture arrives; `--no-vision` leaves both out. The
Lamina MTP draft layer (0.8 GB) comes with it and the setup asks whether to draft with it (`--drafting off` / `on`):
`--calibrate` tunes the PCIe share, the draft floor and the CPU workers for this PC and measures the drafter,
`--speed-calibration` measures the drafter alone and keeps the faster, and `--bench` benchmarks the model on
`saguaro-qfn`. Several answers at once (`--parallel 4`) are supported; on this one 16 GB card parallel decoding measured
slower, so the setup writes no `parallel` for it (2 where a config runs on several GPUs). The Models page has the
model's own fields: the context (changed live from Settings > Model), the K/V window kept in VRAM
(`--kv-resident`), the RAM for the experts, answers at once, drafting and pictures; its Turbo button turns drafting on
and writes one answer at a time. To build only the engine and its tests by hand:

```sh
ROCM=/opt/rocm HIP_ARCHS=gfx1201 tools/qfn/build_hip.sh build-qfn-hip
HIP_VISIBLE_DEVICES=0 ctest --test-dir build-qfn-hip --output-on-failure
```

Measured on an RX 9070 XT (16 GB, gfx1201, ROCm 7.1) with a Ryzen 7 9800X3D and 96 GB of DDR5, the model on a PCIe 4.0
NVMe, with the config the setup writes (64K context, MTP drafting on, a 71 GB RAM tier), 256-token greedy answers to
fresh prompts (one load, 2026-11-23):

| prompt | 1K | 4K | 8K | 16K | 32K | 60K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt read, tok/s | 116 | 274 | 524 | 1,079 | 805 | 924 |
| decode, tok/s | 27.5 | 34.3 | 40.9 | 40.5 | 39.8 | 34.9 |

Load 45 s. The card holds 2,997 experts (8.8 GiB) next to the dense weights, the K/V and the draft head, the RAM tier
69.6 GiB - all but 2 GiB of the experts the card does not hold; 53-66% of the decode's experts were in the card and the
rest in RAM, none read from the drive while answering; a whole prompt read pulled 0.07-0.65 GB from the drive. Answers
vary about 25% in speed from load to load on the same text here.

Correctness against the T4 build (the same five 12-29K-token prompts, 5,150 teacher-forced positions): KL 0.0057, the
same top token 96.8% of the time (99.8% where the T4 is confident), perplexity 0.2% lower; the T4 against itself is
bit-identical. Q8_0 (177 GB) on the same PC: KL 0.0050 against the T4's Q8_0, answers at 14-15 tok/s, prompts read at 73-315 tok/s (70 GB of its experts in RAM, ~50 GB read from the SSD as they are routed). MTP drafting does not change the answers on AMD (the same greedy tokens with it on and off). Not checked on AMD yet: pictures (the setup downloads the tower and builds its reader now, but no AMD run of one is measured here) and two AMD GPUs (the setup offers one card there).

## Instinct MI300X / MI325X

AMD Instinct MI300X / MI325X (`gfx942`, CDNA3) is accepted, on one GPU; the MI325X has been benched below, the MI300X itself has not
been tried yet. Like the MI100 it is a wave64 card and runs through the hip_compat half-wavefront layer.

Install ROCm's development packages for the card first (ROCm 7.14: `amdrocm-core-dev7.14-gfx942` and
`amdrocm-blas-dev7.14`), then run the setup above without the `SAGUARO_GLM_RAM_GB` override. On an MI325X (256 GB) every
expert of Maya-L fits in HBM next to a 1M-token context; with that much room, `"--kv-streaming", "off"` in the config's
`"args"` keeps the whole latent cache on the card too.

```sh
./saguaro.sh --backend hip --gpu 0 --context 1048576 --no-vision --yes --model Maya-L --download-model
```

`./saguaro.sh --bench` on one MI325X (v1.1.27, Maya-L, 1M context, `--kv-streaming off`): answers at 84.0 tokens/s,
prompts at 1307 tokens/s (2K tokens) and 1663 tokens/s (8K tokens). Not tried yet: the MI300X itself, more than one GPU.

## Windows

`START-SAGUARO.bat --backend hip` sets Saguaro up natively on Windows 10/11 for the same GPUs, Strix Halo and Gorgon Halo
included (Ryzen AI Max 300 / 400, Radeon 8050S / 8060S / 8065S, all `gfx1151`). It compiles the engine with ROCm's
clang in Visual Studio's environment, with Ninja, as `tools\hip\build_maya_windows.bat` does (#54). The download, the
pack and the dashboard are the same as on an NVIDIA PC.

1. Install once: a current AMD driver (AMD Software: Adrenalin Edition),
   [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 2022 or 2026 with
   "Desktop development with C++", 64-bit Python 3.12 (`winget install -e --id Python.Python.3.12 --scope user`),
   Git, and [AMD's HIP SDK for Windows](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) (7.2 is
   measured below; without it the setup offers AMD's ROCm SDK wheels).
2. On an APU, give the GPU most of the memory: Variable Graphics Memory in AMD Software (Performance > Tuning), or
   the iGPU memory size in the BIOS - 160 GB of 192 GB on the PC below, 96 GB on a 128 GB one. Restart.
3. Then:

   ```bat
   START-SAGUARO.bat --backend hip --gpu 0 --check
   START-SAGUARO.bat --backend hip --gpu 0 --setup --model glm-maya-l
   ```

The setup takes the ROCm it finds first: `ROCM_PATH`, TheRock's `rocm-sdk` (`ROCM_VENV`, Saguaro's `.venv` or PATH), AMD's
HIP SDK (`HIP_PATH`, else the newest in `C:\Program Files\AMD\ROCm`). With none it offers AMD's ROCm SDK wheels in
Saguaro's `.venv` (pip, from `repo.amd.com/rocm/whl-multi-arch`: `rocm[libraries,devel,device-<arch>]==7.14.1`, the first
with Gorgon Halo); `--check` prints that command instead. GPU numbers are hipInfo's (HIP's own order); without
hipInfo, Windows' AMD display adapters in the registry's order.

The build copies the ROCm SDK's HIP runtime (`amdhip64_7.dll`, `amd_comgr*.dll`, `rocm_kpack.dll`) next to
`build-hip\saguaro-glm.exe`: Windows would otherwise load the driver's own from System32 before the SDK's.

**Memory on a Windows APU.** Windows gives the APU's GPU a fixed carve-out and does not count it as RAM: the 192 GB PC
below, with 160 GB of Variable Graphics Memory, shows 32 GB of RAM, and HIP reports 171.9 GB (the carve-out and three
quarters of Windows' shared half of its RAM). The engine therefore sizes it like a discrete card - the expert pool from
the GPU memory HIP reports free, the RAM tier from the free RAM and commit, the rest read from the SSD - and not from
`MemAvailable` as on Linux above. Windows' HIP runtime allocates at most 64 GiB plus the RAM Windows sees in one
piece (95.7 GiB there), so a larger pool goes into several allocations, each holding whole layers. Setup writes a
3 GiB reserve (the desktop runs on the same GPU), `SAGUARO_GLM_PREFILL_SUB=1024`, and a 6144 MiB prompt budget when RAM
and GPU memory together are at least 96 GiB (4096 MiB otherwise).

**The page file on a Windows APU.** Windows charges every allocation on the APU's GPU to its commit - RAM plus page
file - although it does not count the carve-out as RAM: on the PC below, the free commit fell by each of the engine's
GPU allocations, whatever their size. So the page file must be at least about the GPU's share. With less, either the
experts that do not fit the commit stay off the GPU (slower answers), or other programs run out of memory while the
model runs. Both engines (GLM-5.3-Flash's and DeepSeek's) keep their GPU experts within what Windows can still
commit, less max(8 GB, 10% of the RAM) for the other programs, and say how many GB of experts that left off the GPU,
and the page file that keeps them on it, in their log, the dashboard's Monitor and `/v1/status`. `--check` and the setup ask for the GPU's share plus max(8 GB, 10% of the RAM) - the RAM
holds Windows and the other programs - and warn when the page file is smaller: set a custom size (initial and maximum,
in MB) of at least that under System > About > Advanced system settings > Performance > Settings > Advanced > Virtual
memory > Change, then restart Windows. It is a reserve more than a file in use: the PC below ran with 160 GB of page
file for its 160 GB GPU share (the setup asks for 180 GB there: at 160 GB about 6 GB of commit was left while the
model ran), and at most 12 GB of it was ever written.

**Kernel submission.** Windows' HIP runtime keeps launches in a batch until the host waits on the GPU; Linux submits
each one. The engine's service thread waits for routes the GPU writes into host memory, so after 1 ms without one it
submits the batch (`cudaStreamQuery`, which does not wait). `GPU_FLUSH_ON_EXECUTION=1`, the runtime's own switch to
submit every launch, also works but costs ~31 us a launch.

Checked on a Ryzen AI Max+ PRO 495 / Radeon 8065S (Gorgon Halo, 192 GB LPDDR5X, 160 GB of it the GPU's), Windows 11,
HIP SDK 7.2 (HIP 7.2.60201), driver 32.0.31041, Maya-L, 32K context: all 15 HIP checks of [Validation](#validation)
pass, and every expert sits in the GPU pool (134.3 GB, 288 slots a layer, warmed from the NVMe in 44 s).

## Prompt speed

Prompts run in chunks, and inside a chunk the mixers and the dense FFN run in
sub-batches. Two settings matter on AMD:

- `SAGUARO_GLM_PREFILL_SUB` sets the sub-batch size. It was a fixed 256; it is
  now a runtime setting with 256 as the default. Larger sub-batches give the
  matrix cores bigger GEMMs. Setup writes `1024` together with
  `SAGUARO_GLM_PREFILL_MB=4096` for cards with 20 GB or more; the extra prompt
  memory is borrowed from the expert pool only while a prompt runs.
- `SAGUARO_HIPBLASLT_TUNING` points at a table of hipBLASLt solutions for the
  prompt's FP16 projections, measured per architecture with
  `tools/hip/tune_hipblaslt` (`tools/hip/<arch>-glm-hipblaslt-<version>.txt`).
  Setup writes it when a table for the card exists. The engine accepts a table
  only for its own architecture and hipBLASLt version; otherwise, and for shapes
  without a row, it keeps plain hipBLAS. Builds of one hipBLASLt version can
  number their solutions differently, so a version may have more than one table
  (`<version>.txt`, `<version>-b.txt`): the engine keeps every matching table
  and takes, per shape, the first whose solution its hipBLASLt accepts. The tuner's default shapes are
  DeepSeek's, so a table generated from them covers none of another model's
  projections: pass `--preset glm` for GLM-5.3-Flash (or `--shape T,N,K`) and
  check the engine's log, which now prints one line per shape that misses the
  table. On a 8060S, four GLM rows took a prompt from 128 to 168 tok/s. The tuner's absolute-error gate now
  scales with `sqrt(K / 4096)`, so long-K projections are no longer rejected.

Measured prompt speed, GLM-5.3-Flash Maya-S-v2 IQ2_XXS, ROCm 7.2:

| Setting | RX 7900 XT | Ryzen AI Max+ 395 (8060S) | R9700 |
|---|---|---|---|
| prompt chunk forced to 256 (the earlier config) | ~50 tok/s | ~70 | - |
| engine-sized chunk | ~198 | ~116 | - |
| + tuned hipBLASLt | ~253-259 | ~195 | - |
| + `SAGUARO_GLM_PREFILL_SUB=1024`, `SAGUARO_GLM_PREFILL_MB=4096` | ~413 | ~217 | ~490-560 |

The 8060S runs used the same prompt-path changes on a local build. The R9700
runs used a pinned RAM tier sized for every expert the VRAM pool does not hold. On
gfx1201, plain hipBLAS is already close to hipBLASLt for most shapes, so its
table keeps only the rows that were at least 2% faster.

## What changed

- Added HIP GPU detection, compilation, rebuild stamps, and serving configs to
  `saguaro.py`, with separate CUDA and HIP build directories.
- Completed the CUDA-shaped runtime/BLAS mappings needed by GLM, excluded
  NVIDIA's profiling ranges from HIP, and linked the GLM prompt path to HIP MMQ
  and hipBLAS.
- Replaced the GLM hyper-connection kernel's unsupported FP32 `__ldcg` with
  agent-scope acquire loads on HIP. CUDA keeps its existing implementation.
- Kept CUDA WMMA attention guarded to CUDA builds. HIP uses the F32 attention
  kernel; since v1.0.7 its 32-row tile keeps the latent rows FP16 in shared
  memory (34 KB), within RDNA's 64 KiB workgroup limit.
- Fixed the KDA parity fixture to initialize its recurrence state explicitly;
  its numerical tolerance is unchanged.
- The HIP post-store system fences in both shared doorbell kernels and Saguaro's
  fused routing signal (backported from Strata) arrived separately in v1.0.7
  (#2); the validation below includes them.

Upstream Strata was inspected at `d5ea713` (0.1.40.3). Its AMD backend documents
runtime selection, memory limits, desktop VRAM headroom, and newer packed-byte
optimizations. Saguaro's GLM engine is separate from Strata's current Qwen engine,
so Strata's model benchmarks do not establish Saguaro performance.

## Validation

Test host: RX 7900 XT 20 GiB, Ryzen 7 9800X3D, 192 GB installed DDR5, native Ubuntu Linux, system ROCm 7.2.1 / Clang 22, plus AI PRO R9700 32 GiB. Saguaro base v1.0.10 (`e4bd57e`).

The HIP engine and selected test targets build. All 14 selected GPU checks
pass: device allocation, expert upload staging, packed-byte/shuffle
intrinsics, asynchronous mapped-memory handoff, IQ1_S arithmetic, GLM
hyper-connections, KDA, FFN, DSA, layer arithmetic, synthetic-model logits
against the committed reference, and HIP MMQ. Six isolated installer tests
pass. The fused HC decode kernel also agrees with the independent CPU
reference across eight consecutive 4096-wide steps, including in-place gate
updates, in all three variants (default HC3, forced HC2, and forced HC1).
The full-model runs are below.

The real batched MLA attention kernel agrees with a double-precision CPU
softmax reference (absolute tolerance `5e-5`), including empty and masked cell
lists, partial and multiple tiles, and multiple head groups. It reads the
FP16 latent cache added in Maya v1.0.4.

The GLM handoff test replays the real routing graph for 100 disk requests
and 100 CPU-lane requests with changing IDs, weights, inputs and answers. The
CPU observes each request without a driver query or stream synchronization.
The same multi-architecture engine and test build pass all 14 checks on the
7900 XT and R9700. HC1 and HC2 also pass on the R9700. The HIP intrinsics check
covers all 65,536 pairs of byte values with varying neighboring lanes and
permutation selectors, against per-byte CPU references. The runtime rejects
the uncompiled `gfx1036` integrated GPU with the actual device and compiled
architecture list in its message. See [R9700 build details](AMD_RDNA4.md) and
[the shared AMD backport audit](AMD_UPSTREAM.md).

```sh
python3 tools/test_maya_hip.py
LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -R '^(glm_(hc|kda|ffn|dsa|layer)_parity|glm_model_test|hip_intrinsics|hip_(glm_handoff|glm_prefill_attention|handoff|device_selftest|prefill_mmq_parity|expert_cache_staging)|iq1_s_parity)$'
SAGUARO_GLM_HC2=1 LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  build-hip/glm_hc_parity --selftest
SAGUARO_GLM_HC1=1 LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  build-hip/glm_hc_parity --selftest
```

The relevant build targets must be built before running CTest. Build and test
logs are kept in `build-hip/`.

Full model (Maya-S, 8K context) on the RX 7900 XT with the config setup writes:
seven smoke requests (arithmetic, code, a two-turn conversation, three
400-token answers) and five back-to-back prompt+answer rounds complete, with
prompts at 307-385 tok/s on 1.8-2.5K-token prompts and the hipBLASLt table
loaded. The R9700 runs the full model with every expert in VRAM or a pinned
RAM tier; the prompt speeds above were measured there.


### Confirmed working 
-  Test host: RX 7900 XT 20 GiB, Ryzen 9 7900X, 96 GB installed DDR5, decokerized Ubuntu Linux, system ROCm 7.2.4, Saguaro base v1.1.46 (`3589d6f`). 
- Model : Maya-S24, 1M context
