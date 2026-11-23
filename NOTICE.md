# Saguaro

Saguaro is a fork of [Project Maya](https://github.com/peasantsmith) (itself a fork of Strata), run on a 4 x Tesla T4 box:
GLM-5.3-Flash, DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next (engines/qfn, from Strata), MiniMax-M3.1-Flash, Hy4-Flash and
Inkling 2 Mini (both engines from Project Maya v1.1.72) under one server.

Credits: Project Maya (Peasant Smith and the Project Maya contributors) and Strata (Niko1221 and the Strata contributors) -
their MIT License and copyright notices are kept unchanged in LICENSE and engines/qfn/LICENSE and apply to this code.
Model weights keep their own licenses (LICENSE.deepseek-v41, LICENSE.minimax-m31 - MiniMax Community License,
LICENSE.qwen-community-1.0, LICENSE.hy4-flash - Tencent Hy License; Inkling 2 Mini: Apache 2.0). Third-party pieces keep theirs (serve/web/fonts/OFL.txt, serve/web/LICENSE.lucide.txt, ...).

Strata packs (.strata): src/strata_fmt/ and tests/strata_ref.h port Strata v0.2.19's container reader, manifest
parser, S_K / SF4 / Q8_ROW decode arithmetic and Inkling name map (engine/src/artifact/strata_reader.cpp,
engine/src/inkling2/ink_json.h, ink_gguf_loader.h, engine/src/quant/sk_codecs.cpp; MIT License, Copyright (c) 2026
Strata contributors - https://github.com/Niko1221/Strata). Strata's packs on Hugging Face are Apache 2.0.

Names kept for compatibility: MAYA_* / STRATA_* settings, the engines' file names, session formats and version strings.
