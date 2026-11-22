"""Inkling 2 Mini: a Strata pack's trunk tensors under llama.cpp master's GGUF names (the "rewrite" layout
saguaro-ik2 reads), values unchanged - the inverse of Strata's engine/src/inkling2/ink_gguf_loader.h (MIT).

trunk_map(pack) -> {gguf_name: (strata_name, transform)}; transform(f32 array in the pack's shape) -> GGUF-order array.
Only permutations: no arithmetic, so a mapped tensor holds the pack's values exactly.
"""
from __future__ import annotations

import numpy as np


def _same(a, shape):
    return a.reshape(shape)


def trunk_map(manifest):
    tc = manifest["config"]["text_config"]
    n_layer = tc["num_hidden_layers"]
    loop = tc.get("loop_config", {})
    core0, core1 = loop.get("core_start", 0), loop.get("core_end", 0)
    local = set(tc["local_layer_ids"])
    C, KV = tc["hidden_size"], tc["num_key_value_heads"] * tc["head_dim"]
    k = tc["sconv_kernel_size"]
    m = {
        "token_embd.weight": ("model.llm.embed.weight", None),
        "token_embd_norm.weight": ("model.llm.embed_norm.weight", None),
        "output.weight": ("model.llm.unembed.weight", None),
        "output_norm.weight": ("model.llm.norm.weight", None),
        "output_res_query.weight": ("model.llm.attn_res.output_query", None),
    }
    if loop.get("num_passes", 1) > 1:
        m["loop_res_query.weight"] = ("model.llm.middle_cycle_cross_pass_queries", None)
        m["loop_kv_res_query.weight"] = ("model.llm.middle_cycle_cross_pass_kv_queries", None)
    for l in range(n_layer):
        b, p = f"blk.{l}.", f"model.llm.layers.{l}."
        swa = l in local
        for g, s in (("attn_q", "attn.wq_du"), ("attn_k", "attn.wk_dv"), ("attn_v", "attn.wv_dv"),
                     ("attn_rel", "attn.wr_du"), ("attn_sink_proj", "attn.attn_sink_proj"),
                     ("attn_gate", "attn.gate_attn"), ("attn_output", "attn.wo_ud"), ("ffn_gate_inp", "mlp.gate"),
                     ("ffn_norm", "mlp_norm"), ("attn_post_conv", "attn_sconv"), ("ffn_post_conv", "mlp_sconv")):
            m[b + g + ".weight"] = (p + s + ".weight", None)
        m[b + "exp_probs_b.bias"] = (p + "mlp.gate.bias", None)
        m[b + "attn_q_clip.weight"] = (p + "attn.q_l2_clip.scale", None)
        m[b + "attn_rel_clip.weight"] = (p + "attn.rel_l2_clip.scale", None)
        # canonical [d_rel][extent]; master [extent][d_rel]
        m[b + "attn_rel_bias.weight"] = (p + "attn.rel_logits_proj.proj", lambda a: np.ascontiguousarray(a.reshape(tc["d_rel"], -1).T))
        if swa:
            m[b + "attn_norm.weight"] = (p + "attn_norm.weight", None)
            m[b + "attn_k_conv.weight"] = (p + "attn.k_sconv.weight", None)
            m[b + "attn_v_conv.weight"] = (p + "attn.v_sconv.weight", None)
        else:
            for x in "qkv":
                m[b + f"attn_{x}_inp_norm.weight"] = (p + f"{x}_input_norm.weight", None)
            # canonical static_w [k][KV] lag 0..k-1; master [KV][k] lag k-1..0
            for x in "kv":
                m[b + f"attn_{x}_conv.weight"] = (p + f"attn.{x}_sconv.static_w",
                                                  lambda a: np.ascontiguousarray(a.reshape(k, KV)[::-1].T))
                m[b + f"attn_{x}_conv_proj.weight"] = (p + f"attn.{x}_sconv.weight_proj", None)
        if l > 0:
            m[b + "attn_res_query.weight"] = (f"model.llm.attn_res.attn_queries.{l - 1}", None)
        m[b + "ffn_res_query.weight"] = (f"model.llm.attn_res.mlp_queries.{l}", None)
        if core0 <= l < core1:
            for md in ("attn", "k", "v", "mlp"):
                for g in ("down", "gain", "forget"):
                    m[b + f"loop_{md}_{g}.weight"] = (p + f"controllers.{md}.{g}_weight", None)
    return m
