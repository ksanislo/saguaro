#!/usr/bin/env python3
"""bench/hy4/kld_seed.py OUT PACK [N_CTX 512] [CHUNKS 8] - a --kld base file with this repo's docs as its tokens and empty
rows: `saguaro-hy4 --kld OUT --kld-save BASE` then writes a setting's own log-probabilities over them (the KL it
prints against the empty rows means nothing), and `--kld BASE` compares another setting to it."""
import pathlib, struct, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST

out, pack = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
n_ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 512
chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 8
tk = ST.Tokenizer.from_dir(pack / "tokenizer")
text = "".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted((ROOT / "docs").glob("*.md")))
ids = tk.encode(text)
step = max(n_ctx, (len(ids) - n_ctx) // chunks)
n_vocab = len(tk.tokens)
nv = 2 * ((n_vocab + 1) // 2) + 4
nrow = n_ctx - 1 - n_ctx // 2
with out.open("wb") as f:
    f.write(b"_logits_" + struct.pack("<iii", n_ctx, n_vocab, chunks))
    for c in range(chunks):
        f.write(struct.pack(f"<{n_ctx}i", *ids[c * step:c * step + n_ctx]))
    row = bytes(nv * 2)
    for _ in range(chunks * nrow):
        f.write(row)
print(out, n_ctx, chunks, n_vocab)
