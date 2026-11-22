#!/usr/bin/env python3
"""bench/hy4/make_prompts.py OUT_DIR PACK - the bench's prompts as token ids (the pack's template and tokenizer, no GPU):
short.ids, 10k.ids, 60k.ids - a question after 0 / ~10K / ~60K tokens of this repo's docs, thinking off."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST
from serve.frontend import ChatTemplate

out, pack = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tk = ST.Tokenizer.from_dir(pack / "tokenizer")
tpl = ChatTemplate(pack / "tokenizer" / "chat_template.jinja")
text = "".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted((ROOT / "docs").glob("*.md")))
text += (ROOT / "CHANGELOG.md").read_text(encoding="utf-8", errors="replace")
Q = "Write a detailed, numbered explanation of how a hash table works, with collision handling and resizing."
ids_text = tk.encode(text)
for name, n in (("short", 0), ("10k", 10000), ("60k", 60000)):
    body = b"".join(tk.token_bytes(i) for i in ids_text[:n]).decode("utf-8", "replace")
    user = (f"[{name}] Here is some documentation:\n\n{body}\n\nIgnore the documentation above. " if n else f"[{name}] ") + Q
    p = tpl.render([{"role": "user", "content": user}], enable_thinking=False)
    ids = tk.encode(p, parse_special=True)
    (out / f"{name}.ids").write_text(",".join(map(str, ids)))
    print(name, len(ids))

# varied short requests (HY4_HOT's learning, drafting's acceptance): code, prose, reasoning, a list, another language
VARIED = ("Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
          "Explain to a new sysadmin how TCP congestion control works, in plain paragraphs.",
          "A farmer has 17 sheep, sells 5, buys 9, and loses a third of the flock to a storm. Reason step by step: how many remain?",
          "Write a short story about a lighthouse keeper who finds a message in a bottle.",
          "Give a C++ class for a fixed-size ring buffer with push, pop and iteration, and comment each method.",
          "用中文简要介绍一下光合作用的过程和意义。",
          "Summarize the causes and consequences of the 1929 stock market crash as a bulleted list.",
          "Write a bash script that rotates log files older than 7 days and explain each line.")
for i, q in enumerate(VARIED):
    for think in (False, True):
        p = tpl.render([{"role": "user", "content": q}], enable_thinking=think)
        (out / f"v{i}{'t' if think else ''}.ids").write_text(",".join(map(str, tk.encode(p, parse_special=True))))
print("varied", len(VARIED))
