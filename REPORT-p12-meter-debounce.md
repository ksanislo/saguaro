# P12 report: the context meter's off-card estimate stops stringifying the whole chat per keystroke

Branch `saguaro/q1-p12-meter-debounce`, one change commit (`83d060853`) plus this report.

## What changed

`serve/web/chats.js`, `updateCtxMeter()`'s `unloaded()` branch keyed its estimate on
`"off-card " + JSON.stringify(apiMessages(S.current, msgs, true))` - the whole chat rebuilt
into API message objects and serialized - on every key press (the composer's `onInput`
calls `updateCtxMeter()` per character). It now keys a new `offCardKey(msgs)` helper's cheap
fields:

- the chat's own `id` (switching chats always re-estimates: never another chat's figure),
- the chat's instructions' length (`chatInstructions` / Settings' new-chat instructions
  change the estimate),
- `S.editIndex` (where an edit cuts the message list),
- `draftChat()`'s message count and each message's text length plus its image/file counts,
- `S.draft.length` and `S.attachments.length` (the composer and the +menu).

The branch's shape is unchanged - re-estimate locally when the key moves or the meter holds
no figure, keep the `off-card` mark clear of the exact count's JSON `ctxKey` (the two key
forms can't collide: the exact key is a JSON body starting with `{`), still ask nothing of
the model off-card, still let the real `api/tokens` count ask once the gate opens. The
plan's other option - riding the same 400 ms `ctxTimer` - was ruled out: the existing
`test_the_meter_re_estimates_when_the_chat_changes_while_unloaded` requires the switch and
the draft change to re-estimate within 50 ms, which a 400 ms debounce cannot meet.

Untouched, as the plan required: everything else, `serve/server.py`, `engines/`,
`serve/qfn_engine.py` (no driver change, so `tools/saguaro/extract_qfn_engine.py` was not
run). The on-card path, `liveCtx()`, `apiMessages`, `estimateTokens` and all callers are
as they were.

## Tests

- New: `serve.test_dashboard.PageLogic.test_typing_off_card_never_stringifies_the_whole_chat` -
  node, model off the card behind llama-swap, `JSON.stringify` wrapped to count whole-chat
  estimates (its first argument an array): opening a chat, 200 rapid composer inputs, a
  switch to another chat and an edited message shortened in the composer stay at **at most
  2** whole-chat stringifications (with the fixed code: 0), while the meter's figure still
  moves with the draft, the switch and the edit re-estimate (`approx`, another chat's own
  figure), and `api/tokens` is still never asked off-card. Ran the same node body against a
  `scratch/` copy of the HEAD~1 `chats.js`: it counts 201 and fails, so the test pins the
  change. (The edit step edits b's only message and shortens it - re-keying an edit to the
  identical text legitimately leaves the estimate equal, which an earlier draft of the test
  hit.)
- Run list, one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest`:
  - `serve.test_dashboard`: 81 tests OK (80 before the new one; node v20 present, none
    skipped)
  - `serve.test_new_engines_ui`: 38 tests OK

## Open questions

- The cheap key moves on counts and lengths, not on content: an off-card edit that keeps
  the message count and every text length identical (swap "llama" for "cactus") does not
  re-estimate until the model is back on the card and `api/tokens` counts for real. That is
  the trade the plan's "cheap key (chat id, message count, the last message's length, the
  draft's length)" asked for; the meter already says "approximate" off-card.
- `estimateTokens` itself still runs per keystroke while the key moves (a character-count
  loop, no allocation) - the old code paid it too, and the draft's length changing every
  key press is exactly when it must: the whole-chat *build and stringify* is what left.
- The new test counts whole-chat estimates through a `JSON.stringify` wrap rather than an
  internal counter; any unrelated array serialization in the typed path would raise the
  count. None exists today (the asserts use "at most a couple", ≤ 2, for that headroom).
