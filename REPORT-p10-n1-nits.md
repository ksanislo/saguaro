# P10 report - the four nits left from the model-names review (./ref/nits.md findings 1-4)

One commit per finding, each with a test; the run list was only the six modules the plan names.
Nothing under engines/, saguaro.py or serve/qfn_engine.py was touched, and the llama-swap gate in
poll() is exactly as it was.  Commits (base 1d5cee8ef):

- b3d7c6bbb finding 1, 70cd5a92f finding 2, e73efaf92 finding 3, ccfb8f5f1 finding 4

## Finding 1 - the quant-only case is asserted now  (b3d7c6bbb)

serve/test_new_engines_ui.py:585: `self.assertTrue(got["quantOnly"])` added to the assertion block of
test_an_open_page_follows_a_model_switch, between the otherLabel and saysNothing lines.  The case was already
built (:567-568, same id, same label, only `model_quant` moved); only the read-back was missing, so the
`model_name_pretty`/`model_quant` loop in serve/web/server.js (`for (const k of [...])`, :64-65 now) was unpinned.

Pin-check: with the loop the case is true; the three other new gates below were likewise shown to fail when
the fix is removed (finding 2 checked directly, see below).

## Finding 2 - no duplicate /health on page load  (70cd5a92f)

serve/web/server.js:
- `let healthSeen = false;` (:50, beside healthWait) - set in loadHealth right after `S.health = h` (:19).
- poll()'s re-read gate (:118-119): the identityChanged term is now `(healthSeen && identityChanged(...))`;
  the `conn === "down" || conn === "key"` terms and the whole llama-swap gate (swapReady / `back` /
  swapRunning) are untouched.
- the comment is corrected: the first poll asks nothing because its `S.health` is still state.js'
  placeholder and the identity term waits for healthSeen - the old comment only claimed app.js' own
  loadHealth was on its way, which the identity term did not honor.

New test serve/test_dashboard.py::PageLogic::test_the_first_poll_of_a_page_load_asks_no_health_of_its_own:
the first poll (health not yet read) logs `["metrics"]` only; after loadHealth lands, an unchanged engine logs
`["metrics"]`, and a different engine logged by the metrics after healthSeen still re-reads once -
`["metrics", "health"]`.  Verified unpinned-without-the-fix: with `healthSeen &&` removed the test fails.

## Finding 3 - one quant namer, and it knows the off-list labels  (e73efaf92)

Picked side: the namer, not modelLabel.  The pick-side alternative (lib.js falling back to the row's `quant`
when `model_quant` is empty) would not have fixed the named requirement: the abliterated config's
`model_quant` is not empty, it is the *wrong* string `abliterated`, so an empty-only fallback keeps the bare
key on screen.

- tools/maya_quants.py: added `OTHER_QUANTS = {("deepseek", "abliterated"): "Mix-256G"}` (:153, the map
  serve/models.py used to hold) and quant_name reads it between OTHERS and the bare key (:356-358), with
  models.quant_label's `(fam, str(key).lower())` normalization.
- serve/models.py: `OTHER_QUANTS = maya_quants.OTHER_QUANTS` (:60) - one map, quant_label (:855) unchanged,
  so its file-stem last fallback (the hand-made configs) stays exactly as it was.
- Effect: a config installed under saguaro.py's old `DS4_ABL_ALIAS` shows `DeepSeek V4.1 (Mix-256G)` again in
  every modelLabel place (header, sidebar, Models, Saved) - /health's and /metrics' model_quant come from
  display_names -> quant_name.

New test tools/test_maya_quants.py::Aliases::test_quant_name_knows_the_other_quants_labels: quant_name's
abliterated -> Mix-256G, the newer `abliterated-256g` (OTHERS), a tier and an unknown key unchanged, and
display_names of such a config -> `("DeepSeek V4.1", "Mix-256G")` (what /health, /metrics and the /api/models
rows carry).

## Finding 4 - the two raw-id tooltips carry the name  (ccfb8f5f1)

serve/web/chat-view.js:
- the chat header's model button (:70): `title=${S.health.model}` -> `title=${name}` (the compact one label,
  same as the phone header's button which had no title at all and the sidebar's `title=${name}`).
- the new chat's model card (:469): `title=${h.model || ""}` -> `title=${name}`.

Test: serve/test_server.py::test_the_model_shown_model_first - the test that pinned the sidebar's identical
choice - now also reads chat-view.js and refuses `title=${S.health.model}` and `title=${h.model ||`,
requiring `title=${name}`.

## Tests run (python -m unittest, one module at a time, final tree)

| module                  | result |
|-------------------------|--------|
| serve.test_dashboard    | 79 tests, OK (the new poll test included) |
| serve.test_new_engines_ui | 38 tests, OK |
| serve.test_models       | 63 tests, OK |
| serve.test_server       | 178 tests, OK |
| serve.test_model_api    | 28 tests, OK |
| tools.test_maya_quants  | 26 tests, OK (1 skip, pre-existing: setup_look's Textual/rich not installed here) |

No test pinned an old behaviour this plan changed; the two label tests (test_dashboard:959 "no quant to say",
test_models:641 the hand-made suffix case) keep passing untouched.

## Open questions

- nit 3's *second* half is deliberately not changed: a hand-made config with no recognised quant suffix
  still has an empty model_quant (the header shows no badge), where models.quant_label once fell back to the
  config's file stem.  The named requirement was the abliterated label; the empty case is pinned as intended
  by test_dashboard.py:959 and the Models/Saved rows still carry `quant`.  A page-side stem fallback would be
  the follow-up if the manager wants it.
- nit 5 (Saguaro's own version under the model's name in the model menu, chat-view.js:35) stays as reported -
  the plan takes findings 1-4 only; it awaits the manager's ruling like REPORT-n1-model-names.md said.
