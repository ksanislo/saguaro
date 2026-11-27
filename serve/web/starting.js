// serve/web/starting.js - the screen that stands on the port before the model is up (issue #20 phase 1): the page
// opens while the engine still loads and waits with it - the model's name, how long the load has run, what the
// engine was last saying - and becomes the normal page the second /health answers 200 (app.js' watchStarting).
// A load that failed shows its error and, when another config loaded before, a button that starts that one again;
// the server says no last_good under a supervisor (it owns the starts), so the button appears only where it works.
import {html, useState} from "./preact.js";
import {ago, api, modelLabel} from "./lib.js";
import {S, emit, useStore} from "./state.js";

// the screen's text from a starting /health (pure, unit-tested): how to name the model, how long it has run, the
// engine's last line, and a failed load's error and offer
export function startingView(h) {
  h = h || {};                                     // (S.starting is null on the first paint: a default {} is not)
  const failed = h.status === "failed";
  return {
    state: h.status || "waiting",                  // (waiting: /health has not answered to this page yet)
    failed,
    name: h.model ? modelLabel(h, {compact: true}) : "",
    elapsed: h.elapsed_s != null ? ago(+h.elapsed_s || 0) : "",
    line: h.line || "",
    // (P125m: the load's progress is the engine's own lines, mapped by the server: a percent and the step's words)
    pct: !failed && h.pct != null && h.pct !== "" ? Math.max(0, Math.min(100, Math.round(+h.pct || 0))) : null,
    step: failed ? "" : (h.step || ""),
    error: failed ? (h.error || "the model did not start") : "",
    lastGood: h.last_good || null,                 // ({config, time}: the server says none under a supervisor)
    restarting: h.restarting || null,              // (this page asked: waiting for that start to answer)
  };
}

// the failed page's button: start again with the config that worked (the server notes it and ends with RESTART_EXIT)
export async function startLastGood() {
  const j = await api("api/last-good/start", {method: "POST", body: {}});
  S.starting = {...(S.starting || {}), restarting: j.starting};
  emit();
  return j;
}

export function StartingPage() {
  useStore();
  const v = startingView(S.starting);
  const [err, setErr] = useState("");
  const start = () => { setErr(""); startLastGood().catch((e) => setErr(e.message)); };
  return html`<div class="start">
    <div class="start__card">
      <img class="start__logo" src="web/icon.svg" alt="" />
      ${v.failed ? html`
        <h1 class="start__title">The model did not start</h1>
        ${v.name ? html`<p class="start__model">${v.name}</p>` : null}
        <p class="start__error">${v.error}</p>
        ${v.line ? html`<p class="start__meta"><code>${v.line}</code></p>` : null}`
      : v.state === "loading" ? html`
        <div class="start__spin" aria-hidden="true"></div>
        <h1 class="start__title">Loading the model</h1>
        ${v.name ? html`<p class="start__model">${v.name}</p>` : null}
        ${v.pct != null ? html`
          <div class="start__prog" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${v.pct}>
            <i style=${`width:${v.pct}%`}></i></div>
          <p class="start__meta">${v.step ? html`<b>${v.step}</b> · ` : null}${v.pct}%${v.elapsed ? ` · it has been running for ${v.elapsed}` : ""}</p>`
        : html`<p class="start__meta">${v.elapsed ? `it has been running for ${v.elapsed}` : ""}${v.line ? html` · <code>${v.line}</code>` : null}</p>`}`
      : html`
        <div class="start__spin" aria-hidden="true"></div>
        <h1 class="start__title">Saguaro</h1>`}
      ${v.restarting ? html`<p class="start__meta">Starting again with <b>${v.restarting}</b> …</p>` : null}
      ${v.failed && !v.restarting && v.lastGood ? html`
        <button type="button" class="btn btn--primary" onClick=${start}>Start with ${v.lastGood.config} · the last config that worked</button>` : null}
      ${err ? html`<p class="start__error">${err}</p>` : null}
      ${v.failed && !v.restarting && !v.lastGood
        ? html`<p class="start__hint">Fix what that line says, then start Saguaro again.</p>` : null}
    </div>
  </div>`;
}
