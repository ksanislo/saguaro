// serve/web/turbo.js - the Turbo button (Settings > Chat defaults, beside the model's parameters).  On: greedy decoding and
// the fastest thinking level in this browser's settings, and - in the engine's run config, the same "settings saved, Saguaro
// starts again" flow as the Models page - a 32K context (GLM, MiniMax), expert skipping and the fastest drafting where the
// engine has them.  Off: every one of those back as it was.  The snapshot of the page's settings is kept here (localStorage);
// the engine's is in the run config (serve/models.py).  Below it lives the LIVE per-request Turbo (PLAN-turbo-live): the
// model's global options for requests that carry none of their own - engine state only, applied at once, no config rewrite,
// no reload, and it stays visible under llama-swap (it never touches configs there).
import {html} from "./preact.js";
import {api, esc, store} from "./lib.js";
import {S, THINKING, P, confirmDialog, emit, loadPrefs, prefsKey, toast} from "./state.js";
import {loadTime} from "./server.js";
import {TURBO_DROPS, TURBO_SKIPS, turboEnter, turboLeave} from "./turbo-core.js";
import {Seg, Switch} from "./ui.js";

S.turbo = {on: false, engine: false, plan: {}, env: {}, restarts: false, busy: false};

export async function loadTurbo() {
  try {
    const t = await api("api/turbo");
    S.turbo = {...S.turbo, ...t, busy: S.turbo.busy};
  } catch (e) { S.turbo = {...S.turbo, unavailable: true}; }   // (an older server: no Turbo)
  emit();
  return S.turbo;
}

const snapKey = () => `turbo_${prefsKey()}`;
const WHAT = {context: (v) => `context ${Math.round(v / 1024)}K`, parallel: () => "one answer at a time",
              draft: (v) => `${{dspark: "DSpark", mtp: "MTP", dflash2: "DFlash2"}[v] || v} drafting`, prompt_lookup: () => "prompt lookup"};

function engineLines(t) {
  const out = Object.entries(t.plan || {}).map(([k, v]) => (WHAT[k] ? WHAT[k](v.to) : `${k} ${v.to}`));
  if (t.env && Object.keys(t.env).length) out.push("expert skipping");
  return out;
}

// the request-level half, at once
function enterPrefs() {
  if (store.get(snapKey(), null)) return;                      // (already in Turbo here: the first snapshot stays)
  const chat = S.current && S.current.thinking != null ? S.current.thinking : null;
  const {snapshot, next} = turboEnter(store.get(prefsKey(), null), THINKING, chat);
  store.set(snapKey(), snapshot);
  store.set(prefsKey(), next);
  loadPrefs();
  if (S.current) S.current.thinking = P.thinking;
}
function leavePrefs() {
  const snap = store.get(snapKey(), null);
  if (!snap) return;
  const back = turboLeave(snap, S.current ? S.current.thinking : null);
  store.set(prefsKey(), back.stored);
  try { localStorage.removeItem("maya." + snapKey()); } catch (e) { /* ignore */ }
  loadPrefs();
  if (S.current) S.current.thinking = back.chatThinking !== undefined ? back.chatThinking : P.thinking;
}

// follow the restart the engine's part asks for, then load the page again with the model that runs
async function followRestart(on) {
  S.turbo = {...S.turbo, busy: true, restarting: true};
  emit();
  let down = false;
  for (let i = 0; i < 450; i++) {
    await new Promise((ok) => setTimeout(ok, 2000));
    let h = null;
    try { const r = await fetch("health", {cache: "no-store"}); h = r.ok ? await r.json() : null; } catch (e) { h = null; }
    if (!h) { down = true; continue; }
    if (down && h.status === "ok") {
      toast("success", on ? "Turbo is on" : "Turbo is off", "Reloading the page…", 3000);
      setTimeout(() => location.reload(), 900);
      return;
    }
  }
  S.turbo = {...S.turbo, busy: false, restarting: false};
  emit();
  toast("error", "Saguaro is not back yet", "The terminal it runs in shows why. This page reconnects when it is.", 30000);
}

export async function setTurbo(on) {
  const t = S.turbo;
  if (t.busy || on === t.on) return;
  const lines = engineLines(t);
  // the engine's part restarts it: say so first (the context and the experts are loaded again)
  if (on && t.engine && t.restarts && lines.length) {
    const ok = await confirmDialog("Switch Turbo on?",
      `<p>Greedy decoding, no thinking and the fastest settings the engine has: <b>${esc(lines.join(", "))}</b>.</p>` +
      `<p>Saguaro starts again with them, which takes ${esc(loadTime())}. Switching Turbo off puts everything back as it is now.</p>` +
      "<ul><li>Your chats stay here; the next answer reads the chat again.</li>" +
      "<li>Apps using the API get a “try again” until it is back.</li></ul>", "Switch Turbo on");
    if (!ok) return;
  }
  S.turbo = {...t, busy: true};
  emit();
  try {
    if (on) enterPrefs();
    const out = await api("api/turbo", {method: "POST", body: {on}});
    if (!on) leavePrefs();
    S.turbo = {...S.turbo, ...out, busy: false};
    emit();
    if (out.restarting_now) { followRestart(on); return; }
    toast("success", on ? "Turbo is on" : "Turbo is off",
          out.changed ? "The engine settings apply from Saguaro's next start." : on ? "Greedy decoding and the fastest thinking." : "Your settings are back.", 4000);
  } catch (e) {
    if (on) leavePrefs();                                      // (the engine said no: nothing stays half on)
    S.turbo = {...S.turbo, busy: false};
    emit();
    toast("error", on ? "Turbo did not start" : "Turbo did not switch off", e.message, 7000);
  }
}

// the button: a small flame that burns while Turbo is on (turbo.css)
export function TurboButton() {
  if (S.health && S.health.supervisor) return null;   // (under llama-swap the engine settings are its config)
  const t = S.turbo;
  if (t.unavailable || t.supported === false) return null;   // (Inkling 2: its engine has no Turbo)
  const tip = t.on ? "Turbo ON: trades quality for maximum speed (greedy decoding, no thinking, expert skipping, aggressive drafting). Click to put your settings back."
                   : "Turbo: trades quality for maximum speed, for benchmarks and playing with the engine. Your own settings come back when you switch it off.";
  return html`<button type="button" class=${`turbo${t.on ? " turbo--on" : ""}${t.busy ? " turbo--busy" : ""}`} aria-pressed=${t.on ? "true" : "false"}
      title=${tip} disabled=${t.busy} onClick=${() => setTurbo(!t.on)}>
    <svg class="turbo__flame" viewBox="0 0 24 24" aria-hidden="true">
      <path class="turbo__outer" d="M12 2c.6 3.2-1.2 4.8-2.8 6.7C7.5 10.6 6 12.4 6 15a6 6 0 0 0 12 0c0-2.2-1-3.9-2.2-5.2-.3 1.3-.9 2.1-1.8 2.6.6-3.8-.3-7.8-2-10.4z"/>
      <path class="turbo__inner" d="M12 22a3.6 3.6 0 0 0 3.6-3.6c0-1.6-1-2.6-1.9-3.6-.4 1-.9 1.5-1.5 1.8.2-1.7-.2-3.2-1.2-4.4-.3 2.1-2.6 3.1-2.6 6.2A3.6 3.6 0 0 0 12 22z"/>
    </svg>
    <span>${t.on ? "Turbo ON" : "Turbo"}</span>
  </button>`;
}

// ---- live per-request Turbo (PLAN-turbo-live): the model's global setting for the control panel.  Every request
// that does not carry its own `turbo` - the page's per-chat chip overrides it, API clients and agents included -
// runs with these lossy options.  Changes apply at once (from the next request), persist across restarts, and
// touch no config: unlike Maya's Turbo above there is no restart, and this stays visible under llama-swap.

S.turboLive = undefined;   // until loaded: null = exact (off); {"drop": N, "skip": "off"|"ssd"|"ram"}; still undefined = an older server has none

export async function loadTurboLive() {
  try {
    const t = await api("api/turbo/options");
    S.turboLive = t.supported === false ? undefined : t.turbo || null;   // (Inkling 2: none, as an older server)
  } catch (e) { S.turboLive = undefined; }   // (an older server: no live Turbo)
  emit();
}

export async function setTurboLive(next) {   // {"drop","skip"} (parts that stay are the current ones) or null
  const prev = S.turboLive;
  S.turboLive = next;
  emit();
  try {
    const t = await api("api/turbo/options", {method: "POST", body: {turbo: next}});
    S.turboLive = t.turbo || null;
    toast("success", S.turboLive ? "Live Turbo is on" : "Live Turbo is off",
          S.turboLive ? "Every request without its own Turbo runs with these options, from the next one."
                      : "Requests run exact again, from the next one.", 4000);
  } catch (e) {
    S.turboLive = prev;
    toast("error", "The live Turbo setting was not saved", e.message, 6000);
  }
  emit();
}

export function TurboLive() {
  if (S.turboLive === undefined) return null;   // older server: no live Turbo
  const t = S.turboLive;
  const on = (v) => setTurboLive({drop: t ? t.drop : 1, skip: t ? t.skip : "off", ...(v || {})});
  return html`<div class="set-head"><h3>Turbo, live</h3></div>
    <p class="set-intro">Lossy speed for every request this model answers that does not set its own - the apps, the
      agents and new chats. A chat's own Turbo (the bolt in its message box) wins over this. It applies to the next
      request as it is asked - nothing restarts - and a request without it stays exact when this is off.</p>
    <div class="set-row"><span><span>On</span><span class="set-row__sub">drop the weakest experts, keep an answer the model still stands behind</span></span>
      <${Switch} label="Live Turbo for this model" on=${!!t} onChange=${(v) => setTurboLive(v ? {drop: 1, skip: "off"} : null)}/></div>
    ${t ? html`<div class="set-row"><span><span>Drop experts</span><span class="set-row__sub">per token, the lowest router weights of the 10; the rest carry their share</span></span>
      <${Seg} label="Experts dropped" value=${String(t.drop)} onChange=${(v) => on({drop: +v})}
        options=${TURBO_DROPS.map((d) => ({id: String(d), label: `${d}`, title: `run top-10 minus ${d}: the ${d} weakest experts of every token dropped, the kept ones renormalised`}))}/></div>
      <div class="set-row"><span><span>Skip</span><span class="set-row__sub">an expert the engine would fetch the slow way is dropped instead</span></span>
      <${Seg} label="Skip tier" value=${t.skip || "off"} onChange=${(v) => on({skip: v})}
        options=${TURBO_SKIPS.map((s) => ({id: s.id, label: s.name, title: s.hint}))}/></div>` : null}`;
}
