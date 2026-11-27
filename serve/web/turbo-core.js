// serve/web/turbo-core.js - Turbo's request-level half, as plain functions (no page, no storage; serve/test_turbo.py runs them
// under node): what a chat's stored settings become when Turbo goes on, and what comes back when it goes off.  The engine's
// half (the context, the expert skipping, the drafting) is the server's: serve/models.py turbo_enable / turbo_disable.

// the fastest thinking level a model has: none where it offers it (GLM, DeepSeek, MiniMax), else its lowest (Qwen's low)
export const fastestThinking = (levels) => (levels.includes("none") ? "none" : levels[0]);

// Turbo on: `stored` is what the browser holds for this model's chat settings (only what its user chose; null for nothing).
// The snapshot keeps it whole, so that going off puts back exactly it; `next` is what Turbo stores: greedy decoding
// (temperature 0 - the other samplers do nothing then) and the fastest thinking level.
export function turboEnter(stored, levels, chatThinking = null) {
  const own = stored && typeof stored === "object" ? stored : null;
  const think = fastestThinking(levels);
  return {
    snapshot: {stored: own ? JSON.parse(JSON.stringify(own)) : null, chatThinking, think},
    next: {...(own || {}), temperature: 0, thinking: think, prefs_v: 3},
  };
}

// Turbo off: the stored settings as they were ({} for none: the model's own defaults).  The open chat's level goes back too,
// unless its user changed it since (it still reads Turbo's level).
export function turboLeave(snapshot, chatNow = null) {
  if (!snapshot) return {stored: null, chatThinking: undefined};
  const back = snapshot.chatThinking != null && chatNow === snapshot.think ? snapshot.chatThinking : undefined;
  return {stored: snapshot.stored ? JSON.parse(JSON.stringify(snapshot.stored)) : {}, chatThinking: back};
}

// the ideas a new chat offers while Turbo is on: coding, all of them
export const CODING_IDEAS = [
  {icon: "code", label: "Write a Python function",
   q: "Write a Python function that merges overlapping intervals in a list of (start, end) pairs, with a few tests."},
  {icon: "box", label: "A three.js scene in one HTML file",
   q: "Write a single-file HTML page with a three.js scene: a slowly rotating, softly lit crystal on a dark background, with orbit controls."},
  {icon: "tool", label: "Explain and improve some code",
   q: "Explain step by step what this code does, then suggest two improvements:\n\n```\n\n```"},
  {icon: "code", label: "Fix a bug",
   q: "This function should return the median of a list but is wrong for even lengths. Find the bug and fix it:\n\n```python\n\n```"},
];

// ---- live per-request Turbo (PLAN-turbo-live): the chat's own lossy options, sent with every turn of it and
// changeable mid-conversation (they apply from the next turn).  `chat.turbo` holds {"drop": 0-3, "skip":
// "off"|"ssd"|"ram"}, or nothing while the chat is exact.  serve/turbo_live.py reads the same shape on the wire.

// the words the Settings rows and the chat menu show (PLAN-p5: name the options plainly, and say out loud what
// the RAM tier risks); warn: only RAM carries one, the pages show it under the row / in the item hint.
export const TURBO_DROPS = [1, 2, 3];
export const TURBO_DROP_HINT = "fewer of the 10 routed experts a token, the weakest dropped";
export const TURBO_SKIPS = [
  {id: "off", name: "off", hint: "every expert the router picks is used"},
  {id: "ssd", name: "SSD", hint: "an expert that would be read from the SSD is left out"},
  {id: "ram", name: "RAM", hint: "an expert that is not on the card is left out",
   warn: "can break answers: only where almost everything is on the card"},
];

// the warning a skip level carries ("" for off and SSD)
export const turboSkipWarn = (skip) => (TURBO_SKIPS.find((s) => s.id === skip) || {}).warn || "";

// the chat's Turbo as the request sends it (null = exact: the field is not sent at all)
export function turboOfChat(chat) {
  const t = chat && typeof chat === "object" ? chat.turbo : null;
  if (!t || (typeof t !== "object")) return null;
  const drop = Number.isInteger(t.drop) && t.drop > 0 && t.drop <= 3 ? t.drop : 0;
  const skip = ["ssd", "ram"].includes(t.skip) ? t.skip : "off";
  return drop || skip !== "off" ? {drop, skip} : null;
}

// flipping the toggle: on starts at drop 1 / skip off; off clears the chat's options
export function turboToggle(chat) {
  return turboOfChat(chat) ? null : {drop: 1, skip: "off"};
}

// one choice inside the menu (drop 1/2/3, skip off/SSD/RAM) applied to what the chat has now; an exact chat that
// picks a skip level joins Turbo with drop 0
export function turboChoose(chat, key, value) {
  const cur = turboOfChat(chat) || {drop: 0, skip: "off"};
  const next = {...cur, [key]: value};
  return turboOfChat({turbo: next}) ? next : null;
}

// the Monitor's badge text for a request's options ("" when exact) - the page's own words for the dict the
// monitor rows carry (serve/server.py sends it raw; the old Python twin in serve/turbo_live.py read nothing)
export const turboBadge = (t) => {
  if (!t || typeof t !== "object") return "";
  const parts = [];
  if (t.drop) parts.push(`-${t.drop} experts`);
  if (t.skip && t.skip !== "off") parts.push(({ssd: "no SSD", ram: "VRAM only"})[t.skip] || t.skip);
  return parts.join(" · ");
};
