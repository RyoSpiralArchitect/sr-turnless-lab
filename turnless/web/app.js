"use strict";
const $ = (id) => document.getElementById(id);
const t = window.I18N.t;
let backends = [], lastSnapshot = null;
const busyStates = new Set(["queued", "generating", "delaying"]);
const visibleKinds = new Set(["utterance", "yield", "forced_stop", "interrupt", "pause", "wake", "typing", "protocol_error", "provider_error", "discard", "flag", "annotation", "session_open", "session_close", "harness_error"]);
let token = "", sessionId = null, cursor = 0, status = "waiting", selected = null;
let events = [], known = new Set(), openaiReady = false, providerReady = {}, compatHint = "", mutationChain = Promise.resolve();
const storage = {
  get(key) { try { return sessionStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { sessionStorage.setItem(key, value); } catch { /* storage may be disabled */ } },
  remove(key) { try { sessionStorage.removeItem(key); } catch { /* storage may be disabled */ } },
};
let typingActive = false, typingLastSent = 0, typingTimer, creating = false, sendPending = false, toastTimer, pollErrors = 0;
let localInspection = null;
let pollPending = null;
let annotationTargets = [], annotationScope = "event", annotationPending = false;
const annotationDrafts = new Map();

function toast(message, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").className = "toast" + (error ? " error" : "");
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 7000 : 3000);
}
async function request(path, body = undefined) {
  const response = await fetch(path, {method: body === undefined ? "GET" : "POST",
    headers: {"X-Turnless-Token": token, ...(body === undefined ? {} : {"Content-Type": "application/json"})},
    body: body === undefined ? undefined : JSON.stringify(body)});
  let data;
  try { data = await response.json(); } catch { throw new Error(t("unreadableResponse")); }
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function enqueue(task) {
  const result = mutationChain.then(task);
  mutationChain = result.catch(() => {});
  return result;
}
function mutate(path, extra = {}) {
  const sid = sessionId;
  if (!sid) return Promise.reject(new Error(t("startFirst")));
  return enqueue(async () => {
    const data = await request(path, {session: sid, ...extra});
    if (sid === sessionId) ingest(data);
    return data;
  });
}
function node(tag, className = "", text = undefined) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function timeLabel(ms) { return (ms / 1000).toFixed(1) + "s"; }
function controlText(e) {
  if (e.kind === "yield") return t("yieldEvent", {n: e.emitted_in_cycle});
  if (e.kind === "forced_stop") return `harness stop · ${e.reason}`;
  if (e.kind === "interrupt") return `interrupt · ${e.reason === "human_typing" ? t("typingEvent") : e.reason}`;
  if (e.kind === "provider_error") return t("providerError");
  if (e.kind === "protocol_error") return t("protocolError");
  return e.kind;
}
function appendConversation(e) {
  if (e.kind !== "utterance" && !["yield", "forced_stop", "interrupt", "protocol_error", "provider_error"].includes(e.kind)) return;
  $("empty-state")?.remove();
  const stream = $("conversation");
  const shouldScroll = stream.scrollHeight - stream.scrollTop - stream.clientHeight < 110;
  if (e.kind === "utterance") {
    const article = node("article", "utterance " + e.speaker);
    article.dataset.seq = String(e.seq);
    article.append(node("div", "avatar", e.speaker === "human" ? "U" : "A"));
    const body = node("div", "utterance-content");
    const heading = node("div", "speaker-line");
    const pick = node("button", "speaker utterance-select", e.speaker === "human" ? "HUMAN" : "ASSISTANT");
    pick.type = "button";
    pick.title = t("selectUtterance");
    pick.addEventListener("click", () => selectEvent(e.seq));
    heading.append(pick, node("span", "event-time", `${timeLabel(e.at_ms)} · #${e.seq}`));
    body.append(heading, node("p", "", e.text));
    article.append(body);
    stream.append(article);
  } else {
    const cls = e.kind === "interrupt" ? "interrupted" : e.kind.includes("error") || e.kind === "forced_stop" ? "warning" : "";
    const row = node("div", "control-row " + cls);
    row.dataset.seq = String(e.seq);
    row.append(node("span", "control-mark"), node("span", "", controlText(e)), node("span", "event-time", `${timeLabel(e.at_ms)} · #${e.seq}`));
    if (["yield", "forced_stop"].includes(e.kind)) {
      const review = node("button", "text-button review-sequence-button", t("observeSequence"));
      review.type = "button";
      review.addEventListener("click", () => {
        selectEvent(e.seq, "cycle");
        const panel = document.querySelector(".observer");
        if (window.matchMedia("(max-width: 1160px)").matches) {
          panel.scrollIntoView({behavior: "smooth", block: "start"});
        } else {
          panel.scrollTop = 0;
        }
      });
      row.append(review);
    }
    stream.append(row);
  }
  if (shouldScroll) stream.scrollTop = stream.scrollHeight;
}
function renderTrace() {
  const box = $("trace");
  const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  const scroll = box.scrollTop;
  const all = $("trace-filter").value === "all";
  const shown = events.filter(e => all || visibleKinds.has(e.kind)).slice(-100);
  box.replaceChildren();
  if (!shown.length) { box.append(node("p", "trace-empty", t("traceEmpty"))); return; }
  for (const e of shown) {
    const warning = e.kind.includes("error") || e.kind === "forced_stop" || e.kind === "flag";
    const row = node("button", `trace-event${warning ? " warning" : ""}${selected === e.seq ? " active" : ""}`);
    row.type = "button";
    const content = node("div");
    content.append(node("strong", "", e.kind === "utterance" ? e.speaker + ".emit" : e.kind));
    content.append(node("span", "", e.text || e.reason || e.tag || e.error || (e.kind === "raw_output" ? (e.discarded ? "discarded · " : "") + e.raw : `#${e.seq} · ${e.source || "harness"}`)));
    row.append(node("time", "", timeLabel(e.at_ms)), content);
    row.addEventListener("click", () => selectEvent(e.seq));
    box.append(row);
  }
  box.scrollTop = stick ? box.scrollHeight : scroll;
}
function draftKey() { return `${annotationScope}:${selected}`; }
function rememberDraft() {
  if (selected) annotationDrafts.set(draftKey(), {tag: $("tag").value, note: $("note").value});
}
function observationTarget() {
  return annotationTargets.find(item => item.target_seq === selected && item.scope === annotationScope);
}
function targetObservations(target) {
  return events.filter(e => e.kind === "annotation" && e.target_seq === target.target_seq
    && (e.scope || "event") === target.scope);
}
function activeObservation(target) {
  const history = targetObservations(target);
  const replaced = new Set(history.map(e => e.supersedes));
  return history.filter(e => !replaced.has(e.seq)).at(-1);
}
function observationName(event) {
  return event.judgment === "unsure" ? t("judgmentUnsure")
    : event.judgment === "note" ? t("judgmentNote") : t("tag_" + event.tag);
}
function renderReviewNavigation() {
  const cycles = annotationTargets.filter(item => item.scope === "cycle");
  $("review-sequence").replaceChildren(node("option", "", t("chooseSequence")));
  $("review-sequence").firstChild.value = "";
  for (const target of cycles) {
    const terminal = events.find(e => e.seq === target.target_seq);
    const label = `#${target.target_seqs[0]}–${target.target_seq} · ${t("sequenceCount", {n: target.target_seqs.length - 1})} · ${terminal.kind}`;
    const option = node("option", "", (activeObservation(target) ? "✓ " : "") + label);
    option.value = String(target.target_seq);
    $("review-sequence").append(option);
  }
  $("review-sequence").value = annotationScope === "cycle" ? String(selected) : "";
  $("review-sequence").disabled = !cycles.length;
  $("next-unreviewed").disabled = !cycles.some(target => !activeObservation(target));
  $("latest-sequence").disabled = !cycles.length;
  $("review-progress").textContent = cycles.length ? t("reviewProgress", {
    n: cycles.filter(target => activeObservation(target)).length, total: cycles.length}) : t("noSequences");
}
function updateObservationChoice() {
  const value = $("tag").value;
  $("tag-help").textContent = t(value === "__unsure" ? "unsureHelp" : value === "__note" ? "noteOnlyHelp"
    : value ? "definition_" + value : "observationChoiceHelp");
  $("note").required = value === "__note";
  $("annotate").disabled = !sessionId || !observationTarget() || annotationPending || !value
    || (value === "__note" && !$("note").value.trim());
}
function renderObservation() {
  const target = observationTarget();
  const event = events.find(e => e.seq === selected);
  const cycle = annotationTargets.find(item => item.target_seq === selected && item.scope === "cycle");
  $("annotation-scope").disabled = !target || !cycle;
  $("annotation-scope").value = annotationScope;
  $("tag").disabled = !target || annotationPending;
  $("note").disabled = !target || annotationPending;
  $("target-label").textContent = target
    ? `${t(target.scope === "cycle" ? "scopeCycle" : "scopeEvent")} · ${target.target_seqs.map(seq => "#" + seq).join(", ")}`
    : t(event ? "notObservationTarget" : "chooseTarget");
  const latest = annotationTargets.filter(item => item.scope === "cycle").at(-1);
  const older = target && latest && target.target_seq < latest.target_seq;
  $("target-age").hidden = !older;
  $("target-age").textContent = older
    ? t("olderTarget", {selected: target.target_seq, latest: latest.target_seq}) : "";
  $("target-preview").replaceChildren();
  $("target-preview").hidden = !target;
  $("target-context").hidden = true;
  if (target) {
    const firstSeq = target.target_seqs[0];
    const human = events.filter(e => e.kind === "utterance" && e.speaker === "human" && e.seq < firstSeq).at(-1);
    if (human) {
      $("target-context").replaceChildren(node("strong", "", t("precedingHuman", {seq: human.seq})), node("p", "", human.text));
      $("target-context").hidden = false;
    }
    for (const seq of target.target_seqs) {
      const item = events.find(e => e.seq === seq);
      const preview = node("div", "target-event");
      preview.append(node("strong", "", `#${seq} · ${item.speaker || item.kind}`),
        node("p", "", item.text || controlText(item)));
      $("target-preview").append(preview);
    }
  }
  $("tag").replaceChildren();
  const placeholder = node("option", "", t("chooseObservation")); placeholder.value = "";
  $("tag").append(placeholder);
  if (target) {
    const open = node("optgroup"); open.label = t("openJudgment");
    for (const [value, key] of [["__unsure", "judgmentUnsure"], ["__note", "judgmentNote"]]) {
      const option = node("option", "", t(key)); option.value = value; open.append(option);
    }
    const labels = node("optgroup"); labels.label = t("descriptiveLabels");
    for (const tag of target.allowed_tags) {
      const option = node("option", "", t("tag_" + tag)); option.value = tag; labels.append(option);
    }
    $("tag").append(open, labels);
  }
  const draft = annotationDrafts.get(draftKey());
  $("tag").value = draft?.tag || "";
  $("note").value = draft?.note || "";
  const active = target && activeObservation(target);
  $("annotate").textContent = annotationPending ? t("savingObservation") : target
    ? t(active ? "reviseTarget" : "saveTarget", {seq: target.target_seq}) : t("annotate");
  $("observation-history").replaceChildren();
  if (target) {
    const history = targetObservations(target);
    for (const entry of history) {
      const replaced = history.some(e => e.supersedes === entry.seq);
      const row = node("div", "saved-observation" + (replaced ? " superseded" : ""));
      row.append(node("strong", "", `#${entry.seq} · ${observationName(entry)}${replaced ? " · " + t("superseded") : ""}`));
      if (entry.note) row.append(node("p", "", entry.note));
      $("observation-history").append(row);
    }
    if (active) $("observation-history").prepend(node("p", "help", t("revisionHelp")));
  }
  for (const el of $("conversation").querySelectorAll("[data-seq]")) {
    el.classList.toggle("selected", !!target?.target_seqs.includes(Number(el.dataset.seq)));
  }
  updateObservationChoice();
}
function selectEvent(seq, scope = "event") {
  rememberDraft();
  selected = seq;
  annotationScope = scope;
  const event = events.find(e => e.seq === seq);
  if (!event) return;
  $("selected-label").textContent = `#${seq} / ${event.kind}`;
  let detail = event;
  if (event.kind === "utterance" && event.call_id) {
    const raw = events.find(e => e.kind === "raw_output" && e.call_id === event.call_id);
    detail = {public_event: event, provider_output: raw || null};
  }
  $("inspect").textContent = JSON.stringify(detail, null, 2);
  renderObservation();
  renderReviewNavigation();
  renderTrace();
}
function clearSelection() {
  selected = null;
  annotationScope = "event";
  $("selected-label").textContent = t("noneSelected");
  $("inspect").textContent = t("inspectHelp");
  renderObservation();
  renderReviewNavigation();
  renderTrace();
}
function ingest(data) {
  if (data.session_id !== sessionId || data.cursor < cursor) return;
  lastSnapshot = data;
  document.querySelector(".workbench").classList.add("has-session");
  status = data.status;
  const additions = (data.events || []).filter(e => !known.has(e.seq));
  for (const e of additions) { known.add(e.seq); events.push(e); }
  events.sort((a, b) => a.seq - b.seq);
  annotationTargets = data.annotation_targets || [];
  for (const e of additions) appendConversation(e);
  cursor = Math.max(cursor, data.cursor || 0);
  const m = data.metrics;
  for (const key of ["assistant", "yields", "forced_stops"]) $("metric-" + key).textContent = String(m[key]);
  $("metric-calls").replaceChildren(document.createTextNode(String(m.calls)), node("span", "", ` / ${m.budget}`));
  $("state").dataset.state = status;
  $("state-label").textContent = t("state_" + status);
  $("session-label").textContent = `RUN ${sessionId.slice(0, 8)} · ${data.settings.policy}`;
  const fixture = data.settings.provider === "fixture";
  const modelLabel = data.settings.model.length > 40 ? data.settings.model.slice(0, 40) + "…" : data.settings.model;
  $("provider-label").textContent = fixture ? t("fixtureActive") : `${data.settings.provider} / ${modelLabel}`;
  $("footer-status").textContent = fixture ? t("fixtureMetrics", m) : t("liveMetrics", {...m, input: m.input_tokens, output: m.output_tokens}) + (data.inflight ? t("upstream") : "");
  $("wake").disabled = busyStates.has(status) || status === "closed";
  $("stop").disabled = status === "closed";
  $("message").disabled = status === "closed";
  $("send").disabled = status === "closed" || sendPending;
  $("export").disabled = false;
  $("presence-bar").hidden = status === "closed";
  const humanTyping = !!data.presence?.human_typing, assistantTyping = !!data.presence?.assistant_composing;
  for (const [id, active, key] of [["human-presence", humanTyping, humanTyping ? "humanTyping" : "humanIdle"],
    ["assistant-presence", assistantTyping, assistantTyping ? "assistantTyping" : "assistantIdle"]]) {
    const label = t(key);
    if ($(id).textContent !== label) $(id).textContent = label;
    $(id).classList.toggle("is-typing", active);
  }
  $("active-typing-mode").textContent = t(data.settings.typing_mode === "presence" ? "modePresence" : "modeInterrupt");
  $("active-typing-mode").title = t(data.settings.typing_mode === "presence" ? "presenceHelp" : "interruptHelp");
  if (additions.length) {
    rememberDraft();
    renderTrace();
    renderReviewNavigation();
    renderObservation();
  }
}
function settings() {
  return {provider: $("provider").value, model: $("provider").value === "fixture" ? "" : $("model").value.trim(),
    compat_base_url: $("provider").value === "openai_compat" ? $("compat-base-url").value.trim() : "",
    conversation_language: $("conversation-language").value,
    anthropic_mode: $("anthropic-mode").value, local_device: $("local-device").value,
    local_runtime: $("local-runtime").value, local_prompt_mode: $("local-prompt-mode").value,
    typing_mode: $("typing-mode").value,
    fixture_case: $("fixture-case").value, structured: $("structured").value === "true",
    proactive: $("proactive").checked, policy: $("policy").value,
    max_utterances: Number($("max-utterances").value), min_gap_ms: Number($("min-gap").value),
    call_budget: Number($("call-budget").value), max_output_tokens: Number($("max-tokens").value),
    context: $("context").value, prompt_extra: $("prompt-extra").value, consent: $("consent").checked};
}
function providerChanged() {
  const provider = $("provider").value;
  const spec = backends.find(b => b.id === provider);
  const fixture = provider === "fixture", local = provider === "local";
  $("model-fields").hidden = fixture;
  $("model").required = !fixture;
  $("fixture-field").hidden = !fixture;
  $("consent-field").hidden = fixture;
  $("consent").required = !fixture;
  $("compat-url-field").hidden = provider !== "openai_compat";
  $("compat-base-url").required = provider === "openai_compat";
  $("local-device-field").hidden = !local;
  $("local-options").hidden = !local;
  $("local-device").disabled = $("local-runtime").value === "mlx" || localInspection?.runtime === "mlx";
  $("typing-mode-help").textContent = t($("typing-mode").value === "presence" ? "presenceHelp" : "interruptHelp");
  $("anthropic-mode-field").hidden = provider !== "anthropic";
  $("model-label").textContent = t(local ? "modelDirectory" : "modelId");
  $("model").placeholder = t(local ? "localPlaceholder" : "modelPlaceholder");
  $("provider-help").textContent = t("help_" + provider);
  $("structured").disabled = spec ? !spec.structured_supported : false;
  if (local) $("structured").value = "false";
  if (fixture) $("key-status").textContent = "";
  else if (local) $("key-status").textContent = t("localReady");
  else if (provider === "ollama") $("key-status").textContent = t("ollamaReady");
  else if (provider === "openai_compat") $("key-status").textContent = t("customCredentials");
  else $("key-status").textContent = t(providerReady[provider] ? "credentialsReady" : "credentialsMissing", {env: spec?.key_env || ""});
  const mode = $("structured").value !== "true" ? "prompt_only"
    : provider === "anthropic" && $("anthropic-mode").value === "tool" ? "anthropic_forced_tool" : spec?.schema_mode;
  $("constraint-help").textContent = t(local ? "localConstraint" : fixture ? "fixtureConstraint" : "constraint", {mode});
  if (localInspection) renderLocalInspection();
}
function renderLocalInspection() {
  const info = localInspection;
  $("local-check").textContent = t("localChecked", {
    runtime: info.runtime, prompt: t(info.prompt_mode === "native" ? "promptNative" : "promptPlain"),
    context: info.context_window ?? "?", library: info.library,
    version: info.library_version || t("notInstalled")});
}
function resetLocalInspection() {
  localInspection = null;
  $("local-check").textContent = "";
  providerChanged();
}
async function checkLocal() {
  const input = {model: $("model").value.trim(), local_runtime: $("local-runtime").value,
    local_prompt_mode: $("local-prompt-mode").value};
  $("check-local").disabled = true;
  try {
    const info = await request("/api/local-model", input);
    if (input.model !== $("model").value.trim() || input.local_runtime !== $("local-runtime").value
      || input.local_prompt_mode !== $("local-prompt-mode").value) return;
    localInspection = info;
    providerChanged();
  } catch (error) { $("local-check").textContent = error.message; }
  finally { $("check-local").disabled = false; }
}
function localizeLiveView() {
  rememberDraft();
  providerChanged();
  for (const option of $("provider").options) if (option.value === "fixture") option.textContent = t("fixtureLabel");
  if (sessionId) {
    $("conversation").replaceChildren();
    for (const event of events) appendConversation(event);
    $("start").replaceChildren(document.createTextNode(t("newExperiment")), node("span", "", "↗"));
    if (lastSnapshot) ingest(lastSnapshot);
    if (selected) selectEvent(selected, annotationScope);
  }
  renderReviewNavigation();
  renderObservation();
  renderTrace();
}
$("ui-language").addEventListener("change", () => {
  window.I18N.setLanguage($("ui-language").value);
  localizeLiveView();
});
$("structured").addEventListener("change", providerChanged);
$("anthropic-mode").addEventListener("change", providerChanged);
$("provider").addEventListener("change", providerChanged);
$("typing-mode").addEventListener("change", providerChanged);
$("local-runtime").addEventListener("change", resetLocalInspection);
$("local-prompt-mode").addEventListener("change", resetLocalInspection);
$("model").addEventListener("input", resetLocalInspection);
$("check-local").addEventListener("click", checkLocal);
$("max-utterances").addEventListener("input", () => { $("max-value").value = $("max-utterances").value; });
$("min-gap").addEventListener("input", () => { $("gap-value").value = $("min-gap").value + " ms"; });
$("trace-filter").addEventListener("change", renderTrace);
$("settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (creating) return;
  const config = settings();
  creating = true; $("start").disabled = true;
  clearTimeout(typingTimer); typingActive = false; typingLastSent = 0;
  try {
    // Drain the old poll before closing/pruning that session; pause new polls.
    if (pollPending) await pollPending.catch(() => {});
    await enqueue(async () => {
      if (sessionId) {
        try { await request("/api/close", {session: sessionId}); } catch { /* stale prior server/session */ }
      }
      const data = await request("/api/session", config);
      sessionId = data.session_id; cursor = 0; selected = null; events = []; known = new Set();
      annotationScope = "event"; annotationTargets = []; annotationDrafts.clear();
      clearTimeout(typingTimer); typingActive = false; typingLastSent = 0;
      $("conversation").replaceChildren();
      $("inspect").textContent = t("inspectHelp");
      $("selected-label").textContent = t("noneSelected");
      storage.set("turnless-session", sessionId);
      ingest(data);
      $("start").replaceChildren(document.createTextNode(t("newExperiment")), node("span", "", "↗"));
      toast(config.provider === "fixture" ? t("fixtureStarted") : t("modelStarted"));
    });
  } catch (error) { toast(error.message, true); }
  finally { creating = false; $("start").disabled = false; }
});
function stopTyping() {
  clearTimeout(typingTimer);
  if (!typingActive) return;
  typingActive = false;
  if (sessionId && status !== "closed") mutate("/api/typing", {active: false}).catch(() => {});
}
function signalTyping() {
  if (!sessionId || creating || status === "closed") return;
  if (!$("message").value) { stopTyping(); return; }
  clearTimeout(typingTimer);
  // Activity refreshes presence at most once per second; no draft ever leaves this field.
  if (!typingActive || Date.now() - typingLastSent >= 1000) {
    typingActive = true; typingLastSent = Date.now();
    mutate("/api/typing", {active: true}).catch(error => { typingActive = false; toast(error.message, true); });
  }
  typingTimer = setTimeout(stopTyping, 2000);
}
$("message").addEventListener("input", signalTyping);
$("message").addEventListener("compositionupdate", signalTyping);
$("message").addEventListener("blur", stopTyping);
document.addEventListener("visibilitychange", () => { if (document.hidden) stopTyping(); });
window.addEventListener("pagehide", stopTyping);
$("message").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault(); $("composer").requestSubmit();
  }
});
$("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("message").value;
  if (!text.trim() || !sessionId || sendPending) return;
  const autoReply = $("auto-reply").checked;
  sendPending = true; $("send").disabled = true; $("message").value = "";
  clearTimeout(typingTimer); typingActive = false;
  try { await mutate("/api/human", {text, auto_reply: autoReply}); }
  catch (error) { if (!$("message").value) $("message").value = text; toast(error.message, true); }
  finally { sendPending = false; $("send").disabled = status === "closed"; $("message").focus(); }
});
$("stop").addEventListener("click", () => mutate("/api/interrupt", {reason: "operator_stop"}).then(() => toast(t("stopped"))).catch(error => toast(error.message, true)));
$("wake").addEventListener("click", () => { mutate("/api/wake").catch(error => toast(error.message, true)); });
$("annotation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const target = observationTarget(), value = $("tag").value;
  if (!target || !value || annotationPending) return;
  const sid = sessionId, key = draftKey(), previous = activeObservation(target);
  const judgment = value === "__unsure" ? "unsure" : value === "__note" ? "note" : "labeled";
  const payload = {target_seq: target.target_seq, scope: target.scope, judgment,
    tag: judgment === "labeled" ? value : null, note: $("note").value, supersedes: previous?.seq ?? null};
  if (judgment === "note" && !payload.note.trim()) return;
  rememberDraft();
  annotationPending = true;
  renderObservation();
  try {
    await mutate("/api/annotate", payload);
    if (sid === sessionId) {
      annotationDrafts.delete(key);
      // A subsequent observation must explicitly choose its target. Do not clear
      // another target selected while this save was in flight.
      if (draftKey() === key) clearSelection();
      toast(t("annotatedTarget", {seq: target.target_seq}));
    }
  } catch (error) { toast(error.message, true); }
  finally {
    annotationPending = false;
    renderObservation();
    renderReviewNavigation();
  }
});
$("tag").addEventListener("change", () => { rememberDraft(); updateObservationChoice(); });
$("note").addEventListener("input", () => { rememberDraft(); updateObservationChoice(); });
$("annotation-scope").addEventListener("change", () => selectEvent(selected, $("annotation-scope").value));
$("review-sequence").addEventListener("change", () => {
  if ($("review-sequence").value) selectEvent(Number($("review-sequence").value), "cycle");
});
$("next-unreviewed").addEventListener("click", () => {
  const next = annotationTargets.find(target => target.scope === "cycle" && !activeObservation(target));
  if (next) selectEvent(next.target_seq, "cycle");
});
$("latest-sequence").addEventListener("click", () => {
  const latest = annotationTargets.filter(target => target.scope === "cycle").at(-1);
  if (latest) selectEvent(latest.target_seq, "cycle");
});
$("export").addEventListener("click", async () => {
  if (!sessionId) return;
  try {
    const response = await fetch(`/api/export?session=${sessionId}`, {headers: {"X-Turnless-Token": token}});
    if (!response.ok) throw new Error(t("exportFailed"));
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a"); link.href = url; link.download = `turnless-${sessionId.slice(0,8)}.jsonl`;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 5000);
    toast(t("exported"));
  } catch (error) { toast(error.message, true); }
});
async function poll() {
  const sid = sessionId;
  if (sid && !creating) {
    try {
      pollPending = request(`/api/events?session=${sid}&after=${cursor}`);
      const data = await pollPending;
      if (sid === sessionId) ingest(data);
      pollErrors = 0;
    } catch (error) {
      pollErrors++;
      if (pollErrors === 3) toast(t("reconnecting"), true);
    } finally { pollPending = null; }
  }
  setTimeout(poll, pollErrors ? 1500 : 250);
}
async function boot() {
  try {
    const data = await request("/api/bootstrap");
    token = data.token;
    providerReady = data.providers || {};
    backends = data.backends || [];
    $("provider").replaceChildren(...backends.map(spec => {
      const option = node("option", "", spec.id === "fixture" ? t("fixtureLabel") : spec.label);
      option.value = spec.id; return option;
    }));
    openaiReady = !!providerReady.openai;
    compatHint = data.compat_base_url || "";
    if (compatHint && !$("compat-base-url").value) $("compat-base-url").value = compatHint;
    renderObservation();
    renderReviewNavigation();
    providerChanged();
    const stored = storage.get("turnless-session");
    if (stored) {
      try {
        const existing = await request(`/api/events?session=${stored}`);
        sessionId = stored; ingest(existing);
        const config = existing.settings;
        const fields = {provider: "provider", model: "model", structured: "structured", policy: "policy",
          max_utterances: "max-utterances", min_gap_ms: "min-gap", call_budget: "call-budget",
          max_output_tokens: "max-tokens", context: "context", prompt_extra: "prompt-extra",
          compat_base_url: "compat-base-url", conversation_language: "conversation-language",
          anthropic_mode: "anthropic-mode", local_device: "local-device", fixture_case: "fixture-case",
          local_runtime: "local-runtime", local_prompt_mode: "local-prompt-mode", typing_mode: "typing-mode"};
        for (const [key, id] of Object.entries(fields)) if (config[key] !== undefined) $(id).value = String(config[key]);
        $("proactive").checked = config.proactive;
        $("consent").checked = config.consent;
        $("max-value").value = config.max_utterances;
        $("gap-value").value = config.min_gap_ms + " ms";
        localizeLiveView();
      } catch { storage.remove("turnless-session"); }
    }
    poll();
  } catch (error) { toast(t("bootError", {error: error.message}), true); $("start").disabled = true; }
}
boot();
