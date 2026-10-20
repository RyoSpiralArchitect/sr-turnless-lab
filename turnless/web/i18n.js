"use strict";
(() => {
  const messages = {
    en: {
      title: "SR Turnless Lab · A laboratory for conversational boundaries",
      tagline: "An utterance can end before the conversation is handed over.",
      uiLanguage: "Interface language", notConnected: "Scripted demo / no model connected",
      export: "Export JSONL", experimentSettings: "Experiment settings", backend: "Model backend",
      fixtureLabel: "Scripted demo (no key)", modelId: "Model ID", modelDirectory: "Checkpoint directory",
      modelPlaceholder: "Exact model ID", localPlaceholder: "/absolute/path/to/checkpoint",
      baseUrl: "Endpoint base URL", device: "Local device", demoCase: "Demo case",
      case_normal: "Consecutive utterances → yield", case_redundant: "Repeat the same utterance",
      case_yield: "Yield immediately", case_malformed: "Malformed JSON", case_fake: "Fabricated human role",
      case_overtalk: "Continue until the cap", proactive: "Start before the human",
      proactiveHelp: "Wake once at session_open", outputFormat: "Output format",
      promptOnly: "Prompt only / observe format errors", schema: "JSON Schema / constrain the format",
      anthropicMode: "Claude constraint method", nativeSchema: "Native JSON Schema",
      forcedTool: "Forced tool (separate condition)", policy: "Speaking policy", neutral: "Neutral",
      sparse: "Sparse / leave room", exploratory: "Exploratory / allow development",
      utteranceCap: "Consecutive utterance cap", minGap: "Minimum gap",
      gapHelp: "Cap stops are separate from model yield. The gap is added after inference.",
      advanced: "Context, prompt & budget", conversationLanguage: "Conversation language",
      languageHelp: "An experiment condition. Changing the interface language does not change it.",
      context: "Shared context", extraPrompt: "Additional experiment instruction",
      extraPlaceholder: "After asking a question, you may wait for a reply.", callBudget: "Call budget",
      maxTokens: "Output tokens / call", budgetHelp: "The call budget covers the whole experiment. Each utterance and each yield decision requires inference.",
      consent: "Send the conversation and context to the selected model. Hosted APIs may incur charges.",
      start: "Start experiment", newExperiment: "Start new experiment",
      settingsHelp: "Settings apply to new experiments. Traces are saved locally in runs/.",
      conversation: "Conversation", notStarted: "Not started", state_waiting: "Waiting",
      state_queued: "Queued", state_generating: "Generating", state_delaying: "Waiting to speak",
      state_closed: "Closed", state_log_error: "Trace write error", utterances: "Utterances",
      forcedStops: "Forced stops", calls: "Calls", publicUtterances: "Public utterances",
      emptyTitle: "It is nobody's turn yet.", emptyBody: "Either person can begin. Make room for a different rhythm.",
      typingInterrupt: "Interrupt when typing", wake: "Resume", stop: "■ Stop", humanMessage: "Human message",
      messagePlaceholder: "You can speak in the middle, too.", send: "Send message",
      autoReply: "Wake model after sending", keyboardHelp: "Enter to send / Shift + Enter for a new line",
      observer: "Trace and observations", observe: "Observe the boundaries",
      observerIntro: "Keep successes and failures. Public utterances and raw outputs are separate events.",
      traceFilter: "Event filter", controlEvents: "Control & utterances", allEvents: "All / including raw",
      traceEmpty: "Events will appear here.", selectedEvent: "Selected event", noneSelected: "None",
      inspectHelp: "Select an event to inspect its contents.", annotationLabel: "Observation label",
      observationNote: "Observation note", notePlaceholder: "What happened? What would you change?",
      annotate: "Save observation", observerHelp: "Automatic flags are candidates. Useful continuation and good silence still need human judgment.",
      footer: "The model's native chat template is unchanged.",
      help_fixture: "A fixed script for testing the harness. This is not an LLM.",
      help_openai: "Responses API. Credentials remain on the server.",
      help_anthropic: "Claude Messages API. Native JSON Schema or an explicitly selected forced-tool condition.",
      help_gemini: "Gemini through its Chat Completions compatibility endpoint. Uses separate Gemini credentials.",
      help_mistral: "Mistral through Chat Completions. Uses separate Mistral credentials.",
      help_openai_compat: "Custom Chat Completions server, including vLLM and LM Studio. Capabilities depend on the endpoint and model.",
      help_ollama: "A local Ollama server. The model must already be installed.",
      help_local: "A local Transformers checkpoint with a chat template. Offline-only loading; no automatic downloads.",
      credentialsReady: "Server credentials found. API connectivity has not been tested.",
      credentialsMissing: "Server credentials are missing ({env}). Configure them securely before starting.",
      customCredentials: "Uses only the custom endpoint credential. A local server may not require authentication.",
      localReady: "Install the optional local dependencies. The selected device must be available; no device fallback.",
      ollamaReady: "Use the name of a model already installed in your local Ollama server.",
      constraint: "Recorded constraint method: {mode}. Model support is required.",
      localConstraint: "Local HF uses Prompt only. JSON Schema is unavailable.",
      fixtureConstraint: "Format selection does not change the scripted demo.",
      selectUtterance: "Inspect this utterance", yieldEvent: "model yield · waiting after {n} utterances",
      typingEvent: "human started typing", providerError: "provider error · inspect the raw event",
      protocolError: "protocol error · malformed output recorded", fixtureActive: "Scripted demo / not an LLM",
      fixtureMetrics: "Scripted fixture · interrupts {interrupts} · discarded {discards} · observations {annotations}",
      liveMetrics: "Input {input} / output {output} tokens · interrupts {interrupts} · observations {annotations}",
      upstream: " · upstream request pending", startFirst: "Start an experiment first.",
      unreadableResponse: "Cannot read the server response.", fixtureStarted: "Scripted demo started. These outputs are not LLM measurements.",
      modelStarted: "Model experiment started.", stopped: "Publication stopped. Upstream computation or billing may continue.",
      annotated: "Observation saved. It is not included in model input.", exportFailed: "Export failed.",
      exported: "JSONL exported. It includes conversation and prompts; review before sharing.",
      reconnecting: "Check the server connection. Retrying…", bootError: "Could not start: {error}"
    },
    ja: {
      title: "SR Turnless Lab · 発話の実験室", tagline: "発話の終わりと、会話を渡す瞬間を分ける。",
      uiLanguage: "表示言語", notConnected: "固定デモ / モデル未接続", export: "JSONLを書き出す",
      experimentSettings: "実験の条件", backend: "モデル接続", fixtureLabel: "固定デモ（キー不要）",
      modelId: "モデルID", modelDirectory: "チェックポイントのディレクトリ", modelPlaceholder: "正確なモデルID",
      localPlaceholder: "/absolute/path/to/checkpoint", baseUrl: "エンドポイントのBase URL", device: "ローカルの実行デバイス",
      demoCase: "デモのケース", case_normal: "連続発話 → yield", case_redundant: "同じことを繰り返す",
      case_yield: "すぐにyield", case_malformed: "JSONを壊す", case_fake: "人間の役割を捏造する",
      case_overtalk: "上限まで話し続ける", proactive: "人間より先に始める", proactiveHelp: "session_openで一度だけ起動",
      outputFormat: "出力形式", promptOnly: "Prompt only / 形式違反も観測", schema: "JSON Schema / 形式を制約",
      anthropicMode: "Claudeの形式制約", nativeSchema: "ネイティブJSON Schema", forcedTool: "強制ツール（別の実験条件）",
      policy: "発話の方針", neutral: "Neutral / 中立", sparse: "Sparse / 余白を残す", exploratory: "Exploratory / 展開を許す",
      utteranceCap: "連続発話の上限", minGap: "最低発話間隔", gapHelp: "上限停止はmodel yieldと分けて記録。間隔は推論後に加算されます。",
      advanced: "文脈・プロンプト・予算", conversationLanguage: "会話の言語",
      languageHelp: "実験条件の一つです。表示言語を切り替えても、この設定は変わりません。",
      context: "共有する文脈", extraPrompt: "追加の実験指示", extraPlaceholder: "例：問いを置いたあとは、返事を待ってよい。",
      callBudget: "呼び出し上限", maxTokens: "出力token / 回", budgetHelp: "呼び出し上限は実験全体に適用。1発話ごとに再推論し、yieldの判断にも呼び出しを使います。",
      consent: "会話と文脈を選択したモデルに送信する。外部APIでは利用料金が発生する場合があります。",
      start: "実験を開始", newExperiment: "新しい実験を開始", settingsHelp: "設定は新しい実験から適用。ログはローカルのruns/に保存されます。",
      conversation: "会話の流れ", notStarted: "まだ開始していません", state_waiting: "待機", state_queued: "呼び出し待ち",
      state_generating: "生成中", state_delaying: "発話まで待つ", state_closed: "終了", state_log_error: "ログ保存エラー",
      utterances: "発話", forcedStops: "上限・強制停止", calls: "呼び出し", publicUtterances: "公開された発話",
      emptyTitle: "まだ、誰の番でもない。", emptyBody: "人間が先でも、モデルが先でも。違う会話のリズムを試してみる。",
      typingInterrupt: "入力開始で割り込む", wake: "再開する", stop: "■ 止める", humanMessage: "人間の発話",
      messagePlaceholder: "途中で話しかけてもいい。", send: "発話を送信", autoReply: "送信後にモデルを起動",
      keyboardHelp: "Enterで送信 / Shift + Enterで改行", observer: "トレースと観測", observe: "境界を観測する",
      observerIntro: "成功も失敗も残す。表示された発話と生出力は別のイベント。", traceFilter: "イベント表示",
      controlEvents: "制御と発話", allEvents: "すべて / rawを含む", traceEmpty: "ここにイベントが積み重なります。",
      selectedEvent: "選択したイベント", noneSelected: "未選択", inspectHelp: "イベントを選ぶと、内容を確認できます。",
      annotationLabel: "観測ラベル", observationNote: "観測ノート", notePlaceholder: "何が起きた？ 次は何を変える？",
      annotate: "観測を残す", observerHelp: "自動フラグは候補だけ。良い継続や良い沈黙は、人間が判断します。",
      footer: "モデルのnative chat templateは変更しません。", help_fixture: "動作検証用の固定脚本です。LLMの挙動ではありません。",
      help_openai: "Responses APIに接続。キーはサーバー内だけで扱います。",
      help_anthropic: "Claude Messages API。ネイティブJSON Schemaと強制ツールは別の実験条件です。",
      help_gemini: "GeminiのChat Completions互換経路。専用のGemini認証情報を使用します。",
      help_mistral: "MistralのChat Completions経路。専用のMistral認証情報を使用します。",
      help_openai_compat: "vLLM・LM StudioなどのChat Completions互換サーバー。対応機能は接続先とモデルによります。",
      help_ollama: "ローカルOllamaに接続。モデルは事前に用意してください。",
      help_local: "chat template付きのローカルTransformersモデル。オフラインで読み込み、自動ダウンロードはしません。",
      credentialsReady: "サーバーの認証情報を確認済み。API到達性は未確認です。",
      credentialsMissing: "サーバーの認証情報が未設定です（{env}）。開始前に安全に設定してください。",
      customCredentials: "互換エンドポイント専用の認証情報だけを使用します。ローカルサーバーでは認証が不要な場合があります。",
      localReady: "ローカル推論用の追加依存が必要です。指定したデバイスを使い、別デバイスへ自動変更しません。",
      ollamaReady: "ローカルOllamaに用意したモデル名を指定してください。",
      constraint: "記録される制約方式：{mode}。モデル側の対応が必要です。", localConstraint: "ローカルHFはPrompt only固定です。",
      fixtureConstraint: "形式の選択は固定デモの出力を変えません。", selectUtterance: "この発話を観測する",
      yieldEvent: "model yield · {n}発話で待機", typingEvent: "人間が入力を開始", providerError: "provider error · 生出力を確認してください",
      protocolError: "protocol error · 形式違反を記録しました", fixtureActive: "固定デモ / LLMではありません",
      fixtureMetrics: "固定脚本の検証 · 割り込み {interrupts} · 破棄 {discards} · 観測 {annotations}",
      liveMetrics: "入力 {input} / 出力 {output} tokens · 割り込み {interrupts} · 観測 {annotations}",
      upstream: " · 上流処理中", startFirst: "まず実験を開始してください。", unreadableResponse: "サーバー応答を読み取れませんでした。",
      fixtureStarted: "固定デモを開始。出力は脚本であり、LLMの実測ではありません。", modelStarted: "実モデルとの実験を開始しました。",
      stopped: "発話の公開を停止。上流の推論や課金が続く場合があります。", annotated: "観測を追記しました。モデルの入力には含めません。",
      exportFailed: "書き出しに失敗しました。", exported: "JSONLを書き出しました。会話とプロンプトを含むため、共有前に確認してください。",
      reconnecting: "サーバーとの接続を確認してください。再接続を試みています。", bootError: "起動できませんでした：{error}"
    }
  };
  let language = "en";
  try { if (localStorage.getItem("turnless-ui-language") === "ja") language = "ja"; } catch { /* optional storage */ }
  function t(key, values = {}) {
    const value = messages[language][key] ?? messages.en[key] ?? key;
    return value.replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? `{${name}}`));
  }
  function apply() {
    document.documentElement.lang = language;
    document.title = t("title");
    document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll("[data-i18n-placeholder]").forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    document.querySelectorAll("[data-i18n-aria]").forEach(el => { el.setAttribute("aria-label", t(el.dataset.i18nAria)); });
    document.getElementById("ui-language").value = language;
  }
  window.I18N = {t, apply, messages, get language() { return language; }, setLanguage(value) {
    language = value === "ja" ? "ja" : "en";
    try { localStorage.setItem("turnless-ui-language", language); } catch { /* optional storage */ }
    apply();
  }};
  apply();
})();
