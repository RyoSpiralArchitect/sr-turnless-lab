# SR Turnless Lab ∴

**発話の終わりと、会話を渡す瞬間を分ける。**

[English](README.md) · [実験ガイド](experiments.md) · [バックエンド契約](docs/backends.md)

会話の境界を観測する小さなローカル実験系です。モデルは1回に1つ、公開発話 `emit` または待機 `yield` を返します。発話ごとに公開済みの履歴を渡して判断し直します。人間の割り込みや、人間からの連続入力も扱えます。

これは **application-levelのプロトコル**です。重み、tokenizer、native chat template、EOSの意味は変更しません。`emit` / `yield` はJSONの値であり特殊トークンではありません。ツール実行、バックグラウンド思考、タイマーによる自動再開、FT用データへの自動変換はありません。

## 最初の3分

Python 3.10以上。固定デモとHTTPバックエンドは標準ライブラリだけで動きます。デモにはビルド、npm、APIキーは不要です。

```sh
git clone https://github.com/RyoSpiralArchitect/sr-turnless-lab.git
cd sr-turnless-lab
python3 -m turnless
```

**http://127.0.0.1:8765** を開きます。Windowsでは `py -3 -m turnless` または `run.bat`、macOS/Linuxでは `./run.sh` も使えます。

1. 固定デモで実験を開始すると、3連続発話の後にyieldします。
2. 「再開する」の後、入力を始めると未公開応答が無効になります。入力本文は送信するまでモデルに送りません。
3. 「送信後にモデルを起動」をOFFにすると、人間の発話を連続して記録できます。
4. 発話やイベントを選択し、観測ラベル・ノートを追記してJSONLを書き出します。

**固定デモは脚本であり、LLMの挙動の実測ではありません。** 反復、形式違反、人間の役割の捏造、上限停止も、意図的に注入した検証用の失敗です。

初期表示は英語です。ヘッダーから日本語に切り替えると、そのブラウザに選択を保存します。**会話言語は独立した実験条件**で、「文脈・プロンプト・予算」から設定します。表示言語を変えてもプロンプト、記録済みの発話、進行中の実験条件は変わりません。

```sh
python3 -m turnless --port 8766 --log-dir ./runs
python3 -m unittest discover -s tests -v
python3 -m turnless.analyze "examples/*.jsonl"
```

パッケージとして使う場合は `python3 -m pip install .`。`sr-turnless-lab` と `turnless-lab` の両方で起動できます。

## バックエンドの一元化

共通の設定、リクエスト記録、結果契約、形式検証、割り込み処理、観測UIを使います。通信方式や形式制約の違いは、実験条件として明示的に残します。

| 接続先 | 通信方式 | 形式制約 | サーバーの認証設定 |
| --- | --- | --- | --- |
| OpenAI | Responses | `text.format` | `OPENAI_API_KEY` |
| Anthropic Claude | Messages | 標準は `output_config.format`。強制ツール方式を明示選択可能 | `ANTHROPIC_API_KEY` |
| Gemini | 固定のChat Completions互換経路 | `response_format` | `GEMINI_API_KEY` |
| Mistral | 固定のChat Completions経路 | `response_format` | `MISTRAL_API_KEY` |
| 汎用互換サーバー | 指定したBase URLのChat Completions | 対応していれば `response_format` | `OPENAI_COMPAT_API_KEY`。ローカルでは不要な場合あり |
| Ollama | loopbackの `/api/chat` | `format` | 不要 |
| ローカルHF | チェックポイントのディレクトリとnative chat template | Prompt only | 不要 |

利用可能な**正確なモデルID**、またはローカルのディレクトリを指定します。Schema対応はモデルと接続先によります。未対応の要求はエラーとして残し、別方式への自動切替や再試行は行いません。GeminiとMistralは専用の接続先・認証設定を使います。汎用互換のBase URLはHTTPSまたはloopback HTTPに限定します。

キーはサーバー内に保持し、ブラウザ、実験設定、リクエスト記録には含めません。安全な認証管理の仕組みを利用してください。CodexではOpenAI Platformの安全なキー設定フローが使えます。既存のサーバー環境では表の環境変数名を読み取ります。ソース外のキーファイルは `--key-file`、`--anthropic-key-file`、`--gemini-key-file`、`--mistral-key-file`、`--compat-key-file` で指定できます。変更後はサーバーを再起動してください。

互換経路の既定URLは `--compat-base-url` で設定できます。Ollamaは `http://127.0.0.1:11434` が既定で、`OLLAMA_HOST` により別のloopback HTTP接続先を指定できます。HTTPリダイレクトは拒否します。汎用互換経路のキーは1つなので、送信先に合う認証情報を用いてください。

### HF_HUB_OFFLINE=1 とローカルモデル

ローカル推論を行う端末だけに追加依存を導入します。

```sh
python3 -m pip install '.[local]'
HF_HUB_OFFLINE=1 python3 -m turnless
```

「Hugging Face · offline directory」を選び、既存のチェックポイントのディレクトリと、CPU / MPS / CUDAを指定します。PowerShellでは起動前に `$env:HF_HUB_OFFLINE = "1"` を設定します。

`local_files_only=True`、`trust_remote_code=False` で読み込み、HubモデルIDからのダウンロードやtemplateの自動追加はしません。モデル側に対応したchat templateが必要です。未対応の役割・templateは明示的なエラーになります。template適用後のtokenizeには `add_special_tokens=False` を使い、特殊トークンの重複を防ぎます。生成はgreedy、1 beam、1 sequenceです。デバイスの自動変更はしません。

モデルの読込と生成はセッション間でも直列化します。同じ解決済みディレクトリとデバイスのモデルはプロセス内で再利用します。チェックポイントを上書きした場合やメモリを解放する場合はサーバーを再起動してください。キャンセルは読込・生成の前とロック待機中に確認します。実行中の生成は中断せず、古い結果を公開前に破棄します。EOS終了、出力上限、その他の停止を区別して記録します。greedyでも、異なるデバイスやライブラリ版の間でビット単位の再現性を保証するものではありません。

### 出力形式

**Prompt only** はプロンプトだけでJSONを依頼します。形式が壊れたら生出力を保存して停止します。Markdownフェンスの除去、自動修復、再質問はしません。

**JSON Schema** はバックエンドごとの制約を指定します。Claudeはnative JSON出力が標準で、強制ツール方式は別の実験条件です。強制ツールでは正しい名前の決定が厳密に1つ、追加の公開テキストなしの場合のみ採用します。ツールは実行しません。複数choice、複数の公開決定、予期しないツール、拒否、不明な終了理由、未完了の出力は、公開発話やmodel yieldになりません。

## プロトコルと割り込み

```json
{"action":"emit","text":"ひとつ、置いておく。","delay_ms":0}
```

```json
{"action":"yield","text":"","delay_ms":0}
```

1回の推論で1つの決定を返します。`emit` は発話を公開し、その履歴で次の判断を要求します。`yield` は人間入力または明示的な再開を待ちます。実験開始は外部トリガーであり、モデルが自律的に起きたことを示しません。

`delay_ms` は**推論結果を受け取った後**の待機で、0〜5000ms。2発話目以降は提案値と最低発話間隔の大きい方を適用し、両方を記録します。

入力開始、送信、停止で現在のepochを無効にします。遅延応答は破棄として保存し、公開しません。公開済みの発話は履歴に残ります。1セッション1ワーカーで、待機中の再開要求は最新の履歴にまとめます。

**上流の推論中断や課金停止は保証しません。** 同期HTTPが返るかタイムアウトするまで次の推論を待つ場合があります。HTTP I/Oは60秒、サイクル予算は120秒ですが、後者は推論間での確認です。既定はサイクルごとに5発話、セッション全体で40呼び出し。yield判断も1回を消費します。

## 記録と評価

- `session_open`：固定された条件、制約方式、ソースのmanifest/hash、package・protocol・adapterの版、Python版、取得可能ならこのリポジトリのGit commitと変更状態。
- `request`：instructions、公開履歴、プロンプトhash、キーを除いた実リクエストbodyまたはローカル生成条件。
- `raw_output`：公開候補テキスト、公開応答ブロック、正規化した状態、上流の終了理由、モデル・request ID、返されたtoken使用量と遅延。ローカルではtemplate・適用後prompt・入力token列のhashと生成設定も記録。
- `utterance` / `yield` / `forced_stop` / `discard`：別イベント。上限停止をモデルの沈黙判断にしません。
- `flag`：自動候補であり正解ではありません。`annotation`：モデル入力に混ぜない観測者のラベルとノートです。

私的なthinking/reasoningブロックは観測用に取得せず、保存しません。ツールのinputはparse済みの公開payloadで、HTTPの生文字列ではありません。通信方式を記録で識別します。ソースhashは実験系の識別であり、ホストされたモデルの重みを特定しません。ローカルのtemplate・設定記録も重み全体のfingerprintではありません。

会話・文脈・プロンプトはローカルのJSONLに追記保存します。`runs/` と `reviews/` はGitと配布パッケージから除外します。共有前に内容を確認してください。export時に上流処理が残っているかは `export_metadata.upstream_pending` に記録します。タブを閉じてもサーバーのサイクルは続くため、停止操作またはサーバー終了が必要です。

公開発話は全件、制御イベントは直近24件をモデルに渡します。instructionsとinputが80,000文字を超えると、無断要約せず停止します。文字数とtoken数は別です。

サーバーはloopbackのみで待ち受け、Host・API tokenを検証し、CSPを設定し、発話をテキストとして描画します。単一利用者向けの研究プロトタイプです。外部公開には別途デプロイ・セキュリティ設計が必要です。

## 検証範囲と研究段階

検証環境と制限は [validation](docs/validation.md) に記載します。通常のテストは合成応答を使うオフライン検証です。追加のCPU結合テストは、小さな検証用チェックポイントをローカルで作り、実際のTransformers読込・template適用・EOS検出を確認します。ダウンロードは行いません。

```sh
TURNLESS_TEST_LOCAL=1 HF_HUB_OFFLINE=1 python3 -m unittest discover -s tests -p test_offline_checkpoint.py -v
```

これらはモデルの会話品質やホストされたAPIの信頼性を示しません。`examples/` は元のv0.1固定デモとして保持しています。公開前の個人的な探索ログは公開リポジトリに含めず、ベンチマークとも扱いません。次の研究段階は、具体的な場面と観測者判断によって「良い継続」「良い沈黙」「判断が割れる例」を集めることです。

## 構成とライセンス

`catalog.py` がバックエンド定義、`providers.py` が共通factoryとアダプター、`protocol.py` が形式・設定・プロンプト、`engine.py` が実行管理、`provenance.py` が版の記録、`server.py` がローカルAPI、`web/` が英日UI、`analyze.py` が集計です。

[Apache License 2.0](LICENSE)。Copyright 2026 Ryo ∴ SpiralArchitect。モデル重み、データセット、外部サービスにはそれぞれのライセンス・利用条件が適用されます。
