# BracketCloser-LLM

最初にやること
docker pull sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1

### ステップ 2: Docker コンテナのセットアップと実行

次に、プロジェクトディレクトリをマウントし、コンテナを実行します。

```sh
export PROJECT_DIR=$(pwd)
docker run -it \
  -v "${PROJECT_DIR}:/app/project" \
  --workdir /app/project \
  --tmpfs /tmp:rw,size=10g \
  bracket-closer-image \
  bash
```

### ステップ 3: データセットの作成

モデルの学習に必要なデータセットを生成します。以下のスクリプトを実行してください。生成するサンプル数は、`num_samples`の値を変更することで調整可能です。学習時間を考慮すると、300〜500サンプルが適当です。

```sh
python dataset.py
```

### ステップ 4: モデルの学習

以下のコマンドを実行してモデルを学習します。

```sh
python train.py
```

実行後、学習したいモデルを選択する画面が表示されます。次の5つのモデルから選択できます。入力の際には、以下の形式で指定してください（例: "gru 1min"）。
省略しておりますが、全てのモデルで、1min~2daysまで選択できます。

- **GRU (Gated Recurrent Unit)**
  - 1min: `gru 1min`
  - 10min: `gru 10min`
  - 1hour: `gru 1hour`
  - 6hours: `gru 6hours`
- **Transformer**
  - 1min: `tra 1min`
  - 10min: `tra 10min`
  - 1hour: `tra 1hour`
- **LSTM**
  - 1min: `lstm 1min`
  - 10min: `lstm 10min`
  - 1hour: `lstm 1hour`
- **BERT**
  - 1min: `ber 1min`
  - 10min: `ber 10min`
- **GPT**
  - 1min: `gpt 1min`

学習が完了すると、モデルは `models` フォルダに保存されます。例えば、`gru_20240605_163409_15m`というフォルダ名は、GRUモデルが2024年6月5日16時34分9秒に学習され、15分かかったことを示しています。

### ステップ 5: モデルの評価

以下のコマンドでモデルを評価します。

```sh
python evaluate.py
```

コマンド実行後、最新順に並んだモデルのリストが表示されます。評価したいモデルを番号で選択します。各評価は100個のテストデータを生成し、正解率がパーセンテージで表示されます。
例：
Available models (sorted by latest):
1: bert_20240606_200133_1m
2: gpt_20240606_150634_0m
3: gpt_20240606_150419_0m

特定のモデルを評価するには、以下を実行します。



### ステップ 6: Optuna を使ったハイパーパラメーターチューニング

BracketCloser-LLMは、大規模言語モデル（LLM）について学ぶためのプロジェクトで、簡単なLLMを作成し、実際にLLMが学習できることを体験します。限られたトークンセットを使用しており、簡単にルールを学習できます。

## プロジェクト概要
- **目的**: LLMを学ぶために簡単なLLMを作成し、学習プロセスを理解する。
- **トークン**: LLMは以下の9つのトークンを使用します。
  ```
  tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
  ```
- **学習タスク**: LLMは括弧を閉じるルールを学習します。
- **最新アーキテクチャ**: いくつかの最新のアーキテクチャ(gru,transformer,lstm,bert,gpt)を通じて、LLMの作成方法を学びます。

## 学習と評価の例
LLMは入力シーケンスに基づいて括弧を閉じる予測を学習します。以下はその例です：
