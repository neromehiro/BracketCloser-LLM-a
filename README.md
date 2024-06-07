# BracketCloser-LLM プロジェクト

BracketCloser-LLMは、大規模言語モデル（LLM）について学ぶためのプロジェクトで、簡単なLLMを作成し、実際にLLMが学習できることを体験します。限られたトークンセットを使用しており、簡単にルールを学習できます。本プロジェクトは、基本的なLLMの理解を深めるための実践的なステップを提供します。

## プロジェクト概要

- **目的**: 簡単なLLMを作成し、LLMの学習プロセスを理解すること。
- **トークン**: 以下の9つのトークンを使用します。(ID0はパディング用トークン)
  ```python
  tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
  ```
- **学習タスク**: LLMは括弧を閉じるルールを学習します。具体的には、`input`トークンの中身に基づいて、`output`トークンの右側に現れる括弧を予測します。
- **アーキテクチャ**: 以下の最新のアーキテクチャ（GRU、Transformer、LSTM、BERT、GPT）を使用して、LLMの作成方法を学びます。

## 使い方

### ステップ 1: Docker イメージの取得

まず、プロジェクトのDockerイメージを取得します。

```sh
docker pull nero1014/bracket-closer-image
```

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

```sh
python evaluate2.py
```

評価したいモデルの相対パスを入力します。

### ステップ 6: Optuna を使ったハイパーパラメーターチューニング

以下のコマンドでハイパーパラメーターチューニングを行います。

```sh
python hyper.py
```

過去の学習を継続するか、新規で学習を始めるかを選べます。モデルアーキテクチャ、学習時間、並列ジョブ数（1〜5推奨）を入力します。

例:

```sh
ubuntu@8326ea14da18:/app/project$ python hyper.py
Choose an option:
1. Resume existing study
2. Start a new study
Enter 1 or 2: 2 ←入力(新規学習を選択)
Enter the model architecture (gru, transformer, lstm, bert, gpt): gru ←入力(モデルを選択)
Enter the training time limit (e.g., '3min', '1hour', '5hour'): 3min ←入力(タイムリミットを選択)
Optimization Progress:   0%|             | 0/180.0 [00:00<?, ?s/s]Enter the number of parallel jobs: 1 ←入力(並列ジョブ数を選択)
[I 2024-06-06 11:08:39,182] A new study created in RDB with name: hyper_gru_3
```

モデルは `optuna_studies` フォルダに保存されます。`evaluate2.py` を使ってそのモデルを評価します。

## 学習と評価の詳細

### 学習プロセス

LLMは、括弧の開閉ルールを学習するために設計されています。具体的には、`input`トークンの内容に基づいて、`output`トークンの右側に現れる括弧のパターンを予測します。この学習プロセスを通じて、モデルは入力シーケンスと出力シーケンスの関係を理解し、適切な括弧の閉じ方を学びます。

#### データセットの例

データセットは、モデルが学習するためのサンプル入力と出力のペアを含んでいます。以下はその一例です。

originalのデータセットの例
original = [
    "input:【()】{{{()}}(({})),output:}",
    "input:(【{{}}{(){}}】,output:)",
]

トークン化されたデータセットの例(0はパディングトークン.)
["①(", "②)", "③【", "④】", "⑤{", "⑥}", "⑦input", "⑧,output", "⑨,"]
preprocessed_sequence = [
    [3, 1, 2, 4, 5, 5, 5, 1, 2, 6, 6, 1, 1, 5, 6, 2, 2, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 3, 5, 5, 6, 6, 5, 1, 2, 5, 6, 6, 4, 9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

]


### モデルの評価

学習が完了したモデルの評価は、指定した入力に対する出力を確認することで行います。以下に、評価結果の一例を示します。
順番も含めて完全正解のみを正解とします。

#### 出力例 (evaluation_result.txt)

問題1 不正解
入力した単語 Input: input:({【】【】()}【,output
出力の単語: 】】
正解の単語: )】

問題2 正解
入力した単語 Input: input:【【{}{}{}()】,output
出力の単語: 】
正解の単語: 】

問題3 不正解
入力した単語 Input: input:{}【(【(),output
出力の単語: 】】)
正解の単語: 】)】

問題4 正解
入力した単語 Input: input:【】【】【】(){,output
出力の単語: }
正解の単語: }

これらの評価結果から、モデルがどの程度正確に括弧の閉じ方を学習したかを確認できます。

### Optunaによるハイパーパラメーターチューニング

Optunaを用いることで、モデルのパフォーマンスを最適化するためのハイパーパラメーターチューニングを行います。これにより、学習プロセスを効率的にし、より高精度なモデルを作成することが可能です。

#### チューニングプロセスの例

Optunaを使用してハイパーパラメータを調整する際には、以下のようなステップを踏みます。

```sh
python hyper.py
```

選択肢として、既存の学習を再開するか、新規の学習を開始するかを選べます。次に、モデルアーキテクチャ、学習時間、並列ジョブ数を入力します。以下は、そのプロセスの一例です。

```sh
ubuntu@8326ea14da18:/app/project$ python hyper.py
Choose an option:
1. Resume existing study
2. Start a new study
Enter 1 or 2: 2
Enter the model architecture (gru, transformer, lstm, bert, gpt): gru
Enter the training time limit (e.g., '3min', '1hour', '5hour'): 3min
Optimization Progress:   0%|             | 0/180.0 [00:00<?, ?s/s]Enter the number of parallel jobs: 1
[I 2024-06-06 11:08:39,182] A new study created in RDB with name: hyper_gru_3
```

このようにして、効率的なモデルチューニングが可能となり、最適なパフォーマンスを引き出すことができます。

## まとめ

BracketCloser-LLMプロジェクトは、LLMの基本的な学習プロセスを理解するための優れた機会を提供します。限られたトークンセットを使用し、様々な最新のアーキテクチャを通じて、括弧の閉じ方を学習するモデルを作成することで、実際のLLMの動作を深く理解することができます。プロジェクトの各ステップを通じて、実践的なスキルを習得し、LLMの可能性を探求してみてください。
