# modules/evaluate.py
import os
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from concurrent.futures import ProcessPoolExecutor
from modules.data_generator import generate_test_data, preprocess_and_save_dataset
from modules.custom_layers import CustomMultiHeadAttention


# ディレクトリ設定
dirs = {
    "original": "./components/dataset/original",
    "tokenize": "./components/dataset/tokenize",
    "preprocessed": "./components/dataset/preprocessed",
}

# 必要なディレクトリを作成
for dir_path in dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# トークンとIDを対応付ける辞書
tokens = ["[BOS]", "[SEP]", "[EOS]", "(", ")", "【", "】", "{", "}"]
token2id = {token: i + 1 for i, token in enumerate(tokens)}
id2token = {i + 1: token for i, token in enumerate(tokens)}

def get_model_type_from_model(model) -> str:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GRU):
            return "gru"
        elif isinstance(layer, tf.keras.layers.LSTM):
            return "lstm"
        elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
            return "transformer"
        elif isinstance(layer, tf.keras.layers.Dense) and 'bert' in layer.name.lower():
            return "bert"
        elif isinstance(layer, tf.keras.layers.Dense) and 'gpt' in layer.name.lower():
            return "gpt"
    return "unknown"

def load_dataset(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def preprocess_input(input_seq: str) -> List[int]:
    """
    "input:xxx,output" を想定し、[BOS] xxx [SEP] をモデル入力にする
    """
    try:
        input_part = input_seq.split("input:")[1].split(",output")[0]
    except Exception:
        input_part = ""
    token_list = ["[BOS]"] + list(input_part) + ["[SEP]"]
    logging.debug(f"Tokenized input part: {token_list}")
    return [token2id[token] for token in token_list if token in token2id]

def decode_output(output_seq: List[int]) -> str:
    decoded_tokens = []
    for tid in output_seq:
        tok = id2token.get(tid)
        if tok in ("[BOS]", "[SEP]", "[EOS]") or tok is None:
            continue
        decoded_tokens.append(tok)
    decoded = "".join(decoded_tokens)
    logging.debug(f"Decoded output: {decoded}")
    return decoded


def _count_missing_brackets(input_seq: str) -> int:
    bracket_pairs = {"(": ")", "【": "】", "{": "}"}
    opening_brackets = set(bracket_pairs.keys())
    closing_brackets = set(bracket_pairs.values())

    try:
        input_part = input_seq.split("input:")[1].split(",output")[0]
    except Exception:
        input_part = ""

    stack = []
    for token in input_part:
        if token in opening_brackets:
            stack.append(token)
        elif token in closing_brackets:
            if stack and bracket_pairs[stack[-1]] == token:
                stack.pop()
    return len(stack)

def split_input_output(data):
    input_output_pairs = []
    for item in data:
        if isinstance(item, str):
            input_seq = item.split(",output:")[0] + ",output"
            output_seq = item.split(",output:")[1]
            input_output_pairs.append((input_seq, output_seq))
        else:
            logging.error(f"Invalid data format: {item}")
    return input_output_pairs




def evaluate_model(model, test_data, model_type, model_save_path, epoch_num,
                  batch_size: int = 256, num_workers: int = 1,
                  evaluate_single: bool = True, eval_bracket_buckets: List[int] = (2, 3, 4),
                  max_decode_steps: int = 10) -> Dict[str, float]:
    """
    eval_bracket_buckets: 判定する括弧の数。4は「4回以上」扱い。
    evaluate_single=False で 1回括弧はスキップ。
    num_workers>1 で評価をチャンクに分けて並列化（各プロセスでモデルをロード）。
    """
    if not test_data:
        raise ValueError("Test data is empty. Please check if the dataset was generated and saved correctly.")

    input_output_pairs = split_input_output(test_data)
    if len(input_output_pairs) == 0:
        raise ValueError("No input-output pairs found in the test data.")

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        max_seq_length = input_shape[0][1]
    else:
        max_seq_length = input_shape[1]

    # 分割
    buckets = sorted(set(eval_bracket_buckets))
    chunk_size = max(1, len(input_output_pairs) // max(1, num_workers))
    chunks = [
        input_output_pairs[i:i + chunk_size]
        for i in range(0, len(input_output_pairs), chunk_size)
    ]

    def _collect_results(partials: List[Dict[str, Dict[str, float]]]) -> Dict[str, float]:
        agg_counts = {label: {"correct": 0, "total": 0, "records": []}
                      for label in ["bracket_1"] + [f"bracket_{b}" for b in buckets[:-1]] + ["bracket_4plus", "micro"]}

        for part in partials:
            for label, stats in part.items():
                if label not in agg_counts:
                    agg_counts[label] = {"correct": 0, "total": 0, "records": []}
                agg_counts[label]["correct"] += stats.get("correct", 0)
                agg_counts[label]["total"] += stats.get("total", 0)
                agg_counts[label]["records"].extend(stats.get("records", []))

        metrics = {}
        for label, stats in agg_counts.items():
            total = stats["total"]
            metrics[label] = stats["correct"] / total * 100 if total else 0.0
        # micro は 2以上の合算
        total_correct = sum(agg_counts[k]["correct"] for k in agg_counts if k not in ("bracket_1", "micro"))
        total_total = sum(agg_counts[k]["total"] for k in agg_counts if k not in ("bracket_1", "micro"))
        metrics["micro"] = total_correct / total_total * 100 if total_total else 0.0
        return agg_counts, metrics

    total_chunks = len(chunks)
    partials = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, part in enumerate(
                executor.map(
                    _eval_batch_worker,
                    [
                        (chunk, model_save_path, max_seq_length, batch_size, buckets, evaluate_single, max_decode_steps)
                        for chunk in chunks
                    ]
                ),
                start=1
            ):
                partials.append(part)
                print(f"[Eval] chunk {idx}/{total_chunks} done")
    else:
        # 単一プロセスでは渡された model をそのまま使い、再ロードを避けて高速化
        for idx, chunk in enumerate(chunks, start=1):
            part = _eval_inline(
                model=model,
                data_chunk=chunk,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                buckets=buckets,
                evaluate_single=evaluate_single,
                max_decode_steps=max_decode_steps,
            )
            partials.append(part)
            print(f"[Eval] chunk {idx}/{total_chunks} done")

    agg_counts, metrics = _collect_results(partials)

    # 結果をファイルに書き込む
    evaluation_result_path = "evaluation_result.txt"
    with open(evaluation_result_path, "w", encoding="utf-8") as f:
        f.write("[Evaluation Metrics]\n")
        f.write(f"bracket_2 (2回): {metrics.get('bracket_2', 0.0):.2f}% (正解数: {agg_counts.get('bracket_2', {}).get('correct', 0)}/{agg_counts.get('bracket_2', {}).get('total', 0)})\n")
        f.write(f"bracket_3 (3回): {metrics.get('bracket_3', 0.0):.2f}% (正解数: {agg_counts.get('bracket_3', {}).get('correct', 0)}/{agg_counts.get('bracket_3', {}).get('total', 0)})\n")
        f.write(f"bracket_4plus (4回以上): {metrics.get('bracket_4plus', 0.0):.2f}% (正解数: {agg_counts.get('bracket_4plus', {}).get('correct', 0)}/{agg_counts.get('bracket_4plus', {}).get('total', 0)})\n")
        if evaluate_single:
            f.write(f"bracket_1 (1回): {metrics.get('bracket_1', 0.0):.2f}% (正解数: {agg_counts.get('bracket_1', {}).get('correct', 0)}/{agg_counts.get('bracket_1', {}).get('total', 0)})\n")
        f.write(f"micro (2回以上合算): {metrics.get('micro', 0.0):.2f}%\n\n")

        for label in ["bracket_2", "bracket_3", "bracket_4plus"]:
            f.write(f"[{label} samples]\n")
            f.write("\n".join(agg_counts.get(label, {}).get("records", [])))
            f.write("\n\n")

    # モデル固有の結果を保存
    result_dir = os.path.join(os.path.dirname(model_save_path), "evaluation_results")
    os.makedirs(result_dir, exist_ok=True)

    result_filename = f"epoch_{epoch_num}_evaluation_result.txt"
    model_specific_result_path = os.path.join(result_dir, result_filename)

    with open(model_specific_result_path, "w", encoding="utf-8") as f:
        f.write("[Evaluation Metrics]\n")
        f.write(f"bracket_2 (2回): {metrics.get('bracket_2', 0.0):.2f}% (正解数: {agg_counts.get('bracket_2', {}).get('correct', 0)}/{agg_counts.get('bracket_2', {}).get('total', 0)})\n")
        f.write(f"bracket_3 (3回): {metrics.get('bracket_3', 0.0):.2f}% (正解数: {agg_counts.get('bracket_3', {}).get('correct', 0)}/{agg_counts.get('bracket_3', {}).get('total', 0)})\n")
        f.write(f"bracket_4plus (4回以上): {metrics.get('bracket_4plus', 0.0):.2f}% (正解数: {agg_counts.get('bracket_4plus', {}).get('correct', 0)}/{agg_counts.get('bracket_4plus', {}).get('total', 0)})\n")
        if evaluate_single:
            f.write(f"bracket_1 (1回): {metrics.get('bracket_1', 0.0):.2f}% (正解数: {agg_counts.get('bracket_1', {}).get('correct', 0)}/{agg_counts.get('bracket_1', {}).get('total', 0)})\n")
        f.write(f"micro (2回以上合算): {metrics.get('micro', 0.0):.2f}%\n\n")

        for label in ["bracket_2", "bracket_3", "bracket_4plus"]:
            f.write(f"[{label} samples]\n")
            f.write("\n".join(agg_counts.get(label, {}).get("records", [])))
            f.write("\n\n")

    print(f"2回括弧: {metrics.get('bracket_2', 0.0):.2f}% (n={agg_counts.get('bracket_2', {}).get('total', 0)})")
    print(f"3回括弧: {metrics.get('bracket_3', 0.0):.2f}% (n={agg_counts.get('bracket_3', {}).get('total', 0)})")
    print(f"4回以上括弧: {metrics.get('bracket_4plus', 0.0):.2f}% (n={agg_counts.get('bracket_4plus', {}).get('total', 0)})")
    print(f"micro (2回以上合算): {metrics.get('micro', 0.0):.2f}% (n={agg_counts.get('bracket_2', {}).get('total', 0) + agg_counts.get('bracket_3', {}).get('total', 0) + agg_counts.get('bracket_4plus', {}).get('total', 0)})")
    print(f"評価結果は {evaluation_result_path} に保存されました。")

    return metrics


def _eval_batch_worker(args: Tuple[List[Tuple[str, str]], str, int, int, List[int], bool, int]) -> Dict[str, Dict[str, float]]:
    chunk, model_path, max_seq_length, batch_size, buckets, evaluate_single, max_decode_steps = args

    model = load_model(model_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 配列で保持
    preprocessed_inputs = [preprocess_input(item[0]) for item in chunk]
    padded_inputs = pad_sequences(preprocessed_inputs, maxlen=max_seq_length, padding='post', value=0)

    # スタックと欠損括弧数
    bracket_pairs = {"(": ")", "【": "】", "{": "}"}
    opening_brackets = set(bracket_pairs.keys())
    closing_brackets = set(bracket_pairs.values())

    stacks = []
    missing_counts = []
    for input_seq, _ in chunk:
        try:
            input_part = input_seq.split("input:")[1].split(",output")[0]
        except Exception:
            input_part = ""
        stack = []
        for token in input_part:
            if token in opening_brackets:
                stack.append(token)
            elif token in closing_brackets and stack and bracket_pairs[stack[-1]] == token:
                stack.pop()
        stacks.append(stack)
        missing_counts.append(len(stack))

    counters = {
        "bracket_1": {"correct": 0, "total": 0, "records": []},
        "bracket_2": {"correct": 0, "total": 0, "records": []},
        "bracket_3": {"correct": 0, "total": 0, "records": []},
        "bracket_4plus": {"correct": 0, "total": 0, "records": []},
    }

    # バッチごとに生成
    batch_indices = [
        (i, min(i + batch_size, len(chunk)))
        for i in range(0, len(chunk), batch_size)
    ]

    for start, end in batch_indices:
        current_inputs = np.array(padded_inputs[start:end], copy=True)
        batch_stacks = [list(stacks[i]) for i in range(start, end)]
        batch_missing = missing_counts[start:end]
        generated_tokens = [[] for _ in range(end - start)]
        finished = [False] * (end - start)

        for _ in range(max_decode_steps):
            if isinstance(model.input, list):
                attn_mask = (current_inputs != 0).astype(np.float32)
                model_inputs = [current_inputs, attn_mask]
            else:
                model_inputs = current_inputs

            seq_logits = model.predict(model_inputs, verbose=0)  # (B, seq_len, vocab)

            for idx in range(end - start):
                if finished[idx]:
                    continue
                prefix_len = int(np.count_nonzero(current_inputs[idx]))
                pos = max(prefix_len - 1, 0)
                logits = seq_logits[idx, pos]

                if batch_stacks[idx]:
                    need = bracket_pairs[batch_stacks[idx][-1]]
                    allowed = [token2id[need]]
                else:
                    allowed = [token2id["[EOS]"]]

                masked = np.full_like(logits, -1e9, dtype=np.float32)
                masked[allowed] = logits[allowed]
                predicted_id = int(np.argmax(masked, axis=-1))

                if predicted_id == token2id["[EOS]"]:
                    finished[idx] = True
                    continue

                generated_tokens[idx].append(predicted_id)
                if batch_stacks[idx] and token2id[bracket_pairs[batch_stacks[idx][-1]]] == predicted_id:
                    batch_stacks[idx].pop()

                current_inputs[idx] = np.roll(current_inputs[idx], -1)
                current_inputs[idx][-1] = predicted_id

            if all(finished):
                break

        # 判定
        for local_idx, (input_seq, expected_output) in enumerate(chunk[start:end]):
            missing_count = batch_missing[local_idx]
            if missing_count == 1 and not evaluate_single:
                continue
            if missing_count == 1:
                label = "bracket_1"
            elif missing_count == 2:
                label = "bracket_2"
            elif missing_count == 3:
                label = "bracket_3"
            else:
                label = "bracket_4plus"

            predicted_output = decode_output(generated_tokens[local_idx])
            expected = expected_output.strip(",")
            is_correct = predicted_output == expected

            counters[label]["total"] += 1
            if is_correct:
                counters[label]["correct"] += 1
            counters[label]["records"].append(
                f"入力: {input_seq}\n出力: {predicted_output}\n正解: {expected}\n結果: {'正解' if is_correct else '不正解'}\n"
            )

    return counters


def _eval_inline(model, data_chunk, max_seq_length, batch_size, buckets, evaluate_single, max_decode_steps):
    """単一プロセス用：既存モデルを使って再ロードせずに評価する。"""
    # 配列で保持
    preprocessed_inputs = [preprocess_input(item[0]) for item in data_chunk]
    padded_inputs = pad_sequences(preprocessed_inputs, maxlen=max_seq_length, padding='post', value=0)

    bracket_pairs = {"(": ")", "【": "】", "{": "}"}
    opening_brackets = set(bracket_pairs.keys())
    closing_brackets = set(bracket_pairs.values())

    stacks = []
    missing_counts = []
    for input_seq, _ in data_chunk:
        try:
            input_part = input_seq.split("input:")[1].split(",output")[0]
        except Exception:
            input_part = ""
        stack = []
        for token in input_part:
            if token in opening_brackets:
                stack.append(token)
            elif token in closing_brackets and stack and bracket_pairs[stack[-1]] == token:
                stack.pop()
        stacks.append(stack)
        missing_counts.append(len(stack))

    counters = {
        "bracket_1": {"correct": 0, "total": 0, "records": []},
        "bracket_2": {"correct": 0, "total": 0, "records": []},
        "bracket_3": {"correct": 0, "total": 0, "records": []},
        "bracket_4plus": {"correct": 0, "total": 0, "records": []},
    }

    batch_indices = [
        (i, min(i + batch_size, len(data_chunk)))
        for i in range(0, len(data_chunk), batch_size)
    ]

    for start, end in batch_indices:
        current_inputs = np.array(padded_inputs[start:end], copy=True)
        batch_stacks = [list(stacks[i]) for i in range(start, end)]
        batch_missing = missing_counts[start:end]
        generated_tokens = [[] for _ in range(end - start)]
        finished = [False] * (end - start)

        for _ in range(max_decode_steps):
            if isinstance(model.input, list):
                attn_mask = (current_inputs != 0).astype(np.float32)
                model_inputs = [current_inputs, attn_mask]
            else:
                model_inputs = current_inputs

            seq_logits = model.predict(model_inputs, verbose=0)  # (B, seq_len, vocab)

            for idx in range(end - start):
                if finished[idx]:
                    continue
                prefix_len = int(np.count_nonzero(current_inputs[idx]))
                pos = max(prefix_len - 1, 0)
                logits = seq_logits[idx, pos]

                if batch_stacks[idx]:
                    need = bracket_pairs[batch_stacks[idx][-1]]
                    allowed = [token2id[need]]
                else:
                    allowed = [token2id["[EOS]"]]

                masked = np.full_like(logits, -1e9, dtype=np.float32)
                masked[allowed] = logits[allowed]
                predicted_id = int(np.argmax(masked, axis=-1))

                if predicted_id == token2id["[EOS]"]:
                    finished[idx] = True
                    continue

                generated_tokens[idx].append(predicted_id)
                if batch_stacks[idx] and token2id[bracket_pairs[batch_stacks[idx][-1]]] == predicted_id:
                    batch_stacks[idx].pop()

                current_inputs[idx] = np.roll(current_inputs[idx], -1)
                current_inputs[idx][-1] = predicted_id

            if all(finished):
                break

        for local_idx, (input_seq, expected_output) in enumerate(data_chunk[start:end]):
            missing_count = batch_missing[local_idx]
            if missing_count == 1 and not evaluate_single:
                continue
            if missing_count == 1:
                label = "bracket_1"
            elif missing_count == 2:
                label = "bracket_2"
            elif missing_count == 3:
                label = "bracket_3"
            else:
                label = "bracket_4plus"

            predicted_output = decode_output(generated_tokens[local_idx])
            expected = expected_output.strip(",")
            is_correct = predicted_output == expected

            counters[label]["total"] += 1
            if is_correct:
                counters[label]["correct"] += 1
            counters[label]["records"].append(
                f"入力: {input_seq}\n出力: {predicted_output}\n正解: {expected}\n結果: {'正解' if is_correct else '不正解'}\n"
            )

    return counters


def main(model_path, epoch_num, num_test_samples=500, batch_size=256, num_workers=1,
         evaluate_single=True, eval_bracket_buckets=(2, 3, 4), max_decode_steps=10):
    # モデルのロード
    model = load_model(model_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})

    # モデルタイプの取得
    model_type = get_model_type_from_model(model)
    logging.info(f"Model type: {model_type}")

    # モデルのコンパイル
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # テストデータの生成
    test_dataset = generate_test_data(num_test_samples)

    # テストデータの前処理と保存
    preprocess_and_save_dataset(test_dataset, "test_bracket_dataset.json", max_seq_length=30)

    # テストデータの保存パス
    test_data_path = os.path.join(dirs["original"], "test_bracket_dataset.json")

    # テストデータのロード
    test_data = load_dataset(test_data_path)

    # モデルの評価
    print(f"[Eval] start: samples={num_test_samples}, batch_size={batch_size}, workers={num_workers}, buckets={eval_bracket_buckets}")
    metrics = evaluate_model(
        model=model,
        test_data=test_data,
        model_type=model_type,
        model_save_path=model_path,
        epoch_num=epoch_num,
        batch_size=batch_size,
        num_workers=num_workers,
        evaluate_single=evaluate_single,
        eval_bracket_buckets=eval_bracket_buckets,
        max_decode_steps=max_decode_steps,
    )
    print(f"評価結果: {metrics}")
    print(f"評価結果は evaluation_result.txt に保存されました。")

    return metrics
