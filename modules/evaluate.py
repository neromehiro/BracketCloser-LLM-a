# modules/evaluate.py
import os
import json
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List
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
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
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

def tokenize_string(string: str) -> List[str]:
    tokens = []
    current_token = ""
    for char in string:
        if char in token2id:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens

def preprocess_input(input_seq: str) -> List[int]:
    tokens = tokenize_string(input_seq)
    logging.debug(f"Tokenized string: {tokens}")
    return [token2id[token] for token in tokens if token in token2id]

def decode_output(output_seq: List[int]) -> str:
    decoded = "".join([id2token[id] for id in output_seq if id in id2token])
    logging.debug(f"Decoded output: {decoded}")
    return decoded

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




def evaluate_model(model, test_data, model_type, model_save_path, epoch_num):
    if not test_data:
        raise ValueError("Test data is empty. Please check if the dataset was generated and saved correctly.")

    # カテゴリーごとの正答数と総問題数を初期化
    one_answer_correct = 0
    one_answer_total = 0
    multi_answer_correct = 0
    multi_answer_total = 0

    # 結果をカテゴリー別に保存するリスト
    results_one_answer = []
    results_multi_answer = []

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        max_seq_length = input_shape[0][1]
    else:
        max_seq_length = input_shape[1]

    input_output_pairs = split_input_output(test_data)

    if len(input_output_pairs) == 0:
        raise ValueError("No input-output pairs found in the test data.")

    # 括弧の対応関係と判定に使用するセット
    bracket_pairs = {"(": ")", "【": "】", "{": "}"}
    opening_brackets = set(bracket_pairs.keys())
    closing_brackets = set(bracket_pairs.values())

    for idx, (input_seq, expected_output) in enumerate(input_output_pairs):
        preprocessed_input = preprocess_input(input_seq)
        preprocessed_input_padded = pad_sequences(
            [preprocessed_input], maxlen=max_seq_length, padding='post', value=0
        )[0]

        # 入力シーケンスから未閉じの開き括弧の数をカウント
        input_tokens = tokenize_string(input_seq)

        # エンドトークンを除外
        if ",output" in input_tokens:
            end_index = input_tokens.index(",output")
            input_tokens = input_tokens[:end_index]
        else:
            # ",output" が存在しない場合の対応
            pass  # または適切な処理を追加

        stack = []
        for token in input_tokens:
            if token in opening_brackets:
                stack.append(token)
            elif token in closing_brackets:
                if stack and bracket_pairs[stack[-1]] == token:
                    stack.pop()

        num_missing_brackets = len(stack)  # 未閉じの開き括弧の数が必要な回答数

        predicted_output_ids = []

        # モデルの予測を実行
        for i in range(10):  # 最大10個まで
            if isinstance(model.input, list):
                model_inputs = [np.array([preprocessed_input_padded]), np.array([preprocessed_input_padded])]
            else:
                model_inputs = np.array([preprocessed_input_padded])

            predicted_output = model.predict(model_inputs, verbose=0)
            predicted_id = np.argmax(predicted_output[0], axis=-1)

            # エンドトークンをチェック（ここではカンマ "," がエンドトークンとして使用されています）
            if predicted_id == token2id[","]:
                break

            predicted_output_ids.append(predicted_id)

            # 入力をスライドして次のトークンを予測
            if len(preprocessed_input_padded) < max_seq_length:
                preprocessed_input_padded = np.concatenate([preprocessed_input_padded, [predicted_id]])
            else:
                preprocessed_input_padded = np.roll(preprocessed_input_padded, -1)
                preprocessed_input_padded[-1] = predicted_id

        # エンドトークンを除外してデコード
        predicted_output = decode_output(predicted_output_ids)
        expected_output = expected_output.strip(",")  # 正解の出力からエンドトークンを削除

        # 予測結果と期待される出力を比較
        is_correct = predicted_output == expected_output

        # 問題をカテゴリー分けして結果を保存
        if num_missing_brackets == 1:
            one_answer_total += 1
            if is_correct:
                one_answer_correct += 1
                results_one_answer.append(f"問題{idx + 1} 正解\n入力: {input_seq}\n出力: {predicted_output}\n正解: {expected_output}\n")
            else:
                results_one_answer.append(f"問題{idx + 1} 不正解\n入力: {input_seq}\n出力: {predicted_output}\n正解: {expected_output}\n")
        elif num_missing_brackets >= 2:
            multi_answer_total += 1
            if is_correct:
                multi_answer_correct += 1
                results_multi_answer.append(f"問題{idx + 1} 正解\n入力: {input_seq}\n出力: {predicted_output}\n正解: {expected_output}\n")
            else:
                results_multi_answer.append(f"問題{idx + 1} 不正解\n入力: {input_seq}\n出力: {predicted_output}\n正解: {expected_output}\n")
        else:
            # 未閉じの括弧がない場合、評価から除外する（必要に応じて）
            pass

    # 正答率の計算
    if one_answer_total > 0:
        complete_accuracy = one_answer_correct / one_answer_total * 100
    else:
        complete_accuracy = 0.0

    if multi_answer_total > 0:
        partial_accuracy = multi_answer_correct / multi_answer_total * 100
    else:
        partial_accuracy = 0.0

    # 結果をファイルに書き込む
    evaluation_result_path = "evaluation_result.txt"
    with open(evaluation_result_path, "w", encoding="utf-8") as f:
        f.write(f"完全正解率 (1問における正答率): {complete_accuracy:.2f}% (正解数: {one_answer_correct}/{one_answer_total})\n")
        f.write(f"部分正解率 (2問以上の正答率): {partial_accuracy:.2f}% (正解数: {multi_answer_correct}/{multi_answer_total})\n\n")

        f.write("【1問における問題の結果】\n")
        f.write("\n".join(results_one_answer))
        f.write("\n\n【2問以上における問題の結果】\n")
        f.write("\n".join(results_multi_answer))

    # モデル固有の結果を保存
    result_dir = os.path.join(os.path.dirname(model_save_path), "evaluation_results")
    os.makedirs(result_dir, exist_ok=True)

    result_filename = f"epoch_{epoch_num}_evaluation_result.txt"
    model_specific_result_path = os.path.join(result_dir, result_filename)

    with open(model_specific_result_path, "w", encoding="utf-8") as f:
        f.write(f"完全正解率 (1問における正答率): {complete_accuracy:.2f}% (正解数: {one_answer_correct}/{one_answer_total})\n")
        f.write(f"部分正解率 (2問以上の正答率): {partial_accuracy:.2f}% (正解数: {multi_answer_correct}/{multi_answer_total})\n\n")

        f.write("【1問における問題の結果】\n")
        f.write("\n".join(results_one_answer))
        f.write("\n\n【2問以上における問題の結果】\n")
        f.write("\n".join(results_multi_answer))

    # 結果を表示
    print(f"完全正解率 (1問における正答率): {complete_accuracy:.2f}% (正解数: {one_answer_correct}/{one_answer_total})")
    print(f"部分正解率 (2問以上の正答率): {partial_accuracy:.2f}% (正解数: {multi_answer_correct}/{multi_answer_total})")
    print(f"評価結果は {evaluation_result_path} に保存されました。")

    return complete_accuracy, partial_accuracy

def main(model_path, epoch_num, num_test_samples=500):
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
    complete_accuracy, partial_accuracy = evaluate_model(model, test_data, model_type, model_path, epoch_num)
    print(f"モデルの完全正解率: {complete_accuracy:.2f}%")
    print(f"モデルの部分正解率: {partial_accuracy:.2f}%")
    print(f"評価結果は evaluation_result.txt に保存されました。")

    return complete_accuracy, partial_accuracy

