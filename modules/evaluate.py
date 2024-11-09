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
    
    correct_predictions = 0
    partial_correct_predictions = 0
    results = []

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        max_seq_length = input_shape[0][1]
    else:
        max_seq_length = input_shape[1]

    input_output_pairs = split_input_output(test_data)
    
    if len(input_output_pairs) == 0:
        raise ValueError("No input-output pairs found in the test data.")

    for idx, (input_seq, expected_output) in enumerate(input_output_pairs):
        preprocessed_input = preprocess_input(input_seq)
        preprocessed_input_padded = pad_sequences(
            [preprocessed_input], maxlen=max_seq_length, padding='post', value=0
        )[0]

        expected_output_tokens = preprocess_input(expected_output)
        predicted_output_ids = []
        
        # カンマが出力されるか、10個のトークンが生成されるまで予測を継続
        for i in range(10):  # 最大10個まで
            if isinstance(model.input, list):
                model_inputs = [np.array([preprocessed_input_padded]), np.array([preprocessed_input_padded])]
            else:
                model_inputs = np.array([preprocessed_input_padded])

            predicted_output = model.predict(model_inputs, verbose=0) 
            predicted_id = np.argmax(predicted_output[0], axis=-1)
            predicted_output_ids.append(predicted_id)

            # カンマが出力されたら終了
            if predicted_id == token2id[","]:
                break

            # 入力をスライドして次のトークンを予測
            if len(preprocessed_input_padded) < max_seq_length:
                preprocessed_input_padded = np.concatenate([preprocessed_input_padded, [predicted_id]])
            else:
                preprocessed_input_padded = np.roll(preprocessed_input_padded, -1)
                preprocessed_input_padded[-1] = predicted_id

        predicted_output = decode_output(predicted_output_ids)
        expected_output_reconstructed = decode_output(expected_output_tokens)

        if predicted_output == expected_output_reconstructed:
            results.append(f"問題{idx + 1} 完全正解\n入力した単語 Input: {input_seq}\n出力の単語: {predicted_output}\n正解の単語: {expected_output_reconstructed}\n")
            correct_predictions += 1
        elif sorted(predicted_output) == sorted(expected_output_reconstructed):
            results.append(f"問題{idx + 1} 部分正解\n入力した単語 Input: {input_seq}\n出力の単語: {predicted_output}\n正解の単語: {expected_output_reconstructed}\n")
            partial_correct_predictions += 1
        else:
            results.append(f"問題{idx + 1} 不正解\n入力した単語 Input: {input_seq}\n出力の単語: {predicted_output}\n正解の単語: {expected_output_reconstructed}\n")

    total_cases = len(input_output_pairs)
    complete_accuracy = correct_predictions / total_cases * 100
    partial_accuracy = (correct_predictions + partial_correct_predictions) / total_cases * 100

    # Add the complete and partial accuracy to the beginning of the result
    result_filename = f"epoch_{epoch_num}_evaluation_result_{complete_accuracy:.2f}%_complete_{partial_accuracy:.2f}%_partial.txt"
    evaluation_result_path = "evaluation_result.txt"
    
    with open(evaluation_result_path, "w", encoding="utf-8") as f:
        # Write the accuracies at the top
        f.write(f"完全正解率: {complete_accuracy:.2f}%\n")
        f.write(f"部分正解率: {partial_accuracy:.2f}%\n")
        f.write("\n".join(results))  # Then the problem results

    result_dir = os.path.join(os.path.dirname(model_save_path), "evaluation_results")
    os.makedirs(result_dir, exist_ok=True)

    model_specific_result_path = os.path.join(result_dir, result_filename)
    
    with open(model_specific_result_path, "w", encoding="utf-8") as f:
        # Write the accuracies at the top
        f.write(f"完全正解率: {complete_accuracy:.2f}%\n")
        f.write(f"部分正解率: {partial_accuracy:.2f}%\n")
        f.write("\n".join(results))  # Then the problem results

    return complete_accuracy, partial_accuracy

def main(model_path, epoch_num, num_test_samples=100):
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

