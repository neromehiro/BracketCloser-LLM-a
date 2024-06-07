import os
import json
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List
import sys
from datetime import datetime
from modules.data_generator import generate_test_data, preprocess_and_save_dataset
from modules.custom_layers import CustomMultiHeadAttention

# モデルの保存パス
# model_save_path = 'optuna_studies/hyper_gpt/temp_model_1.h5'
# model_save_path = 'optuna_studies/hyper_gpt/temp_model_2.h5'
# model_save_path = 'optuna_studies/hyper_gru/temp_model_27.h5' # 70パーセント以上
# model_save_path = 'optuna_studies/hyper_gru/temp_model_30.h5' # 60パーセント以上
# model_save_path = 'optuna_studies/hyper_lstm/temp_model_70_20240605183703.h5' # 76パーセント以上 68 66 54
# model_save_path = 'optuna_studies/hyper_lstm/temp_model_70_20240605183703.h5' # 意外と良い 71 53 63 66 79
# model_save_path = 'optuna_studies/hyper_lstm_3/best_model.h5' # 意外と良い 65 67 53 64　６７
# model_save_path = 'optuna_studies/hyper_lstm_3/hyper_lstm_3/temp_model_2_20240606081451.h5' # 点数：61 57 58
# model_save_path = 'optuna_studies/hyper_lstm_1/hyper_lstm_29/temp_model_28_20240607020449_epoch_5_pid_20909.h5' # 点数：70 61 64 65 74
# model_save_path = 'models/20240607_023634_lstm_temp/best_model_epoch_4_pid_8566.h5' # 点数： 73 59 73 74 66 78
# model_save_path = 'models/20240607_123655_lstm_temp/best_model.h5' # 点数： 77 71 75 71 67 68 70 70 73
model_save_path = 'optuna_studies/hyper_transformer_2/hyper_transformer_151/temp_model_150_20240607163731.h5' # 点数：
# model_save_path = '' # 点数：

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# ログ設定
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# ディレクトリ設定
dirs = {
    "original": "./components/dataset/original",
    "tokenize": "./components/dataset/tokenize",
    "preprocessed": "./components/dataset/preprocessed",
}

# 必要なディレクトリを作成
for dir_path in dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# モデルの保存パス


model_path = model_save_path

# テストデータの保存パス
test_data_path = os.path.join(dirs["original"], "test_bracket_dataset.json")

# 評価結果の保存パス
evaluation_result_path = "evaluation_result.txt"

# トークンとIDを対応付ける辞書
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
token2id = {token: i + 1 for i, token in enumerate(tokens)}
id2token = {i + 1: token for i, token in enumerate(tokens)}

# モデルのアーキテクチャを判定
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
        # Add other model types as necessary
    return "unknown"


# モデルのロード
model = load_model(model_save_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})

# モデルタイプの取得
model_type = get_model_type_from_model(model)
logging.info(f"Model type: {model_type}")

# モデルのコンパイル
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデルの期待する入力シーケンスの長さを取得
expected_input_shape = model.input_shape[1]
default_max_seq_length = expected_input_shape

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
    if (current_token):
        tokens.append(current_token)
    return tokens

def preprocess_input(input_seq: str) -> List[int]:
    tokens = tokenize_string(input_seq)
    logging.debug(f"Tokenized string: {tokens}")  # デバッグ: トークン化された文字列をログに記録
    return [token2id[token] for token in tokens if token in token2id]

def decode_output(output_seq: List[int]) -> str:
    decoded = "".join([id2token[id] for id in output_seq if id in id2token])
    logging.debug(f"Decoded output: {decoded}")  # デバッグ: デコードされた出力をログに記録
    return decoded

def split_input_output(data):
    input_output_pairs = []
    for item in data:
        if isinstance(item, str):  # itemが文字列であることを確認
            input_seq = item.split(",output:")[0] + ",output"
            output_seq = item.split(",output:")[1]
            input_output_pairs.append((input_seq, output_seq))
        else:
            logging.error(f"Invalid data format: {item}")
    return input_output_pairs

def evaluate_model(model, test_data, model_type):
    if not test_data:
        raise ValueError("Test data is empty. Please check if the dataset was generated and saved correctly.")
    
    correct_predictions = 0
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
        for i in range(len(expected_output_tokens)):
            if isinstance(model.input, list):
                model_inputs = [np.array([preprocessed_input_padded]), np.array([preprocessed_input_padded])]
            else:
                model_inputs = np.array([preprocessed_input_padded])

            predicted_output = model.predict(model_inputs)
            predicted_id = np.argmax(predicted_output[0], axis=-1)
            predicted_output_ids.append(predicted_id)

            if len(preprocessed_input_padded) < max_seq_length:
                preprocessed_input_padded = np.concatenate([preprocessed_input_padded, [predicted_id]])
            else:
                preprocessed_input_padded = np.roll(preprocessed_input_padded, -1)
                preprocessed_input_padded[-1] = predicted_id

        predicted_output = decode_output(predicted_output_ids)
        expected_output_reconstructed = decode_output(expected_output_tokens)

        if predicted_output == expected_output_reconstructed:
            results.append(f"問題{idx + 1} 正解\n入力した単語 Input: {input_seq}\n出力の単語: {predicted_output}\n正解の単語: {expected_output_reconstructed}\n")
            correct_predictions += 1
        else:
            results.append(f"問題{idx + 1} 不正解\n入力した単語 Input: {input_seq}\n出力の単語: {predicted_output}\n正解の単語: {expected_output_reconstructed}\n")

    accurate_percentage = correct_predictions / len(input_output_pairs) * 100

    result_filename = f"evaluation_result_{accurate_percentage:.2f}%.txt"
    with open(evaluation_result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
        f.write(f"\nAccuracy: {accurate_percentage:.2f}%")

    result_dir = os.path.join(os.path.dirname(model_save_path), "evaluation_results")  # 結果保存ディレクトリ
    os.makedirs(result_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成

    model_specific_result_path = os.path.join(result_dir, result_filename)
    with open(model_specific_result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
        f.write(f"\nAccuracy: {accurate_percentage:.2f}%")

    return accurate_percentage

# テストデータのサンプル数
num_test_samples = 100

# テストデータの生成
test_dataset = generate_test_data(num_test_samples)

# テストデータの前処理と保存
preprocess_and_save_dataset(test_dataset, "test_bracket_dataset.json", max_seq_length=30)

# テストデータのロード
test_data = load_dataset(test_data_path)

# モデルの評価
accuracy = evaluate_model(model, test_data, model_type)
print(f"モデルの精度: {accuracy:.2f}%")
print(f"評価結果は {evaluation_result_path} に保存されました。")