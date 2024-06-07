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

# モデルの選択
def get_model_list_sorted_by_date(models_dir='./models'):
    model_list = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    # 不要なフォルダを除外
    model_list = [d for d in model_list if d not in ["残すモデル", "trash"]]
    
    def extract_date(model_name):
        try:
            parts = model_name.split('_')
            if len(parts) >= 3:
                timestamp = parts[1] + parts[2]  # '20240604_213643' => '20240604213643'
                return datetime.strptime(timestamp, '%Y%m%d%H%M%S')
            else:
                raise ValueError("Invalid model name format")
        except Exception as e:
            print(f"Error parsing date from {model_name}: {e}")
            return datetime.min


    model_list.sort(key=extract_date, reverse=True)
    return model_list[:9]  # 最新の9個のモデルのみを表示


def select_model(models_dir='./models'):
    model_list = get_model_list_sorted_by_date(models_dir)
    print("Available models (sorted by latest):")
    for idx, model_name in enumerate(model_list, 1):
        print(f"{idx}: {model_name}")
    
    selected_index = int(input("Select the model index: ")) - 1
    selected_model_path = os.path.join(models_dir, model_list[selected_index])
    return selected_model_path

# モデルの保存パス
model_save_path = select_model()
model_metadata_path = os.path.join(model_save_path, "training_info.json")

# テストデータの保存パス
test_data_path = os.path.join(dirs["original"], "test_bracket_dataset.json")

# 評価結果の保存パス
evaluation_result_path = "evaluation_result.txt"

# トークンとIDを対応付ける辞書
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
token2id = {token: i + 1 for i, token in enumerate(tokens)}
id2token = {i + 1: token for i, token in enumerate(tokens)}



# モデルメタデータをロードしてモデルの種類を自動設定
def get_model_type(metadata_path: str) -> str:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model_architecture = metadata.get("model_architecture", "")
    if "gru" in model_architecture.lower():
        return "gru"
    elif "transformer" in model_architecture.lower():
        return "transformer"
    elif "lstm" in model_architecture.lower():
        return "lstm"
    elif "bert" in model_architecture.lower():
        return "bert"
    elif "gpt" in model_architecture.lower():
        return "gpt"
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

model_type = get_model_type(model_metadata_path)

# モデルのロード
model_path = os.path.join(model_save_path, "best_model.h5")
model = load_model(model_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})

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

    model_specific_result_path = os.path.join(model_save_path, result_filename)
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