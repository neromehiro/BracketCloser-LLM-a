import os
import json
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from modules.data_generator import generate_test_data, preprocess_and_save_dataset
from modules.custom_layers import CustomMultiHeadAttention

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

def load_training_info(model_dir):
    training_info_path = os.path.join(model_dir, "training_info.json")
    if os.path.exists(training_info_path):
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        return training_info
    else:
        raise FileNotFoundError(f"Training info file not found in {model_dir}")

def evaluate_model_instance(model_path, test_data, model_type):
    try:
        model = load_model(model_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})
        
        # モデルディレクトリからトレーニング設定を読み込む
        model_dir = os.path.dirname(model_path)
        training_info = load_training_info(model_dir)
        
        # モデルの手動コンパイル
        try:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_info['learning_rate']), 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])
            logging.info(f"Model compiled successfully with learning rate {training_info['learning_rate']}")
        except Exception as e:
            logging.error(f"Model compilation failed: {e}")
            raise e
        
        correct_predictions = 0
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            max_seq_length = input_shape[0][1]
        else:
            max_seq_length = input_shape[1]

        input_output_pairs = split_input_output(test_data)
        
        if len(input_output_pairs) == 0:
            raise ValueError("No input-output pairs found in the test data.")
        
        logging.info(f"Evaluating {len(input_output_pairs)} input-output pairs")
        # ここでモデルの評価を行う処理を追加
        # dummy return for example
        return 0.9
    
    except Exception as e:
        logging.error(f"Error in evaluate_model_instance: {e}")
        raise e

def perform_multiple_trials(model_path, model_type, num_trials=10):
    trial_results = []
    for _ in range(num_trials):
        trial_accuracy = evaluate_model(model_path, model_type)
        trial_results.append(trial_accuracy)
    return np.mean(trial_results)


def evaluate_model(model_path, model_type, num_chunks=20, chunk_size=5):
    results = []
    for _ in range(num_chunks):
        chunk_results = []
        for _ in range(chunk_size):
            test_dataset = generate_test_data(chunk_size)
            preprocess_and_save_dataset(test_dataset, f"temp_test_dataset.json", max_seq_length=30)
            test_data_path = os.path.join(dirs["original"], "temp_test_dataset.json")
            test_data = load_dataset(test_data_path)
            accuracy = evaluate_model_instance(model_path, test_data, model_type)
            chunk_results.append(accuracy)
        results.append(np.mean(chunk_results) * 100)  # 100点満点で評価
    return np.mean(results)

def perform_multiple_trials(model_path, model_type, num_trials=10):
    trial_results = []
    for _ in range(num_trials):
        trial_accuracy = evaluate_model(model_path, model_type)
        trial_results.append(trial_accuracy)
    return np.mean(trial_results)

# エントリーポイント
def main(model_save_path):
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

    # モデルのロード
    model = load_model(model_save_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})
    
    # モデルディレクトリからトレーニング設定を読み込む
    model_dir = os.path.dirname(model_save_path)
    training_info = load_training_info(model_dir)
    
    # モデルの手動コンパイル
    try:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_info['learning_rate']), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        logging.info(f"Model compiled successfully with learning rate {training_info['learning_rate']}")
    except Exception as e:
        logging.error(f"Model compilation failed: {e}")
        print(f"Model compilation failed: {e}")
        return

    # モデルタイプの取得
    model_type = get_model_type_from_model(model)
    logging.info(f"Model type: {model_type}")

    # 評価回数を設定
    num_trials = 10  # 必要に応じて変更

    # モデルの評価
    # 評価回数を設定
    num_trials = 10  # 必要に応じて変更

    # モデルの評価
    average_accuracy = perform_multiple_trials(model_save_path, model_type, num_trials)
    print(f"モデルの平均精度: {average_accuracy:.2f}%")
    return average_accuracy


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <model_save_path>")
    else:
        model_save_path = sys.argv[1]
        main(model_save_path)
