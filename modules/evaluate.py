import os
import json
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 追加
from modules.custom_layers import CustomMultiHeadAttention
from modules.data_generator import generate_test_data, preprocess_and_save_dataset


# ログ設定
# logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, 
#                     format='%(asctime)s %(levelname)s: %(message)s')

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

def load_dataset(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def tokenize_string(string: str) -> list:
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

def preprocess_input(input_seq: str) -> list:
    tokens = tokenize_string(input_seq)
    logging.debug(f"Tokenized string: {tokens}")  # デバッグ: トークン化された文字列をログに記録
    return [token2id[token] for token in tokens if token in token2id]

def decode_output(output_seq: list) -> str:
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
        
        results = []
        for input_seq, expected_output_seq in input_output_pairs:
            preprocessed_input = pad_sequences([preprocess_input(input_seq)], maxlen=max_seq_length, padding='post')
            prediction = model.predict(preprocessed_input)
            predicted_output = decode_output(np.argmax(prediction, axis=-1).flatten().tolist())
            correct_predictions += int(predicted_output == expected_output_seq)
            results.append((input_seq, expected_output_seq, predicted_output))
        
        accuracy = correct_predictions / len(input_output_pairs)
        logging.info(f"Accuracy: {accuracy}")
        
        return accuracy, results
    
    except Exception as e:
        logging.error(f"Error in evaluate_model_instance: {e}")
        raise e

def evaluate_model(model_path, model_type):
    # 200個のテストデータセットを生成
    test_dataset = generate_test_data(200)
    preprocess_and_save_dataset(test_dataset, f"temp_test_dataset.json", max_seq_length=30)
    test_data_path = os.path.join(dirs["original"], "temp_test_dataset.json")
    test_data = load_dataset(test_data_path)
    
    # 評価を実行
    accuracy, results = evaluate_model_instance(model_path, test_data, model_type)
    
    # 結果を保存
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        for input_seq, expected_output_seq, predicted_output in results:
            f.write(f"Input: {input_seq}\nExpected: {expected_output_seq}\nPredicted: {predicted_output}\n\n")
    
    return accuracy * 100  # 100点満点で評価

def perform_multiple_trials(model_path, model_type, num_trials=10):
    trial_results = []
    for _ in range(num_trials):
        trial_accuracy = evaluate_model(model_path, model_type)
        trial_results.append(trial_accuracy)
    return np.mean(trial_results)

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
    num_trials = 5  # 必要に応じて変更

    # モデルの評価
    average_accuracy = perform_multiple_trials(model_save_path, model_type, num_trials)
    print(f"モデルの平均精度: {average_accuracy:.2f}%")
    return average_accuracy
