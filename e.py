import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from modules.custom_layers import CustomMultiHeadAttention  # ここで CustomMultiHeadAttention をインポート
from modules.evaluate_model import evaluate_model_instance, evaluate_model,load_dataset, get_model_type_from_model
import logging

# ログ設定
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# 'modules' ディレクトリをシステムパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from evaluate_model import evaluate_model_instance, evaluate_model, load_dataset, get_model_type_from_model
from tensorflow.keras.models import load_model
from modules.custom_layers import CustomMultiHeadAttention  # ここで CustomMultiHeadAttention をインポート

# モデルパス
model_path = 'models/gru_20240608_145935_127m/best_model.h5'

# ディレクトリ設定
dirs = {
    "original": "./components/dataset/original",
    "tokenize": "./components/dataset/tokenize",
    "preprocessed": "./components/dataset/preprocessed",
}

# テストデータの保存パス
test_data_path = os.path.join(dirs["original"], "test_bracket_dataset.json")

# モデルのロード
model = load_model(model_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})

# モデルタイプの取得
model_type = get_model_type_from_model(model)

# テストデータのロード
test_data = load_dataset(test_data_path)

def evaluate_in_chunks(model_path, test_data, model_type, num_chunks=20, chunk_size=5):
    chunk_results = []
    for _ in range(num_chunks):
        chunk_accuracy = 0
        for _ in range(chunk_size):
            accuracy = evaluate_model_instance(model_path, test_data, model_type)
            chunk_accuracy += accuracy
        chunk_results.append(chunk_accuracy / chunk_size)
    return np.mean(chunk_results) * 100  # 100点満点で評価

def perform_multiple_trials(model_path, test_data, model_type, num_trials=10):
    trial_results = []
    for _ in range(num_trials):
        trial_accuracy = evaluate_in_chunks(model_path, test_data, model_type)
        trial_results.append(trial_accuracy)
    return np.mean(trial_results)

# 評価回数とサンプル数の設定
num_chunks = 20
chunk_size = 5
num_trials = 10

# モデルの評価
average_accuracy = perform_multiple_trials(model_path, test_data, model_type, num_trials)
print(f"モデルの平均精度: {average_accuracy:.2f}%")
