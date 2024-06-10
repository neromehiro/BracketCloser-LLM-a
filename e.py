import os
import sys
from modules.evaluate import main

# モデルのパス
model_path = "models/20240610_150718_gru_temp/best_model.h5"

# テストデータのサンプル数
num_test_samples = 500  # デフォルトは100

# 評価関数を呼び出して評価を実行
if os.path.exists(model_path):
    average_accuracy = main(model_path, num_test_samples)
    print(f"Evaluation completed. Average accuracy: {average_accuracy:.2f}%")
else:
    print(f"Model file does not exist at path: {model_path}")
