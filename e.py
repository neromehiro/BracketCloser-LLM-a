import os
import sys
from modules.evaluate import main

# モデルのパス
model_path = "good_models/best_model_epoch_200.h5"

# テストデータのサンプル数
num_test_samples = 500  # デフォルトは100

# 評価関数を呼び出して評価を実行
if os.path.exists(model_path):
    complete_accuracy, partial_accuracy = main(model_path, num_test_samples)
    print(f"Evaluation completed.")
    print(f"Complete accuracy: {complete_accuracy:.2f}%")
    print(f"Partial accuracy: {partial_accuracy:.2f}%")
else:
    print(f"Model file does not exist at path: {model_path}")
