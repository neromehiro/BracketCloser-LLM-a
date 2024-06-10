import os
import sys
from modules.evaluate import main

# モデルのパス
model_path = "models/gru_20240608_145935_127m/best_model.h5"

# 評価関数を呼び出して評価を実行
if os.path.exists(model_path):
    average_accuracy = main(model_path)
    print(f"Evaluation completed. Average accuracy: {average_accuracy:.2f}%")
else:
    print(f"Model file does not exist at path: {model_path}")
