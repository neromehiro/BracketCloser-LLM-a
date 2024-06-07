import sys
import os

# モジュールのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import data_generator

# テストデータのサンプル数
num_test_samples = 100

# テストデータの生成
test_dataset = data_generator.generate_test_data(num_test_samples)

# テストデータの前処理と保存
data_generator.preprocess_and_save_dataset(test_dataset, "test_bracket_dataset.json")
print("テストデータセットが保存された場所:", data_generator.dirs["original"] + "/test_bracket_dataset.json")
