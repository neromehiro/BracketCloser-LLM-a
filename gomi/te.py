
import os
from modules.data_generator import generate_test_data, preprocess_and_save_dataset

def main():
    # テストデータのサンプル数
    num_test_samples = 100
    # テストデータのファイル名
    filename = "test_bracket_dataset.json"
    
    # テストデータの生成
    test_dataset = generate_test_data(num_test_samples)

    # テストデータの前処理と保存
    preprocess_and_save_dataset(test_dataset, filename, max_seq_length=30)

    # 保存されたファイルのパス
    print("テストデータセットが保存された場所:", os.path.join("./components/dataset/original", filename))

if __name__ == "__main__":
    main()