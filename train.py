import os
import sys
import json
import numpy as np
from datetime import datetime
import pytz
from modules.data_utils import load_dataset, prepare_sequences, tokens
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.training_utils import train_model_single, plot_training_history
from modules.custom_layers import CustomMultiHeadAttention
from tensorflow.keras.preprocessing.sequence import pad_sequences
import optuna_data_generator  # このインポートを追加

# 日本時間のタイムゾーンを設定
japan_timezone = pytz.timezone("Asia/Tokyo")

# 環境変数設定
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"
# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# データセットの保存先ディレクトリ
dataset_base_dir = "./datasets/"
model_save_path = "./models/"

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model,
    "lstm": define_lstm_model,
    "bert": define_bert_model,
    "gpt": define_gpt_model
}

SHORTCUTS = {
    "gru": "gru",
    "tra": "transformer",
    "lstm": "lstm",
    "ber": "bert",
    "gpt": "gpt"
}

TRAINING_MODES = {
    "1min": {"epochs": 1, "batch_size": 128, "num_files": 5, "learning_rate": 0.01},
    "10min": {"epochs": 3, "batch_size": 256, "num_files": 10, "learning_rate": 0.01},
    "1hour": {"epochs": 7, "batch_size": 512, "num_files": 50, "learning_rate": 0.001},
    "6hours": {"epochs": 20, "batch_size": 1024, "num_files": 300, "learning_rate": 0.001},
    "12hours": {"epochs": 40, "batch_size": 1024, "num_files": 600, "learning_rate": 0.001},
    "24hours": {"epochs": 80, "batch_size": 1024, "num_files": 1200, "learning_rate": 0.0005},
    "2days": {"epochs": 160, "batch_size": 1024, "num_files": 2400, "learning_rate": 0.0005},
    "4days": {"epochs": 320, "batch_size": 1024, "num_files": 4800, "learning_rate": 0.0005},
    "op": {  
        "batch_size": 64, # アプローチ2 :モデルの複雑度を上げ、データセットを大幅に増やす
        "learning_rate": 0.0001,
        "embedding_dim": 128,
        "gru_units": 256,
        "dropout_rate": 0.2,
        "recurrent_dropout_rate": 0.2,
        "epochs": 1000  # 早期停止を有効にする

    }
}

def select_mode():
    mode = input("Select a mode from: " + ", ".join(TRAINING_MODES.keys()) + "\n")
    while mode not in TRAINING_MODES:
        print(f"Invalid mode. Please select a mode from: {', '.join(TRAINING_MODES.keys())}")
        mode = input()
    return TRAINING_MODES[mode]["epochs"], TRAINING_MODES[mode]["batch_size"], TRAINING_MODES[mode]["num_files"], TRAINING_MODES[mode]["learning_rate"]

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []

    for i in range(1, len(encoded_tokens)):
        input_seq = encoded_tokens[:i]
        target_seq = encoded_tokens[i]
        input_sequences.append([int(token) for token in input_seq])
        target_tokens.append(int(target_seq))

    input_sequences = pad_sequences(input_sequences, maxlen=seq_length, padding='post', value=0)
    target_tokens = pad_sequences([target_tokens], maxlen=len(input_sequences), padding='post', value=0)[0]

    return input_sequences, target_tokens

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def select_mode_and_architecture():
    modes = list(TRAINING_MODES.keys())
    architectures = list(SHORTCUTS.keys())
    choices = [f"{arch} {mode}" for arch in architectures for mode in modes]

    print("以下のモードとアーキテクチャから選んでください。選択肢は英語のまま入力してください：\n")
    
    print("1. GRU (Gated Recurrent Unit)")
    for mode in modes:
        print(f"    - {mode}: gru {mode}")

    print("\n2. Transformer")
    for mode in modes:
        print(f"    - {mode}: tra {mode}")

    print("\n3. LSTM")
    for mode in modes:
        print(f"    - {mode}: lstm {mode}")

    print("\n4. BERT")
    for mode in modes:
        print(f"    - {mode}: ber {mode}")

    print("\n5. GPT")
    for mode in modes:
        print(f"    - {mode}: gpt {mode}")

    choice = input("\nあなたの選択: ")
    
    while choice not in choices:
        print(f"\n無効な選択です。以下の選択肢から選んでください：\n")
        
        print("1. GRU (Gated Recurrent Unit)")
        for mode in modes:
            print(f"    - {mode}: gru {mode}")

        print("\n2. Transformer")
        for mode in modes:
            print(f"    - {mode}: tra {mode}")

        print("\n3. LSTM")
        for mode in modes:
            print(f"    - {mode}: lstm {mode}")

        print("\n4. BERT")
        for mode in modes:
            print(f"    - {mode}: ber {mode}")

        print("\n5. GPT")
        for mode in modes:
            print(f"    - {mode}: gpt {mode}")
        
        choice = input("\nあなたの選択: ")
    
    arch, mode = choice.split()
    architecture = SHORTCUTS[arch]
    return MODEL_ARCHITECTURES[architecture], TRAINING_MODES[mode], architecture

def generate_datasets(base_dir, num_samples):
    dataset_sizes = [100, 300, 500, 800, 1000, 3000, 5000, 10000]

    # データセットサイズのリストにない場合でも生成可能にする
    if num_samples not in dataset_sizes:
        print(f"Number of samples {num_samples} is not in the predefined list, but will be generated.")
    
    num_datasets = 1  # 必要に応じて変更可能

    # 既存の生成関数を呼び出し、正しいパスに生成されるようにする
    optuna_data_generator.create_datasets(base_dir, num_datasets, num_samples)
    print(f"Generated {num_samples} samples dataset in {base_dir}")  # デバッグ用に追加



def main():
    model_architecture_func, training_mode, architecture = select_mode_and_architecture()
    num_samples = int(input("Enter the number of samples for the dataset (100, 300, 500, 800, 1000, 3000, 5000, 10000): ").strip())
    
    # 日本時間のタイムゾーンを設定
    start_time = datetime.now(japan_timezone)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # 一時フォルダ名作成
    temp_save_dir = os.path.join(model_save_path, f"{timestamp}_{architecture}_temp")
    os.makedirs(temp_save_dir, exist_ok=True)
    
    # データセットの生成
    dataset_path = temp_save_dir
    generate_datasets(dataset_path, num_samples)
    
    max_seq_length = 30  # 最大シーケンス長を設定

    vocab_set = set(tokens)

    all_input_sequences = []
    all_target_tokens = []

    num_files = training_mode.get("num_files", 10)  # デフォルト値を設定

    # データセットの読み込み位置を修正
    preprocessed_path = os.path.join(dataset_path, "dataset", "preprocessed")
    print(f"Preprocessed path: {preprocessed_path}")  # デバッグ用に追加

    for dirpath, dirnames, filenames in os.walk(preprocessed_path):
        print(f"Checking directory: {dirpath}")  # デバッグ用に追加
        for file in filenames[:num_files]:  # num_filesに基づいてファイル数を制限
            file_path = os.path.join(dirpath, file)
            print(f"Processing file: {file_path}")  # デバッグ用に追加
            encoded_tokens_list = load_dataset(file_path)
            print(f"Loaded data from {file_path}: {encoded_tokens_list[:2]}...")  # デバッグ用に追加（最初の2つのエントリのみ表示）
            for encoded_tokens in encoded_tokens_list:
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=max_seq_length)
                all_input_sequences.extend(input_sequences)
                all_target_tokens.extend(target_tokens)

    if not all_input_sequences or not all_target_tokens:
        print("No data for training.")
        return

    vocab_size = len(vocab_set)

    # モデルアーキテクチャ関数に渡すためのパラメータを抽出
    model_params = {k: v for k, v in training_mode.items() if k not in ["epochs", "batch_size", "num_files", "learning_rate"]}
    learning_rate = training_mode["learning_rate"]
    model = model_architecture_func(max_seq_length, vocab_size + 1, learning_rate, **model_params)

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    training_info_path = os.path.join(temp_save_dir, "training_info.json")

    with open(training_info_path, "w") as info_file:
        json.dump({"training_start_time": timestamp}, info_file)

    model_path = os.path.join(temp_save_dir, "best_model.h5")
    plot_path = os.path.join(temp_save_dir, "training_history.png")

    # モデルの学習
    history, dataset_size = train_model_single(
        model,
        all_input_sequences,
        all_target_tokens,
        epochs=training_mode["epochs"],
        batch_size=training_mode["batch_size"],
        model_path=model_path,
        num_files=num_files,
        learning_rate=learning_rate,
        architecture=architecture,
        model_architecture_func=model_architecture_func,
        **model_params  # 他のパラメータを展開して渡す
    )

    if history:
        plot_training_history(
            history,
            save_path=plot_path,
            epochs=training_mode["epochs"],
            batch_size=training_mode["batch_size"],
            learning_rate=learning_rate,
            num_files=num_files,
            dataset_size=dataset_size
        )

    end_time = datetime.now(japan_timezone)
    training_duration = (end_time - start_time).total_seconds() / 60  # 分単位に変換

    # 実際の学習時間をフォルダ名に反映
    final_save_dir = os.path.join(model_save_path, f"{architecture}_{timestamp}_{int(training_duration)}m")
    os.rename(temp_save_dir, final_save_dir)

    # パスを更新
    training_info_path = os.path.join(final_save_dir, "training_info.json")
    model_path = os.path.join(final_save_dir, "best_model.h5")
    plot_path = os.path.join(final_save_dir, "training_history.png")

    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB単位に変換
    model_params = model.count_params()

    metadata = {
        "training_duration_minutes": training_duration,
        "epochs": training_mode["epochs"],
        "batch_size": training_mode["batch_size"],
        "num_files": num_files,
        "learning_rate": learning_rate,
        "dataset_size": num_samples,
        "model_size_MB": model_size,
        "model_params": model_params,
        "model_architecture": model_architecture_func.__name__,
        "training_end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(training_info_path, "r") as info_file:
        training_info = json.load(info_file)
    training_info.update(metadata)

    with open(training_info_path, "w") as info_file:
        json.dump(training_info, info_file, indent=4)

    print(f"Training finished.")
    print(f"Training duration: {training_duration:.2f} minutes")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Model parameters: {model_params}")

if __name__ == "__main__":
    main()
