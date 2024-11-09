
# train-continue.py
import os
import sys
import json
import numpy as np
from datetime import datetime
import pytz
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import optuna_data_generator  # データ生成用のインポート
from modules.data_utils import load_dataset, tokens
from modules.custom_layers import CustomMultiHeadAttention

# 必要なモジュールをインポート
from modules.evaluate import main as evaluate_main
from modules.training_utils import plot_training_history

# 日本時間のタイムゾーンを設定
japan_timezone = pytz.timezone("Asia/Tokyo")

# 環境変数設定
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# データセットの保存先ディレクトリ
dataset_base_dir = "./datasets/"
model_save_path = "./models/"

def generate_datasets(base_dir, num_samples):
    dataset_sizes = [100, 300, 500, 800, 1000, 3000, 5000, 10000]

    if num_samples not in dataset_sizes:
        print(f"Number of samples {num_samples} is not in the predefined list, but will be generated.")
    
    num_datasets = 1  # 必要に応じて変更可能

    optuna_data_generator.create_datasets(base_dir, num_datasets, num_samples)
    print(f"Generated {num_samples} samples dataset in {base_dir}")

def list_existing_models():
    models = sorted([d for d in os.listdir(model_save_path) if os.path.isdir(os.path.join(model_save_path, d))])
    # 最新の9個のモデルに制限
    return models[-9:] if len(models) > 9 else models


def select_existing_model():
    models = list_existing_models()
    if not models:
        print("No existing models found.")
        return None
    print("Select a model to continue training:")
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")
    choice = int(input("Enter the number of the model: ")) - 1
    return models[choice]

def load_model_and_info(model_dir):
    model_path = os.path.join(model_save_path, model_dir, "best_model.h5")
    model = load_model(model_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})
    info_path = os.path.join(model_save_path, model_dir, "training_info.json")
    with open(info_path, "r") as info_file:
        training_info = json.load(info_file)
    compile_model(model, training_info["learning_rate"])
    return model, training_info

def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []

    for i in range(1, len(encoded_tokens)):
        input_seq = encoded_tokens[:i]
        target_seq = encoded_tokens[i]
        input_sequences.append([int(token) for token in input_seq])
        target_tokens.append(int(target_seq))

    input_sequences = pad_sequences(input_sequences, maxlen=seq_length, padding='post', value=0)
    target_tokens = np.array(target_tokens)
    return input_sequences, target_tokens

def main():
    existing_model_dir = select_existing_model()
    if not existing_model_dir:
        return
    model, training_info = load_model_and_info(existing_model_dir)

    # モデルの保存パスを定義
    model_path = os.path.join(model_save_path, existing_model_dir, "best_model.h5")

    training_params = {
        "learning_rate": training_info["learning_rate"],
        "batch_size": training_info["batch_size"],
        "epochs": training_info["epochs"],
        "num_files": training_info["num_files"],
        "dataset_size": training_info["dataset_size"],
        "architecture": training_info.get("model_architecture", "unknown"),
    }

    additional_epochs = int(input("追加で学習させるエポック数を入力してください: "))
    total_epochs = training_info.get("total_epochs", training_info["epochs"]) + additional_epochs

    # プロットデータの読み込み
    plot_data_path = os.path.join(model_save_path, existing_model_dir, "plot_data.json")
    if os.path.exists(plot_data_path):
        with open(plot_data_path, "r") as plot_file:
            previous_plot_data = json.load(plot_file)
        all_epoch_history = [
            {'loss': loss, 'val_loss': val_loss, 'accuracy': accuracy, 'val_accuracy': val_accuracy}
            for loss, val_loss, accuracy, val_accuracy in zip(
                previous_plot_data.get('loss', []),
                previous_plot_data.get('val_loss', []),
                previous_plot_data.get('accuracy', []),
                previous_plot_data.get('val_accuracy', [])
            )
        ]
        complete_accuracies = previous_plot_data.get('complete_accuracy', [])
        partial_accuracies = previous_plot_data.get('partial_accuracy', [])
    else:
        all_epoch_history = []
        complete_accuracies = []
        partial_accuracies = []

    start_time = datetime.now(japan_timezone)
    dataset_path = os.path.join(model_save_path, existing_model_dir)
    generate_datasets(dataset_path, training_info["dataset_size"])

    # トレーニングデータの準備
    preprocessed_path = os.path.join(dataset_path, "dataset", "preprocessed")
    input_sequences, target_tokens = [], []
    for dirpath, dirnames, filenames in os.walk(preprocessed_path):
        for file in filenames[:training_info["num_files"]]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                seq_input, seq_target = prepare_sequences(encoded_tokens, seq_length=30)
                input_sequences.append(seq_input)
                target_tokens.append(seq_target)

    input_sequences = np.concatenate(input_sequences, axis=0)
    target_tokens = np.concatenate(target_tokens, axis=0)

    for epoch in range(additional_epochs):
        current_epoch = training_info.get("total_epochs", training_info["epochs"]) + epoch + 1
        print(f"Epoch {current_epoch}/{total_epochs}")

        history = model.fit(
            input_sequences,
            target_tokens,
            epochs=1,
            batch_size=training_params["batch_size"],
            validation_split=0.1
        )

        # エポックの履歴データを辞書形式で追加
        epoch_data = {
            'loss': history.history['loss'][0],
            'accuracy': history.history['accuracy'][0],
            'val_loss': history.history.get('val_loss', [None])[0],
            'val_accuracy': history.history.get('val_accuracy', [None])[0]
        }
        all_epoch_history.append(epoch_data)

        # 評価を実行し、正答率を取得
        complete_accuracy, partial_accuracy = evaluate_main(model_path, current_epoch)
        complete_accuracies.append(complete_accuracy)
        partial_accuracies.append(partial_accuracy)

        # プロットデータを更新
        plot_data = {
            'loss': [epoch['loss'] for epoch in all_epoch_history],
            'val_loss': [epoch['val_loss'] for epoch in all_epoch_history],
            'accuracy': [epoch['accuracy'] for epoch in all_epoch_history],
            'val_accuracy': [epoch['val_accuracy'] for epoch in all_epoch_history],
            'complete_accuracy': complete_accuracies,
            'partial_accuracy': partial_accuracies
        }

        # エポック終了後にリアルタイムでプロットとplot_data.jsonを保存
        plot_training_history(
            plot_data,
            save_path=os.path.join(model_save_path, existing_model_dir, "training_history.png"),
            epochs=total_epochs,
            batch_size=training_params["batch_size"],
            learning_rate=training_params["learning_rate"],
            num_files=training_params["num_files"],
            dataset_size=training_params["dataset_size"],
            avg_complete_accuracy=sum(complete_accuracies) / len(complete_accuracies),
            avg_partial_accuracy=sum(partial_accuracies) / len(partial_accuracies),
            initial_metadata=training_info
        )

        # plot_data.jsonを更新
        with open(plot_data_path, "w") as plot_file:
            json.dump(plot_data, plot_file, indent=4)

    print("追加のトレーニングが完了しました。")

if __name__ == "__main__":
    main()

