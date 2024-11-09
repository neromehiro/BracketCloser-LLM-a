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
    return models

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

    training_params = {
        "learning_rate": training_info["learning_rate"],
        "batch_size": training_info["batch_size"],
        "epochs": training_info["epochs"],  # 追加
        "num_files": training_info["num_files"],  # 追加
        "dataset_size": training_info["dataset_size"],  # 追加
        "architecture": training_info.get("model_architecture", "unknown"),  # 追加
    }

    additional_epochs = int(input("追加で学習させるエポック数を入力してください: "))

    num_samples = training_info["dataset_size"]
    num_files = training_info["num_files"]

    # 完全な正答率と部分的な正答率を格納するリストを追加
    complete_accuracies = []
    partial_accuracies = []

    # 日本時間のタイムゾーンを設定
    start_time = datetime.now(japan_timezone)

    dataset_path = os.path.join(model_save_path, existing_model_dir)
    generate_datasets(dataset_path, num_samples)

    preprocessed_path = os.path.join(dataset_path, "dataset", "preprocessed")

    input_sequences = []
    target_tokens = []
    for dirpath, dirnames, filenames in os.walk(preprocessed_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                seq_input, seq_target = prepare_sequences(encoded_tokens, seq_length=30)
                input_sequences.append(seq_input)
                target_tokens.append(seq_target)

    if not input_sequences or not target_tokens:
        print("No data for training.")
        return

    input_sequences = np.concatenate(input_sequences, axis=0)
    target_tokens = np.concatenate(target_tokens, axis=0)

    all_epoch_history = []

    # モデルの保存パスを定義
    model_path = os.path.join(model_save_path, existing_model_dir, "best_model.h5")

    for epoch in range(additional_epochs):
        print(f"Epoch {epoch + 1}/{additional_epochs}")

        history = model.fit(
            input_sequences,
            target_tokens,
            epochs=1,
            batch_size=training_params["batch_size"],
            validation_split=0.1
        )

        # エポックの履歴を保存
        epoch_data = {
            'loss': history.history['loss'][0],
            'accuracy': history.history['accuracy'][0],
            'val_loss': history.history.get('val_loss', [None])[0],
            'val_accuracy': history.history.get('val_accuracy', [None])[0]
        }
        all_epoch_history.append(epoch_data)

        # モデルを保存
        model.save(model_path)

        # 評価を実行し、正答率を取得
        if os.path.exists(model_path):
            complete_accuracy, partial_accuracy = evaluate_main(model_path, epoch + 1)
            complete_accuracies.append(complete_accuracy)
            partial_accuracies.append(partial_accuracy)
            print(f"Evaluation completed.")
            print(f"Complete accuracy: {complete_accuracy:.2f}%")
            print(f"Partial accuracy: {partial_accuracy:.2f}%")
        else:
            complete_accuracies.append(None)
            partial_accuracies.append(None)
            print(f"Model file does not exist at path: {model_path}")

        # プロットの更新
        plot_data = {
            'loss': [epoch['loss'] for epoch in all_epoch_history],
            'val_loss': [epoch['val_loss'] for epoch in all_epoch_history],
            'accuracy': [epoch['accuracy'] for epoch in all_epoch_history],
            'val_accuracy': [epoch['val_accuracy'] for epoch in all_epoch_history],
            'complete_accuracy': complete_accuracies,
            'partial_accuracy': partial_accuracies
        }

        plot_save_path = os.path.join(model_save_path, existing_model_dir, "training_history.png")
        avg_complete_accuracy = sum(acc for acc in complete_accuracies if acc is not None) / len(complete_accuracies)
        avg_partial_accuracy = sum(acc for acc in partial_accuracies if acc is not None) / len(partial_accuracies)

        # initial_metadataを取得または設定
        initial_metadata = {
            "training_duration_minutes": training_info.get("training_duration_minutes", 0),
            "epochs": training_params["epochs"],
            "batch_size": training_params["batch_size"],
            "num_files": training_params["num_files"],
            "learning_rate": training_params["learning_rate"],
            "dataset_size": training_params["dataset_size"],
            "model_size_MB": training_info.get("model_size_MB", 0),
            "model_params": model.count_params(),
            "model_architecture": training_params["architecture"],
            "training_start_time": training_info.get("training_start_time", start_time.strftime("%Y-%m-%d %H:%M:%S")),
            "training_end_time": ""
        }

        plot_training_history(
            plot_data,
            save_path=plot_save_path,
            epochs=training_params["epochs"] + additional_epochs,
            batch_size=training_params["batch_size"],
            learning_rate=training_params["learning_rate"],
            num_files=training_params["num_files"],
            dataset_size=training_params["dataset_size"],
            avg_complete_accuracy=avg_complete_accuracy,
            avg_partial_accuracy=avg_partial_accuracy,
            initial_metadata=initial_metadata
        )

    end_time = datetime.now(japan_timezone)
    training_duration = (end_time - start_time).total_seconds() / 60  # 分単位に変換

    # training_info.jsonの更新
    if "training_sessions" not in training_info:
        training_info["training_sessions"] = []

    new_training_session = {
        "training_date": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": additional_epochs,
        "batch_size": training_params["batch_size"],
        "learning_rate": training_params["learning_rate"],
        "training_duration_minutes": training_duration,
        "final_loss": all_epoch_history[-1]['loss'],
        "final_accuracy": all_epoch_history[-1]['accuracy'],
        "complete_accuracies": complete_accuracies,
        "partial_accuracies": partial_accuracies
    }

    training_info["training_sessions"].append(new_training_session)

    previous_total_epochs = training_info.get("total_epochs", training_info["epochs"])
    training_info["total_epochs"] = previous_total_epochs + additional_epochs

    training_info["training_end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    training_info["training_duration_minutes"] += training_duration

    with open(os.path.join(model_save_path, existing_model_dir, "training_info.json"), "w") as info_file:
        json.dump(training_info, info_file, indent=4, ensure_ascii=False)

    print("追加のトレーニングが完了しました。")
    print(f"最終的な損失: {new_training_session['final_loss']}")
    print(f"最終的な精度: {new_training_session['final_accuracy']}")
    print("各エポックごとの完全な正答率と部分的な正答率:")
    for i, (c_acc, p_acc) in enumerate(zip(complete_accuracies, partial_accuracies), 1):
        print(f"Epoch {i}: Complete Accuracy = {c_acc:.2f}%, Partial Accuracy = {p_acc:.2f}%")

if __name__ == "__main__":
    main()
