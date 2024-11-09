import os
import sys
import json
import numpy as np
from datetime import datetime
import pytz
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from modules.data_utils import load_dataset, tokens
from modules.custom_layers import CustomMultiHeadAttention

# 日本時間のタイムゾーンを設定
japan_timezone = pytz.timezone("Asia/Tokyo")

# 環境変数設定
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# データセットの保存先ディレクトリ
dataset_base_dir = "./datasets/"
model_save_path = "./models/"

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

def generate_datasets(base_dir, num_samples):
    # 必要に応じてデータセットを生成するコードをここに追加
    pass

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

def plot_training_history(history, test_metrics, save_path):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.plot(test_metrics['test_loss'], label='Test Loss per Epoch')
    plt.plot(test_metrics['test_accuracy'], label='Test Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training and Test Metrics per Epoch')
    plt.savefig(save_path)
    plt.close()

def evaluate_main(model_path, epoch):
    # モデル評価処理を実行し、エポック情報を出力する
    print(f"Evaluating model at {model_path} after epoch {epoch}")

def main():
    existing_model_dir = select_existing_model()
    if not existing_model_dir:
        return

    model, training_info = load_model_and_info(existing_model_dir)

    training_params = {
        "learning_rate": training_info["learning_rate"],
        "batch_size": training_info["batch_size"]
    }

    additional_epochs = int(input("追加で学習させるエポック数を入力してください: "))

    num_samples = training_info["dataset_size"]
    num_files = training_info["num_files"]

    # データセットの生成または読み込み
    dataset_path = dataset_base_dir
    generate_datasets(dataset_path, num_samples)

    # データの読み込み
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

    # 各エポックごとのテスト評価結果を格納する辞書
    test_metrics = {
        "test_loss": [],
        "test_accuracy": []
    }

    # モデルのトレーニングと各エポックでの評価
    for epoch in range(additional_epochs):
        print(f"Epoch {epoch + 1}/{additional_epochs}")

        history = model.fit(
            input_sequences,
            target_tokens,
            epochs=1,
            batch_size=training_params["batch_size"],
            validation_split=0.1
        )

        # 1エポックごとにモデルの評価
        test_loss, test_accuracy = model.evaluate(input_sequences, target_tokens, verbose=0)
        print(f"Test Loss after Epoch {epoch + 1}: {test_loss}")
        print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy}")

        # テスト評価結果を記録
        test_metrics["test_loss"].append(test_loss)
        test_metrics["test_accuracy"].append(test_accuracy)

        # evaluate_main関数を呼び出し、エポック数とモデルパスを渡す
        evaluate_main(os.path.join(model_save_path, existing_model_dir, "best_model.h5"), epoch + 1)

    # 学習履歴のプロット
    plot_save_path = os.path.join(model_save_path, existing_model_dir, "training_history.png")
    plot_training_history(history, test_metrics, plot_save_path)

    # training_info.jsonの更新
    if "training_sessions" not in training_info:
        training_info["training_sessions"] = []

    end_time = datetime.now(japan_timezone)
    training_duration = sum(history.history.get('duration', [0])) / 60  # 分に変換

    # 各エポックの評価結果を保存
    new_training_session = {
        "training_date": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": additional_epochs,
        "batch_size": training_params["batch_size"],
        "learning_rate": training_params["learning_rate"],
        "training_duration_minutes": training_duration,
        "final_loss": history.history["loss"][-1],
        "final_accuracy": history.history["accuracy"][-1],
        "test_loss_per_epoch": test_metrics["test_loss"],
        "test_accuracy_per_epoch": test_metrics["test_accuracy"]
    }

    training_info["training_sessions"].append(new_training_session)

    # 合計エポック数の更新
    previous_total_epochs = training_info.get("total_epochs", training_info["epochs"])
    training_info["total_epochs"] = previous_total_epochs + additional_epochs

    # training_info.jsonの保存
    with open(os.path.join(model_save_path, existing_model_dir, "training_info.json"), "w") as info_file:
        json.dump(training_info, info_file, indent=4, ensure_ascii=False)

    # モデルの保存
    model.save(os.path.join(model_save_path, existing_model_dir, "best_model.h5"))

    print("追加のトレーニングが完了しました。")
    print(f"最終的な損失: {new_training_session['final_loss']}")
    print(f"最終的な精度: {new_training_session['final_accuracy']}")
    print("各エポックごとのテスト損失と精度:", test_metrics)

if __name__ == "__main__":
    main()
