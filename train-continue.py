import os
import sys
import json
import numpy as np
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from modules.data_utils import load_dataset, tokens
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.training_utils import train_model_continue, plot_training_history
from modules.custom_layers import CustomMultiHeadAttention
from tensorflow.keras.preprocessing.sequence import pad_sequences
import optuna_data_generator  # このインポートを追加
import tensorflow as tf

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

# 新しいパラメーターの設定
NEW_PARAMETERS = {
    "learning_rate": 0.0031535421928088523,
    "batch_size": 205,
    "regularizer_type": "l2",
    "regularizer_value": 1.1973670625410778e-05,
    "embedding_dim": 127,
    "gru_units": 90,
    "dropout_rate": 0.10421276428973633,
    "recurrent_dropout_rate": 0.21222279862119903,
    "epochs": 10,
    "num_files": 10  # 適切な数値を追加してください
}

def list_existing_models():
    models = sorted([d for d in os.listdir(model_save_path) if os.path.isdir(os.path.join(model_save_path, d))])
    return models[:9]

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
    
    compile_model(model, training_info["learning_rate"])  # モデルのコンパイルを追加
    return model, training_info



def generate_datasets(base_dir, num_samples):
    dataset_sizes = [100, 300, 500, 800, 1000, 3000, 5000, 10000]

    # データセットサイズのリストにない場合でも生成可能にする
    if num_samples not in dataset_sizes:
        print(f"Number of samples {num_samples} is not in the predefined list, but will be generated.")
    
    num_datasets = 1  # 必要に応じて変更可能

    # 既存の生成関数を呼び出し、正しいパスに生成されるようにする
    optuna_data_generator.create_datasets(base_dir, num_datasets, num_samples)
    print(f"Generated {num_samples} samples dataset in {base_dir}")  # デバッグ用に追加

def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=[]  # 追加
    )

    
MODEL_ARCHITECTURES = {
    "define_gru_model": define_gru_model,
    "define_transformer_model": define_transformer_model,
    "define_lstm_model": define_lstm_model,
    "define_bert_model": define_bert_model,
    "define_gpt_model": define_gpt_model
}

ADDITIONAL_TRAINING_TYPES = {
    "lora": "Low-Rank Adaptation",
    "fine_tuning": "Fine Tuning",
    "retnet": "RetNet"
}

def select_training_type():
    print("Select an additional training type:")
    for key, value in ADDITIONAL_TRAINING_TYPES.items():
        print(f"{key}: {value}")
    choice = input("Enter the training type: ").strip().lower()
    if choice not in ADDITIONAL_TRAINING_TYPES:
        print("Invalid choice. Please select a valid training type.")
        return select_training_type()
    return choice



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

    # print(f"Prepared sequences, input: {input_sequences.shape}, target: {target_tokens.shape}")  # デバッグ出力
    return input_sequences, target_tokens

def main():
    existing_model_dir = select_existing_model()
    if not existing_model_dir:
        return

    training_type = select_training_type()  # 追加学習の種類を選択
    model, training_info = load_model_and_info(existing_model_dir)
    compile_model(model, training_info["learning_rate"])  # 追加

    use_new_parameters = input("Use new parameters for training? (yes/no): ").strip().lower()
    if use_new_parameters == "yes":
        training_params = NEW_PARAMETERS
    else:
        continue_training = input("Continue training with existing parameters? (yes/no): ").strip().lower()
        if continue_training == "no":
            training_params = {
                "learning_rate": float(input("Enter new learning rate: ")),
                "epochs": int(input("Enter new number of epochs: ")),
                "batch_size": int(input("Enter new batch size: "))
            }
        else:
            training_params = {
                "learning_rate": training_info["learning_rate"],
                "epochs": training_info["epochs"],
                "batch_size": training_info["batch_size"]
            }

    num_samples = int(input("Enter the number of samples for the dataset (100, 300, 500, 800, 1000, 3000, 5000, 10000): ").strip())

    # 日本時間のタイムゾーンを設定
    start_time = datetime.now(japan_timezone)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    temp_save_dir = os.path.join(model_save_path, f"{timestamp}_temp")
    os.makedirs(temp_save_dir, exist_ok=True)

    max_seq_length = 30  # 最大シーケンス長を設定

    vocab_set = set(tokens)
    vocab_size = len(vocab_set)

    all_input_sequences = []
    all_target_tokens = []

    full_history = []

    model_path = os.path.join(temp_save_dir, "best_model.h5")

    model_architecture_func_name = training_info["model_architecture"]
    model_architecture_func = MODEL_ARCHITECTURES.get(model_architecture_func_name)
    if not model_architecture_func:
        print(f"Invalid model architecture: {model_architecture_func_name}")
        return

    for epoch in range(training_params["epochs"]):
        print(f"Starting epoch {epoch + 1}/{training_params['epochs']}")

        dataset_path = temp_save_dir
        generate_datasets(dataset_path, num_samples)

        preprocessed_path = os.path.join(dataset_path, "dataset", "preprocessed")
        print(f"Preprocessed path: {preprocessed_path}")

        epoch_input_sequences = []
        epoch_target_tokens = []

        for dirpath, dirnames, filenames in os.walk(preprocessed_path):
            for file in filenames[:training_info["num_files"]]:
                file_path = os.path.join(dirpath, file)
                print(f"Loading dataset from: {file_path}")
                encoded_tokens_list = load_dataset(file_path)

                for encoded_tokens in encoded_tokens_list:
                    input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=max_seq_length)
                    epoch_input_sequences.append(input_sequences)
                    epoch_target_tokens.append(target_tokens)

        if not epoch_input_sequences or not epoch_target_tokens:
            print("No data for training.")
            continue

        epoch_input_sequences = np.concatenate(epoch_input_sequences, axis=0)
        epoch_target_tokens = np.concatenate(epoch_target_tokens, axis=0)

        print(f"Epoch {epoch + 1} data shapes: {epoch_input_sequences.shape}, {epoch_target_tokens.shape}")

        all_input_sequences.append(epoch_input_sequences)
        all_target_tokens.append(epoch_target_tokens)

        try:
            if training_type == "lora":
                history, dataset_size = train_model_lora(
                    model,
                    epoch_input_sequences,
                    epoch_target_tokens,
                    epochs=training_params["epochs"],  # 指定したエポック数でトレーニング
                    batch_size=training_params["batch_size"],
                    model_path=model_path,
                    num_files=training_info["num_files"],
                    learning_rate=training_params["learning_rate"],
                    architecture=training_info["model_architecture"],
                    model_architecture_func=model_architecture_func
                )
            elif training_type == "fine_tuning":
                history, dataset_size = train_model_fine_tuning(
                    model,
                    epoch_input_sequences,
                    epoch_target_tokens,
                    epochs=training_params["epochs"],  # 指定したエポック数でトレーニング
                    batch_size=training_params["batch_size"],
                    model_path=model_path,
                    num_files=training_info["num_files"],
                    learning_rate=training_params["learning_rate"],
                    architecture=training_info["model_architecture"],
                    model_architecture_func=model_architecture_func
                )
            elif training_type == "retnet":
                history, dataset_size = train_model_retnet(
                    model,
                    epoch_input_sequences,
                    epoch_target_tokens,
                    epochs=training_params["epochs"],  # 指定したエポック数でトレーニング
                    batch_size=training_params["batch_size"],
                    model_path=model_path,
                    num_files=training_info["num_files"],
                    learning_rate=training_params["learning_rate"],
                    architecture=training_info["model_architecture"],
                    model_architecture_func=model_architecture_func
                )

            if history:
                if isinstance(history, dict):
                    for i in range(len(history["loss"])):
                        epoch_data = {
                            'loss': history["loss"][i],
                            'val_loss': history["val_loss"][i] if i < len(history["val_loss"]) else None,
                            'accuracy': history["accuracy"][i] if i < len(history["accuracy"]) else None,
                            'val_accuracy': history["val_accuracy"][i] if i < len(history["val_accuracy"]) else None
                        }
                        print(f"Epoch log: {epoch_data}")
                        full_history.append(epoch_data)
                else:
                    for epoch_log in history:
                        if isinstance(epoch_log, dict):
                            epoch_data = {
                                'loss': epoch_log.get('loss'),
                                'val_loss': epoch_log.get('val_loss'),
                                'accuracy': epoch_log.get('accuracy'),
                                'val_accuracy': epoch_log.get('val_accuracy')
                            }
                            print(f"Epoch log: {epoch_data}")
                            full_history.append(epoch_data)

            plot_data = {
                'loss': [epoch['loss'] for epoch in full_history],
                'val_loss': [epoch['val_loss'] for epoch in full_history],
                'accuracy': [epoch['accuracy'] for epoch in full_history],
                'val_accuracy': [epoch['val_accuracy'] for epoch in full_history],
            }
            print(f"Plot data: {plot_data}")
            plot_training_history(
                plot_data,
                save_path=os.path.join(temp_save_dir, "training_history.png"),
                epochs=training_params["epochs"],
                batch_size=training_params["batch_size"],
                learning_rate=training_params["learning_rate"],
                num_files=training_info["num_files"],
                dataset_size=dataset_size
            )

        except Exception as e:
            print(f"An error occurred during training: {e}")
            return

    all_input_sequences = []
    all_target_tokens = []

    end_time = datetime.now(japan_timezone)
    training_duration = (end_time - start_time).total_seconds() / 60  # 分単位に変換

    final_save_dir = os.path.join(model_save_path, f"{existing_model_dir}_{timestamp}_{int(training_duration)}m")
    os.rename(temp_save_dir, final_save_dir)

    model_path = os.path.join(final_save_dir, "best_model.h5")
    plot_path = os.path.join(final_save_dir, "training_history.png")

    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB単位に変換

    # 新しいトレーニングセッションの情報を作成
    new_training_session = {
        "training_duration_minutes": training_duration,
        "epochs": training_params["epochs"],
        "batch_size": training_params["batch_size"],
        "learning_rate": training_params["learning_rate"],
        "dataset_size": num_samples,
        "model_size_MB": model_size,
        "training_end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_loss": plot_data["loss"][-1] if plot_data["loss"] else None,
        "final_val_loss": plot_data["val_loss"][-1] if plot_data["val_loss"] else None,
        "final_accuracy": plot_data["accuracy"][-1] if plot_data["accuracy"] else None,
        "final_val_accuracy": plot_data["val_accuracy"][-1] if plot_data["val_accuracy"] else None,
        "training_type": training_type  # 追加
    }

    # トレーニングセッションリストに追加
    if "training_sessions" not in training_info:
        training_info["training_sessions"] = []
    training_info["training_sessions"].append(new_training_session)

    with open(os.path.join(final_save_dir, "training_info.json"), "w") as info_file:
        json.dump(training_info, info_file, indent=4)

    print(f"Additional training finished.")
    print(f"Additional training duration: {training_duration:.2f} minutes")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Model parameters: {model.count_params()}")
    print(f"Final Loss: {new_training_session['final_loss']}")
    print(f"Final Validation Loss: {new_training_session['final_val_loss']}")
    print(f"Final Accuracy: {new_training_session['final_accuracy']}")
    print(f"Final Validation Accuracy: {new_training_session['final_val_accuracy']}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

