import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.data_utils import load_dataset, tokens
from modules.training_utils import train_model
import wandb
from wandb.integration.keras import WandbCallback
from modules.setup import setup, parse_time_limit
from datetime import datetime, timedelta
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model,
    "lstm": define_lstm_model,
    "bert": define_bert_model,
    "gpt": define_gpt_model
}

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []

    for i in range(1, len(encoded_tokens)):
        input_seq = encoded_tokens[:i]
        target_seq = encoded_tokens[i]
        input_sequences.append(input_seq)
        target_tokens.append(target_seq)

    input_sequences = pad_sequences(input_sequences, maxlen=seq_length, padding='post', value=0)
    target_tokens = pad_sequences([target_tokens], maxlen=len(input_sequences), padding='post', value=0)[0]

    return input_sequences, target_tokens

def load_training_data(encode_dir_path, seq_length, num_files=10):
    all_input_sequences = []
    all_target_tokens = []

    for dirpath, _, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            if encoded_tokens_list is None:
                continue
            for encoded_tokens in encoded_tokens_list:
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                all_input_sequences.extend(input_sequences)
                all_target_tokens.extend(target_tokens)

    return all_input_sequences, all_target_tokens

def suggest_model_params(trial, architecture):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),  # 学習率の範囲を適切に設定
        "batch_size": trial.suggest_int("batch_size", 32, 128),  # バッチサイズの範囲を調整
        "regularizer_type": trial.suggest_categorical("regularizer_type", ['l1', 'l2']),
        "regularizer_value": trial.suggest_float("regularizer_value", 1e-6, 1e-3, log=True)  # 正則化の範囲を調整
    }

    if architecture == "gru":
        params.update({
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 128),
            "gru_units": trial.suggest_int("gru_units", 64, 256),  # GRUユニットの範囲を広げる
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3),  # ドロップアウト率の範囲を適切に設定
            "recurrent_dropout_rate": trial.suggest_float("recurrent_dropout_rate", 0.1, 0.3)  # リカレントドロップアウト率の範囲を適切に設定
        })
    elif architecture == "transformer":
        params.update({
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256),  # 埋め込み次元の範囲を調整
            "num_heads": trial.suggest_int("num_heads", 2, 8),  # ヘッド数の範囲を適切に設定
            "ffn_units": trial.suggest_int("ffn_units", 128, 512),  # FFNユニットの範囲を適切に設定
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3)  # ドロップアウト率の範囲を適切に設定
        })
    elif architecture == "lstm":
        params.update({
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 128),
            "lstm_units": trial.suggest_int("lstm_units", 64, 256),  # LSTMユニットの範囲を広げる
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3),  # ドロップアウト率の範囲を適切に設定
            "recurrent_dropout_rate": trial.suggest_float("recurrent_dropout_rate", 0.1, 0.3),  # リカレントドロップアウト率の範囲を適切に設定
            "num_layers": trial.suggest_int("num_layers", 1, 3)  # レイヤー数の範囲を適切に設定
        })
    elif architecture == "bert":
        params.update({
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256),  # 埋め込み次元の範囲を調整
            "num_heads": trial.suggest_int("num_heads", 2, 8),  # ヘッド数の範囲を適切に設定
            "ffn_units": trial.suggest_int("ffn_units", 128, 512),  # FFNユニットの範囲を適切に設定
            "num_layers": trial.suggest_int("num_layers", 1, 3),  # レイヤー数の範囲を適切に設定
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3)  # ドロップアウト率の範囲を適切に設定
        })
    elif architecture == "gpt":
        params.update({
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256),  # 埋め込み次元の範囲を調整
            "num_heads": trial.suggest_int("num_heads", 2, 8),  # ヘッド数の範囲を適切に設定
            "ffn_units": trial.suggest_int("ffn_units", 128, 512),  # FFNユニットの範囲を適切に設定
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3)  # ドロップアウト率の範囲を適切に設定
        })

    return params

def create_model(model_architecture_func, architecture, seq_length, vocab_size, params):
    if architecture == "gru":
        return model_architecture_func(seq_length, vocab_size, params["learning_rate"], params["embedding_dim"], params["gru_units"], params["dropout_rate"], params["recurrent_dropout_rate"], params["regularizer_type"], params["regularizer_value"])
    elif architecture == "transformer":
        return model_architecture_func(seq_length, vocab_size, params["learning_rate"], params["embedding_dim"], params["num_heads"], params["ffn_units"], params["dropout_rate"], params["regularizer_type"], params["regularizer_value"])
    elif architecture == "lstm":
        return model_architecture_func(seq_length, vocab_size, params["learning_rate"], params["embedding_dim"], params["lstm_units"], params["dropout_rate"], params["recurrent_dropout_rate"], params["num_layers"], params["regularizer_type"], params["regularizer_value"])
    elif architecture == "bert":
        return model_architecture_func(seq_length, vocab_size, params["learning_rate"], params["embedding_dim"], params["num_heads"], params["ffn_units"], params["num_layers"], params["dropout_rate"], params["regularizer_type"], params["regularizer_value"])
    elif architecture == "gpt":
        return model_architecture_func(seq_length, vocab_size, params["learning_rate"], params["embedding_dim"], params["num_heads"], params["ffn_units"], params["dropout_rate"], params["regularizer_type"], params["regularizer_value"])


def objective(trial, architecture, best_loss, encode_dir_path, create_trial_folder_func, study, study_name):
    trial_number = trial.number
    trial_path = create_trial_folder_func(trial_number)  # 修正
    model_architecture_func = MODEL_ARCHITECTURES[architecture]
    params = suggest_model_params(trial, architecture)
    seq_length = 30

    model = create_model(model_architecture_func, architecture, seq_length, len(tokens) + 1, params)

    all_input_sequences, all_target_tokens = load_training_data(encode_dir_path, seq_length)

    if not all_input_sequences or not all_target_tokens:
        print("No data for training.")
        return float('inf')

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    save_path = os.path.join('./optuna_studies', study_name, f'trial_{trial_number}')  # 修正
    print(f"Saving to {save_path}")  # 修正
    timestamp = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d%H%M%S")
    temp_model_path = os.path.join(save_path, f"temp_model_{trial.number}_{timestamp}.h5")
    os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    try:
        history, dataset_size = train_model(
            model, 
            all_input_sequences, 
            all_target_tokens, 
            epochs=10,  # Initial epochs
            batch_size=params['batch_size'], 
            model_path=temp_model_path, 
            num_files=10, 
            learning_rate=params['learning_rate'], 
            architecture=architecture, 
            model_architecture_func=model_architecture_func,
            callbacks=[early_stopping]
        )

        if history is None or isinstance(history, float):
            print("Training failed with invalid return. Returning inf loss.")
            return float('inf')

        loss = history.history['loss'][-1]
        print(f"Model saved with loss: {loss}")

        # モデルを一時ファイルに保存
        model.save(temp_model_path)  # 追加

        metadata = {
            "epoch": len(history.history['loss']),
            "logs": {
                "loss": history.history['loss'][-1],
                "accuracy": history.history.get('accuracy', [None])[-1],
                "weighted_accuracy": history.history.get('weighted_accuracy', [None])[-1],
                "val_loss": history.history.get('val_loss', [None])[-1],
                "val_accuracy": history.history.get('val_accuracy', [None])[-1],
                "val_weighted_accuracy": history.history.get('val_weighted_accuracy', [None])[-1]
            },
            "time": timestamp,
            "model_architecture": model_architecture_func.__name__,
            "batch_size": params["batch_size"],
            "epochs": len(history.history['loss']),
            "learning_rate": params["learning_rate"],
            "embedding_dim": params["embedding_dim"],
            "gru_units": params.get("gru_units"),
            "dropout_rate": params["dropout_rate"],
            "recurrent_dropout_rate": params.get("recurrent_dropout_rate"),
            "num_layers": params.get("num_layers"),
            "regularizer_type": params["regularizer_type"],
            "regularizer_value": params["regularizer_value"],
            "model_path": temp_model_path
        }
        
        trial.set_user_attr('metadata', metadata)
        save_best_trial_to_json(study, study_name)
        
        return loss

    except Exception as e:
        print(f"Training failed with exception: {e}")
        return float('inf')

def save_best_trial_to_json(study, study_name):
    best_trial = study.best_trial

    output_dir = os.path.join("optuna_studies", study_name)
    output_path = os.path.join(output_dir, "best_para.json")

    os.makedirs(output_dir, exist_ok=True)

    best_trial_data = {
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs
    }

    if 'epochs' not in best_trial_data['params']:
        best_trial_data['params']['epochs'] = 5

    with open(output_path, 'w') as f:
        json.dump(best_trial_data, f, indent=4)

    print(f"Best trial data has been saved to {output_path}")
