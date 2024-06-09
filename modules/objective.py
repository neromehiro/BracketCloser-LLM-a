# modules/objective.py
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.data_utils import load_dataset, tokens
from modules.training_utils import train_model
import wandb
from wandb.integration.keras import WandbCallback
from modules.setup import setup, parse_time_limit
from datetime import datetime,timedelta
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


def objective(trial, architecture, best_loss, encode_dir_path, create_save_folder_func, study, study_name):
    model_architecture_func = MODEL_ARCHITECTURES[architecture]
    
    # パラメータの提案
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 512)
    regularizer_type = trial.suggest_categorical("regularizer_type", ['l1', 'l2'])
    regularizer_value = trial.suggest_float("regularizer_value", 1e-5, 1e-1, log=True)
    seq_length = 30

    # アーキテクチャに依存するパラメータの提案
    if architecture == "gru":
        embedding_dim = trial.suggest_int("embedding_dim", 64, 128)
        gru_units = trial.suggest_int("gru_units", 64, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        recurrent_dropout_rate = trial.suggest_float("recurrent_dropout_rate", 0.1, 0.4)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, gru_units, dropout_rate, recurrent_dropout_rate, regularizer_type, regularizer_value)
    
    elif architecture == "transformer":
        embedding_dim = trial.suggest_int("embedding_dim", 64, 256)
        num_heads = trial.suggest_int("num_heads", 2, 8)
        ffn_units = trial.suggest_int("ffn_units", 256, 512)
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.4)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, num_heads, ffn_units, dropout_rate, regularizer_type, regularizer_value)
    
    elif architecture == "lstm":
        embedding_dim = trial.suggest_int("embedding_dim", 64, 128)
        lstm_units = trial.suggest_int("lstm_units", 64, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        recurrent_dropout_rate = trial.suggest_float("recurrent_dropout_rate", 0.1, 0.4)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, lstm_units, dropout_rate, recurrent_dropout_rate, num_layers, regularizer_type, regularizer_value)
    
    elif architecture == "bert":
        embedding_dim = trial.suggest_int("embedding_dim", 64, 128)
        num_heads = trial.suggest_int("num_heads", 2, 4)
        ffn_units = trial.suggest_int("ffn_units", 128, 256)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, num_heads, ffn_units, num_layers, dropout_rate, regularizer_type, regularizer_value)
    
    elif architecture == "gpt":
        embedding_dim = trial.suggest_int("embedding_dim", 64, 128)
        num_heads = trial.suggest_int("num_heads", 2, 4)
        ffn_units = trial.suggest_int("ffn_units", 128, 256)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, num_heads, ffn_units, dropout_rate, regularizer_type, regularizer_value)
    
    vocab_set = set(tokens)
    all_input_sequences = []
    all_target_tokens = []

    num_datasets = 0
    num_files = 10

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            if encoded_tokens_list is None:
                print(f"Skipping file {file} as it contains no data")
                continue
            for encoded_tokens in encoded_tokens_list:
                num_datasets += 1
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                all_input_sequences.extend(input_sequences)
                all_target_tokens.extend(target_tokens)

    if not all_input_sequences or not all_target_tokens:
        print("No data for training.")
        return float('inf')

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    save_path = create_save_folder_func()
    print("この部分にsaveします  " + save_path)
    timestamp = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d%H%M%S")
    temp_model_path = os.path.join(save_path, f"temp_model_{trial.number}_{timestamp}.h5")
    os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)

    # Early Stoppingの設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    try:
        history, dataset_size = train_model(
            model, 
            all_input_sequences, 
            all_target_tokens, 
            epochs=10,  # 初期エポック数
            batch_size=batch_size, 
            model_path=temp_model_path, 
            num_files=num_files, 
            learning_rate=learning_rate, 
            architecture=architecture, 
            model_architecture_func=model_architecture_func,
            callbacks=[early_stopping]  # Early Stoppingを追加
        )
        
        # モデルの保存
        print(f"Saving model to {temp_model_path}")
        model.save(temp_model_path)
        print(f"Model saved to {temp_model_path}")
            
        if history is None or isinstance(history, float):
            print("Training failed with invalid return. Returning inf loss.")
            return float('inf')
        else:
            loss = history.history['loss'][-1]
            if loss < best_loss:
                best_loss = loss
                print(f"New best model found with loss: {best_loss}")

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
                    "batch_size": batch_size,
                    "epochs": len(history.history['loss']),
                    "learning_rate": learning_rate,
                    "embedding_dim": embedding_dim,
                    "gru_units": gru_units if architecture == 'gru' else None,
                    "dropout_rate": dropout_rate,
                    "recurrent_dropout_rate": recurrent_dropout_rate if architecture == 'gru' else None,
                    "num_layers": num_layers if architecture in ['lstm', 'bert'] else None,
                    "regularizer_type": regularizer_type,
                    "regularizer_value": regularizer_value,
                    "model_path": temp_model_path
                }
                
                # Optunaのユーザー属性として保存
                trial.set_user_attr('metadata', metadata)
                
                # 最良試行情報を保存
                save_best_trial_to_json(study, study_name)
                
            return loss
    except Exception as e:
        print(f"Training failed with exception: {e}")
        return float('inf')

    
def save_best_trial_to_json(study, study_name):
    best_trial = study.best_trial

    # 保存先ディレクトリのパス
    output_dir = os.path.join("optuna_studies", study_name)
    output_path = os.path.join(output_dir, "best_para.json")

    # ディレクトリが存在するか確認し、存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # データの整形
    best_trial_data = {
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs  # ユーザー属性を含める
    }

    # epochsがparamsに含まれていない場合に追加
    if 'epochs' not in best_trial_data['params']:
        best_trial_data['params']['epochs'] = 5  # 例: デフォルトのエポック数

    # JSON形式で保存
    with open(output_path, 'w') as f:
        json.dump(best_trial_data, f, indent=4)

    print(f"Best trial data has been saved to {output_path}")

    