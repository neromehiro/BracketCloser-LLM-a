# modules/training_utils.py

import os
import time
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from datetime import datetime,timedelta  # datetimeモジュールをインポート
import wandb


class WandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log(logs)



class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)      


class CustomTrainingHistory(tf.keras.callbacks.Callback):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))
        self.history["accuracy"].append(logs.get("accuracy"))
        self.history["val_accuracy"].append(logs.get("val_accuracy"))

# 実質hyperしか使っていない. 正答率をグラフ化するために分けてcustomが誕生
class TrainingHistory(tf.keras.callbacks.Callback):
    def __init__(self, model_path, model_architecture_func, save_interval=1):
        super().__init__()
        self.model_path = model_path
        self.model_architecture_func = model_architecture_func
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        self.save_interval = save_interval
        self.process_id = os.getpid()
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))
        self.history["accuracy"].append(logs.get("accuracy"))
        self.history["val_accuracy"].append(logs.get("val_accuracy"))

        if (epoch + 1) % self.save_interval == 0:
            self.save_metadata_and_model(epoch, logs)

    def load_best_loss(self, metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.best_loss = metadata.get('best_loss', float('inf'))
        except IOError as e:
            print(f"Failed to load metadata: {e}")
# TRAINING_MODES に含まれる変数を渡すための変更
def train_model_single(model, input_sequences, target_tokens, epochs, batch_size, model_path, num_files, learning_rate, architecture, model_architecture_func, **kwargs):
    # 他のパラメータを kwargs から取り出す
    embedding_dim = kwargs.get('embedding_dim', 64)
    gru_units = kwargs.get('gru_units', 64)
    lstm_units = kwargs.get('lstm_units', 64)
    num_heads = kwargs.get('num_heads', 2)
    ffn_units = kwargs.get('ffn_units', 128)
    num_layers = kwargs.get('num_layers', 1)
    dropout_rate = kwargs.get('dropout_rate', 0.2)
    recurrent_dropout_rate = kwargs.get('recurrent_dropout_rate', 0.2)
    
    # 重複するパラメータの優先順位を設定
    kwargs.update({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_files': num_files
    })

    # 他の部分は元のまま
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(input_sequences))

        sample_weights = np.where(target_tokens != 0, 1.0, 0.0)

        # 'transformer', 'gpt', 'bert' などのアーキテクチャに対してのみ attention_mask を作成
        if 'transformer' in architecture or 'gpt' in architecture or 'bert' in architecture:
            attention_mask = (input_sequences != 0).astype(np.float32)

            train_inputs = {
                'input_1': input_sequences[:-num_validation_samples],
                'attention_mask': attention_mask[:-num_validation_samples]
            }
            val_inputs = {
                'input_1': input_sequences[-num_validation_samples:],
                'attention_mask': attention_mask[-num_validation_samples:]
            }

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_inputs,
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (val_inputs,
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)
        else:
            # attention_mask を使用しないアーキテクチャの場合
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[:-num_validation_samples],
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[-num_validation_samples:],
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)

        train_dataset = train_dataset.shuffle(buffer_size=1024)

        # データのバッチ形状を確認
        for data, labels, weights in train_dataset.take(1):
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"Train data batch shape for {key}: {value.shape}")
            else:
                print("Train data batch shape: ", data.shape)
            print("Train labels batch shape: ", labels.shape)
            print("Train sample weights batch shape: ", weights.shape)

        time_callback = TimeHistory()
        checkpoint_callback = ModelCheckpoint(filepath=model_path, save_weights_only=False, save_best_only=False, save_freq='epoch', verbose=1)
        history_callback = CustomTrainingHistory(model_path)

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

        try:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_dataset,
                callbacks=[time_callback, checkpoint_callback, history_callback, early_stopping_callback]
            )
            model.save(model_path, include_optimizer=True)
            return history_callback.history, len(input_sequences)
        except Exception as e:
            print(f"Training failed with exception: {e}")
            print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
            print(f"Train data shape: {input_sequences.shape}, Target data shape: {target_tokens.shape}")
        return None, 0


def train_model(model, input_sequences, target_tokens, epochs, batch_size, model_path, num_files, learning_rate, architecture, model_architecture_func,
                generate_data_func=None, embedding_dim=64, gru_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2, regularizer_type='l2', regularizer_value=0.01, callbacks=None):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(input_sequences))

        sample_weights = np.where(target_tokens != 0, 1.0, 0.0)

        if 'bert' in architecture or 'transformer' in architecture or 'gpt' in architecture:
            attention_mask = (input_sequences != 0).astype(np.float32)

            train_inputs = {
                'input_1': input_sequences[:-num_validation_samples],
                'attention_mask': attention_mask[:-num_validation_samples]  # キー名を 'attention_mask' に統一
            }
            val_inputs = {
                'input_1': input_sequences[-num_validation_samples:],
                'attention_mask': attention_mask[-num_validation_samples:]  # キー名を 'attention_mask' に統一
            }

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_inputs,
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (val_inputs,
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[:-num_validation_samples],
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[-num_validation_samples:],
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)

        train_dataset = train_dataset.shuffle(buffer_size=1024)

        # データの形状を出力
        for data, labels, weights in train_dataset.take(1):
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"Train data batch shape for {key}: {value.shape}")
            else:
                print("Train data batch shape: ", data.shape)
            print("Train labels batch shape: ", labels.shape)
            print("Train sample weights batch shape: ", weights.shape)

        time_callback = TimeHistory()
        checkpoint_callback = ModelCheckpoint(filepath=model_path, save_weights_only=False, save_best_only=True, save_freq='epoch', verbose=1)
        history_callback = TrainingHistory(model_path, model_architecture_func)

        try:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_dataset,
                callbacks=callbacks  # ここで追加されたコールバックを使用
            )
                
            new_best_loss = min(history.history['val_loss'])

            if new_best_loss < history_callback.best_loss:
                print(f"New best model saved with loss: {new_best_loss}")
            else:
                print(f"No improvement in best loss. Previous best loss: {history_callback.best_loss}, Current best loss: {new_best_loss}")
            
            return history, len(input_sequences)
        except Exception as e:
            print(f"Training failed with exception: {e}")
            print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
            print(f"Train data shape: {input_sequences.shape}, Target data shape: {target_tokens.shape}")
            return None, 0
    else:
        print("No data for training.")
        return None, 0

    def save_metadata_and_model(self, epoch, logs):
        metadata_path = self.model_path.replace('.h5', f'_epoch_{epoch + 1}_pid_{self.process_id}_metadata.json')
        model_checkpoint_path = self.model_path.replace('.h5', f'_epoch_{epoch + 1}_pid_{self.process_id}.h5')
        metadata = {
            "epoch": epoch + 1,
            "logs": logs,
            "time": (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),
            "model_architecture": self.model_architecture_func.__name__
        }
        print(f"Saving metadata to {metadata_path} with content: {metadata}")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        except IOError as e:
            print(f"Failed to save metadata: {e}")

        self.model.save(model_checkpoint_path)
        print(f"Model checkpoint saved to {model_checkpoint_path}")


    def load_best_loss(self, metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.best_loss = metadata['logs']['loss']
            print(f"Loaded best loss: {self.best_loss} from {metadata_path}")
        except (IOError, KeyError) as e:
            print(f"Failed to load best loss from {metadata_path}: {e}")



    


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def train_model_continue(model, input_sequences, target_tokens, epochs, batch_size, model_path, num_files, learning_rate, architecture, model_architecture_func, **kwargs):
    # 他のパラメータを kwargs から取り出す
    embedding_dim = kwargs.get('embedding_dim', 64)
    gru_units = kwargs.get('gru_units', 64)
    lstm_units = kwargs.get('lstm_units', 64)
    num_heads = kwargs.get('num_heads', 2)
    ffn_units = kwargs.get('ffn_units', 128)
    num_layers = kwargs.get('num_layers', 1)
    dropout_rate = kwargs.get('dropout_rate', 0.2)
    recurrent_dropout_rate = kwargs.get('recurrent_dropout_rate', 0.2)
    
    # 重複するパラメータの優先順位を設定
    kwargs.update({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_files': num_files
    })

    # 他の部分は元のまま
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(input_sequences))

        sample_weights = np.where(target_tokens != 0, 1.0, 0.0)

        # 'transformer', 'gpt', 'bert' などのアーキテクチャに対してのみ attention_mask を作成
        if 'transformer' in architecture or 'gpt' in architecture or 'bert' in architecture:
            attention_mask = (input_sequences != 0).astype(np.float32)

            train_inputs = {
                'input_1': input_sequences[:-num_validation_samples],
                'attention_mask': attention_mask[:-num_validation_samples]
            }
            val_inputs = {
                'input_1': input_sequences[-num_validation_samples:],
                'attention_mask': attention_mask[-num_validation_samples:]
            }

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_inputs,
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (val_inputs,
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)
        else:
            # attention_mask を使用しないアーキテクチャの場合
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[:-num_validation_samples],
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[-num_validation_samples:],
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)

        train_dataset = train_dataset.shuffle(buffer_size=1024)

        # データのバッチ形状を確認
        for data, labels, weights in train_dataset.take(1):
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"Train data batch shape for {key}: {value.shape}")
            else:
                print("Train data batch shape: ", data.shape)
            print("Train labels batch shape: ", labels.shape)
            print("Train sample weights batch shape: ", weights.shape)

        time_callback = TimeHistory()
        checkpoint_callback = ModelCheckpoint(filepath=model_path, save_weights_only=False, save_best_only=False, save_freq='epoch', verbose=1)
        history_callback = TrainingHistory(model_path, model_architecture_func)

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

        try:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_dataset,
                callbacks=[time_callback, checkpoint_callback, history_callback, early_stopping_callback]
            )
            model.save(model_path, include_optimizer=False, save_format='h5')
            return history_callback.history, len(input_sequences)
        except Exception as e:
            print(f"Training failed with exception: {e}")
            print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
            print(f"Train data shape: {input_sequences.shape}, Target data shape: {target_tokens.shape}")
        return None, 0
    else:
        print("Input sequences or target tokens are empty.")
        return None, 0



# best_model.jsonの後


def save_final_model_metadata(model_path, history, model_architecture_func, best_params):
    metadata_path = model_path.replace('best_model.h5', 'best_para.json')
    metadata = {
        "epoch": len(history),
        "logs": history[-1],
        "time": (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": model_architecture_func.__name__,
        "batch_size": best_params["batch_size"],
        "epochs": best_params["epochs"],
        "learning_rate": best_params["learning_rate"],
        "embedding_dim": best_params.get("embedding_dim"),
        "gru_units": best_params.get("gru_units"),
        "dropout_rate": best_params.get("dropout_rate"),
        "recurrent_dropout_rate": best_params.get("recurrent_dropout_rate"),
        "num_layers": best_params.get("num_layers")
    }
    print(f"Saving final metadata to {metadata_path} ")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Final metadata successfully saved to {metadata_path}")
    except IOError as e:
        print(f"Failed to save final metadata: {e}")



def plot_training_history(history, save_path, epochs, batch_size, learning_rate, num_files, dataset_size, avg_complete_accuracy, avg_partial_accuracy):
    if isinstance(history, dict):
        losses = history.get('loss', [])
        val_losses = history.get('val_loss', [])
        accuracies = history.get('accuracy', [])
        val_accuracies = history.get('val_accuracy', [])
        complete_accuracies = history.get('complete_accuracy', [])
        partial_accuracies = history.get('partial_accuracy', [])
    else:
        losses = [epoch_logs.get('loss', None) for epoch_logs in history]
        val_losses = [epoch_logs.get('val_loss', None) for epoch_logs in history]
        accuracies = [epoch_logs.get('accuracy', None) for epoch_logs in history]
        val_accuracies = [epoch_logs.get('val_accuracy', None) for epoch_logs in history]
        complete_accuracies = [epoch_logs.get('complete_accuracy', None) for epoch_logs in history]
        partial_accuracies = [epoch_logs.get('partial_accuracy', None) for epoch_logs in history]

    epochs_range = range(1, len(losses) + 1)

    plt.figure(figsize=(18, 6))  # プロットのサイズを変更

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, losses, label='Training Loss')
    if any(val is not None for val in val_losses):
        plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, accuracies, label='Training Accuracy')
    if any(val is not None for val in val_accuracies):
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    if complete_accuracies:  # complete_accuraciesが空でない場合のみプロット
        plt.plot(epochs_range, complete_accuracies, label='Complete Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Complete Accuracy')
        plt.title('Complete Accuracy')
        plt.legend()

    if partial_accuracies:  # partial_accuraciesが空でない場合のみプロット
        plt.plot(epochs_range, partial_accuracies, label='Partial Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Partial Accuracy')
        plt.title('Partial Accuracy')
        plt.legend()

    # モデルの平均精度をプロット
    plt.axhline(y=avg_complete_accuracy, color='r', linestyle='--', label='Average Complete Accuracy')
    plt.axhline(y=avg_partial_accuracy, color='b', linestyle='--', label='Average Partial Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.suptitle(f'Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Files: {num_files}, Dataset Size: {dataset_size}', y=1.05)
    plt.savefig(save_path)
    plt.close()



def save_final_model_metadata(model_path, history, model_architecture_func, best_params):
    metadata_path = model_path.replace('best_model.h5', 'best_model_metadata.json')
    metadata = {
        "epoch": len(history),
        "logs": history[-1],
        "time": (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": model_architecture_func.__name__,
        "batch_size": best_params["batch_size"],
        "epochs": best_params["epochs"],
        "learning_rate": best_params["learning_rate"],
        "embedding_dim": best_params.get("embedding_dim"),
        "gru_units": best_params.get("gru_units"),
        "dropout_rate": best_params.get("dropout_rate"),
        "recurrent_dropout_rate": best_params.get("recurrent_dropout_rate"),
        "num_layers": best_params.get("num_layers")
    }
    print(f"Saving final metadata to {metadata_path} ")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Final metadata successfully saved to {metadata_path}")
    except IOError as e:
        print(f"Failed to save final metadata: {e}")
        
def save_metadata(model_path, metadata):
    metadata_path = model_path.replace('.h5', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
        

def save_optuna_best_trial(study, output_path):
    best_trial = study.best_trial
    best_trial_logs = best_trial.user_attrs['metadata']
    best_trial_logs['value'] = best_trial.value
    best_trial_logs['params'] = best_trial.params

    with open(output_path, 'w') as f:
        json.dump(best_trial_logs, f, indent=4)

    print(f"Best trial data has been saved to {output_path}")
    

    
    