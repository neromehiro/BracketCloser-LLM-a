import os
import numpy as np
import tensorflow as tf
from modules.data_utils import load_dataset, prepare_sequences, tokens
from modules.training_utils import train_model
import time
import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import MultiHeadAttention

class CustomMultiHeadAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None, **kwargs):
        if attention_mask is not None:
            batch_size = tf.shape(query)[0]
            seq_length = tf.shape(query)[1]
            # attention_maskの形状を(batch_size, 1, seq_length, seq_length)に変換
            attention_mask = tf.reshape(attention_mask, (batch_size, 1, 1, seq_length))
            attention_mask = tf.cast(attention_mask, dtype=self.compute_dtype)
            attention_mask = tf.where(attention_mask == 0, -1e9, 0)

        return super().call(query, value, key=key, attention_mask=attention_mask, return_attention_scores=return_attention_scores, training=training)



class TrainingHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        self.history.append(logs.copy())

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# データセットの保存先ディレクトリ
encode_dir_path = "./dataset/preprocessed/"

def create_test_data_from_existing_dataset(seq_length, num_files):
    all_input_sequences = []
    all_target_tokens = []

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:  # num_filesに基づいてファイル数を制限
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                if len(encoded_tokens) > seq_length:
                    input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                    all_input_sequences.extend(input_sequences)
                    all_target_tokens.extend(target_tokens)
                else:
                    print(f"Not enough data in: {file_path}")

    return np.array(all_input_sequences), np.array(all_target_tokens)

def define_gpt_model(seq_length, output_dim, learning_rate):
    inputs = tf.keras.layers.Input(shape=(seq_length,), name='input_1')
    attention_mask = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.float32, name='input_2')

    # Embedding Layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=output_dim, output_dim=64)(inputs)
    
    # Custom Multi-Head Attention Layer ここで、マスクを渡します
    attention_layer = CustomMultiHeadAttention(num_heads=8, key_dim=64)(embedding_layer, embedding_layer, attention_mask=attention_mask)
    
    # Add & Norm層
    add_norm_layer = tf.keras.layers.Add()([embedding_layer, attention_layer])
    norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm_layer)
    
    # Feed Forward Network
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(64)
    ])
    ffn_output = ffn(norm_layer)
    
    # Add & Norm層
    add_norm_layer2 = tf.keras.layers.Add()([norm_layer, ffn_output])
    norm_layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm_layer2)
    
    # Global Average Pooling Layer
    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(norm_layer2)
    
    # 出力層
    outputs = tf.keras.layers.Dense(output_dim, activation=None)(gap_layer)
    
    # Create Model
    model = tf.keras.Model(inputs=[inputs, attention_mask], outputs=outputs)
    
    # 出力形状を確認するデバッグログ
    print("Debug: Model output shape after pooling:", outputs.shape)
    
    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    return model


def train_model(model, input_sequences, target_tokens, epochs, batch_size, model_path, num_files, learning_rate, architecture):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(input_sequences))

        if 'transformer' in architecture or 'gpt' in architecture:
            attention_mask = np.ones_like(input_sequences)
            train_dataset = tf.data.Dataset.from_tensor_slices(
                ({'input_1': input_sequences[:-num_validation_samples], 'input_2': attention_mask[:-num_validation_samples]}, target_tokens[:-num_validation_samples])
            ).batch(batch_size)
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                ({'input_1': input_sequences[-num_validation_samples:], 'input_2': attention_mask[-num_validation_samples:]}, target_tokens[-num_validation_samples:])
            ).batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[:-num_validation_samples], target_tokens[:-num_validation_samples])
            ).batch(batch_size)
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[-num_validation_samples:], target_tokens[-num_validation_samples:])
            ).batch(batch_size)

        train_dataset = train_dataset.shuffle(buffer_size=1024)
        validation_dataset = validation_dataset

        # データセットの形状を確認
        for data, labels in train_dataset.take(1):
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"Train data batch shape for {key}: {value.shape}")
            else:
                print("Train data batch shape: ", data.shape)
            print("Train labels batch shape: ", labels.shape)

        # デバッグログ追加: モデル出力の形状確認
        print("Debug: Model output shape:", model.output.shape)
        print("Debug: Target tokens shape:", target_tokens.shape)

        time_callback = TimeHistory()
        checkpoint_callback = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        history_callback = TrainingHistory()

        history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=[time_callback, checkpoint_callback, history_callback])

        model.save(model_path, include_optimizer=False, save_format='h5')
        
        return history_callback.history, len(input_sequences)
    else:
        print("No data for training.")
        return None, 0



def test_gpt_model_with_existing_data():
    seq_length = 1
    vocab_size = len(set(tokens))
    num_files = 5
    learning_rate = 0.001
    batch_size = 128
    epochs = 1

    # 既存のデータセットを使用してテストデータを生成
    input_sequences, target_tokens = create_test_data_from_existing_dataset(seq_length, num_files)

    # モデルを定義
    model = define_gpt_model(seq_length, vocab_size + 1, learning_rate)

    # データセットの形状を確認
    print(f"Input sequences shape: {input_sequences.shape}")
    print(f"Target tokens shape: {target_tokens.shape}")

    # モデルの出力形状を確認
    dummy_input = [tf.constant(input_sequences[:1]), tf.constant(np.ones_like(input_sequences[:1]))]
    dummy_output = model(dummy_input)
    print(f"Model output shape: {dummy_output.shape}")

    # トレーニングを実行
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ({'input_1': input_sequences, 'input_2': np.ones_like(input_sequences)}, target_tokens)
    ).batch(batch_size)
    
    history, dataset_size = train_model(model, input_sequences, target_tokens, epochs=epochs, batch_size=batch_size, model_path='test_model.h5', num_files=num_files, learning_rate=learning_rate, architecture='gpt')

    print("Training completed.")

if __name__ == "__main__":
    test_gpt_model_with_existing_data()
