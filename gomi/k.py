import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import load_model

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


# BERTモデルの読み込み
bert_model_path = 'models/bert_20240605_105833_9m/best_model.h5'
bert_model = load_model(bert_model_path)

# GPTモデルの読み込み
gpt_model_path = 'models/gpt_20240605_021525_151m/best_model.h5'
gpt_model = load_model(gpt_model_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})

# 高次元ベクトルの抽出関数
def extract_bert_vectors(model, input_data):
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('global_average_pooling1d').output)
    intermediate_output = intermediate_layer_model.predict(input_data)
    return intermediate_output

def extract_gpt_vectors(model, input_data, attention_mask):
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('global_average_pooling1d').output)
    intermediate_output = intermediate_layer_model.predict([input_data, attention_mask])
    return intermediate_output

# テストサンプルの定義
test_sample = [
    [0, 1, 4, 0, 4, 5, 1, 4, 2, 3, 2, 3, 0, 1, 4, 5, 4, 8, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
test_sample = tf.constant(test_sample)

# attention_maskの定義 (0の部分はパディングに相当する部分)
attention_mask = tf.constant([[1] * 20 + [0] * 10])

# BERTモデルのベクトル抽出
bert_vectors = extract_bert_vectors(bert_model, test_sample)
print("BERTの高次元ベクトル:", bert_vectors)

# GPTモデルのベクトル抽出
gpt_vectors = extract_gpt_vectors(gpt_model, test_sample, attention_mask)
print("GPTの高次元ベクトル:", gpt_vectors)

# ベクトルの比較
# ユークリッド距離の計算
distance = np.linalg.norm(bert_vectors - gpt_vectors)
print("ベクトル間の距離:", distance)