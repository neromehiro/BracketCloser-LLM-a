# ファイル名：/app/Transformer-self-implementation/kakko/components/use_transformer.py

from tensorflow.keras.models import load_model
import numpy as np

# 元のトークンとIDの対応付け
tokens = ["(", ")", "[", "]", "{", "}"]
token2id = {token: i for i, token in enumerate(tokens)}
id2token = {i: token for i, token in enumerate(tokens)}

# モデルの読み込み先
model_path = "/app/Transformer-self-implementation/kakko/models/mymodel0.h5"

# モデルの読み込み
model = load_model(model_path)

def predict_next_token(model, token_sequence):
    # 入力のトークンをIDに変換
    input_sequence = [token2id[token] for token in token_sequence]
    # モデルへの入力は (1, sequence_length) の形状である必要がある
    input_sequence = np.array(input_sequence)[np.newaxis, :]
    # モデルを使って予測を行う
    predictions = model.predict(input_sequence)
    # 最も確率の高いトークンのIDを取得
    predicted_token_id = np.argmax(predictions[0])
    # IDをトークンに戻す
    predicted_token = id2token[predicted_token_id]
    return predicted_token

if __name__ == "__main__":
    token_sequence = ["(", "[", "{", "}", "]"]
    predicted_token = predict_next_token(model, token_sequence)
    print(f"The next token is likely to be: {predicted_token}")
