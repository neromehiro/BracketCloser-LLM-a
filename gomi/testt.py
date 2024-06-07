
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed, Dropout

# トークンのセット
tokens = ["(", ")", "[", "]", "【", "】", "{", "}", "input", ",output", ","]
# トークンとIDを対応付ける辞書
token2id = {token: i + 1 for i, token in enumerate(tokens)}

# IDとトークンを対応付ける辞書
id2token = {i + 1: token for i, token in enumerate(tokens)}

# サンプルデータ生成
def generate_sample_data():
    input_sequences = [
        "input:{}()[]", "input:(){}[]", "input:[]{}()", "input:【】{}", "input:()【】", 
        "input:{}()[]", "input:()【】", "input:【】{}", "input:【】[]", "input:[]【】",
    ]
    output_sequences = [
        "})]", ")}]", "]})", "】}", ")】", "})]", "】)", "】}", "】]", "]】"
    ]
    
    # 入力データのエンコーディング
    X = [pad_sequences([[token2id[char] for char in seq.replace("input:", "")]], maxlen=10, padding='post')[0] for seq in input_sequences]
    
    # 出力データのエンコーディング
    y = [[token2id[char] for char in seq] for seq in output_sequences]
    y = pad_sequences(y, maxlen=10, padding='post')  # 入力と同じく10次元にパディング
    
    # One-hotエンコード
    y = np.array([to_categorical(seq, num_classes=len(tokens)) for seq in y])
    
    return np.array(X), np.array(y)

X, y = generate_sample_data()

# モデル定義
model = Sequential()
model.add(Embedding(input_dim=len(tokens), output_dim=10, input_length=10))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))  # ドロップアウトレイヤーを追加して過学習を防ぐ
model.add(TimeDistributed(Dense(len(tokens), activation='softmax')))

# モデルコンパイル
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 学習
model.fit(X, y, epochs=500, verbose=1)  # エポック数を増やす

# 推論関数
def predict(input_str):
    input_seq = pad_sequences([[token2id[char] for char in input_str.replace("input:", "")]], maxlen=10, padding='post')[0]
    input_seq = np.expand_dims(input_seq, axis=0)
    predicted_prob_seq = model.predict(input_seq)
    
    # 学習データから不要なパディング部分を切り取る
    predicted_ids = [np.argmax(p, axis=-1) for p in predicted_prob_seq[0] if np.argmax(p, axis=-1) != 0]
    return "".join([id2token[id] for id in predicted_ids])

# 入力に対する出力をテスト
test_inputs = ["input:{}()[]", "input:(){}[]", "input:[]{}()", "input:【】{}", "input:()【】"]
for test_input in test_inputs:
    print(f"Input: {test_input}, Predicted Output: {predict(test_input)}")