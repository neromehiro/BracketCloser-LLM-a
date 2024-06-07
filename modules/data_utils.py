# modules/data_utils.py
import os
import json
import numpy as np

# トークンとIDの対応付け
# 括弧の種類とキーワード
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]

# トークンとIDを対応付ける辞書 ID0はパディング用
token2id = {token: i + 1 for i, token in enumerate(tokens)}

# IDとトークンを対応付ける辞書 ID0はパディング用
id2token = {i + 1: token for i, token in enumerate(tokens)}

import json

def load_dataset(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # データが辞書の場合は値をリストに変換
        if isinstance(data, dict):
            data = list(data.values())
        return data
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None



def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []
    for i in range(len(encoded_tokens) - seq_length):
        input_sequences.append(encoded_tokens[i : i + seq_length])

        # ターゲットトークンとして、入力シーケンスの後に続く全ての括弧を含める
        j = i + seq_length
        while j < len(encoded_tokens) and encoded_tokens[j] in [1, 3, 5]:  # 1, 3, 5 は閉じ括弧のトークンID
            target_tokens.append(encoded_tokens[j])
            j += 1

    return np.array(input_sequences), np.array(target_tokens)