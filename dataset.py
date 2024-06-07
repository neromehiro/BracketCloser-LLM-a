import os
import json
import random
from typing import List, Tuple
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 括弧の種類とキーワード
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]

# トークンとIDを対応付ける辞書
token2id = {token: i + 1 for i, token in enumerate(tokens)}

# IDとトークンを対応付ける辞書
id2token = {i + 1: token for i, token in enumerate(tokens)}

# データの保存先ディレクトリ
dirs = {
    "original": "./components/dataset/original",
    "tokenize": "./components/dataset/tokenize",
    "preprocessed": "./components/dataset/preprocessed",
}

BRACKETS = {'(': ')', '【': '】', '{': '}'}

def ensure_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"ディレクトリ {directory} を作成しました。")
        else:
            print(f"ディレクトリ {directory} は既に存在します。")
    except Exception as e:
        print(f"ディレクトリ {directory} の作成に失敗しました。エラー: {e}")

def tokenize_string(string: str) -> List[str]:
    tokens = []
    current_token = ""
    for char in string:
        if char in token2id:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens

def preprocess_and_save_dataset(dataset, filepath, max_seq_length=30):
    for directory in dirs.values():
        ensure_dir(directory)

    # Save original dataset
    original_path = os.path.join(dirs["original"], filepath)
    try:
        with open(original_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(dataset, ensure_ascii=False))
        print(f"{original_path} の保存に成功しました。")
    except Exception as e:
        print(f"{original_path} の保存に失敗しました。エラー: {e}")

    # Tokenize dataset
    tokenized_dataset = [tokenize_string(data) for data in dataset]
    tokenize_path = os.path.join(dirs["tokenize"], filepath)
    try:
        with open(tokenize_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenized_dataset, ensure_ascii=False))
        print(f"{tokenize_path} の保存に成功しました。")
    except Exception as e:
        print(f"{tokenize_path} の保存に失敗しました。エラー: {e}")

    # Preprocess dataset
    preprocessed_dataset = [[token2id[token] for token in data if token in token2id] for data in tokenized_dataset]

    # パディング
    preprocessed_dataset = pad_sequences(preprocessed_dataset, maxlen=max_seq_length, padding='post', value=0).tolist()

    preprocessed_path = os.path.join(dirs["preprocessed"], filepath)
    try:
        with open(preprocessed_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(preprocessed_dataset, ensure_ascii=False))
        print(f"{preprocessed_path} の保存に成功しました。")
    except Exception as e:
        print(f"{preprocessed_path} の保存に失敗しました。エラー: {e}")

def generate_bracket_sequence(max_depth: int) -> str:
    if max_depth == 0:
        return ""
    
    sequence = ""
    stack = []
    for _ in range(random.randint(1, 20)):  # 1から20個の括弧を生成
        if len(stack) < max_depth and random.random() > 0.3:
            # 開く括弧を追加
            bracket = random.choice(list(BRACKETS.keys()))
            sequence += bracket
            stack.append(bracket)
        else:
            # 閉じる括弧を追加
            if stack:
                open_bracket = stack.pop()
                close_bracket = BRACKETS[open_bracket]
                sequence += close_bracket

    # 残っている開く括弧を閉じる
    while stack:
        open_bracket = stack.pop()
        close_bracket = BRACKETS[open_bracket]
        sequence += close_bracket
    
    return sequence

def close_brackets(seq: str) -> str:
    stack = []
    output_seq = ""
    
    for char in seq:
        if char in BRACKETS.keys():  # Opening brackets
            stack.append(char)
        elif char in BRACKETS.values():  # Closing brackets
            if stack and BRACKETS[stack[-1]] == char:
                stack.pop()
            else:
                output_seq += char
    
    while stack:
        opening_bracket = stack.pop()
        output_seq += BRACKETS[opening_bracket]
    
    return output_seq

def adjust_output_position(input_seq: str, output_seq: str) -> Tuple[str, str]:
    if not output_seq:
        pos = random.randint(1, 3)
        input_seq, moved_output = input_seq[:-pos], input_seq[-pos:] + output_seq
        
        # 指定されたトークンが output に含まれている場合、それらを output の前に移動
        prohibited_tokens = ["(", "【", "{"]
        for token in prohibited_tokens:
            if token in moved_output:
                moved_output = moved_output.replace(token, "")
                input_seq = token + input_seq
        
        return input_seq, moved_output
    return input_seq, output_seq

def generate_brackets(n_samples: int, max_depth: int, min_len: int, max_len: int) -> List[str]:
    dataset = []
    for _ in range(n_samples):
        while True:
            sequence = generate_bracket_sequence(random.randint(1, max_depth))
            if min_len <= len(sequence) <= max_len:
                break
        input_seq = sequence
        output_seq = close_brackets(sequence)
        input_seq, output_seq = adjust_output_position(input_seq, output_seq)
        dataset.append(f"input:{input_seq},output:{output_seq}")
    
    return dataset

num_samples = 3000  # データセットのサンプル数
max_depth = 5  # 括弧の最大深さ
min_len = 5  # シーケンスの最小長
max_len = 20  # シーケンスの最大長

# データセットの生成
dataset = generate_brackets(num_samples, max_depth, min_len, max_len)

# データセットの前処理と保存
preprocess_and_save_dataset(dataset, "bracket_dataset.json", max_seq_length=30)
print("データセットが保存された場所:", os.path.join(dirs["original"], "bracket_dataset.json"))
print("保存するデータセット:", dataset)

# テストデータのサンプル数
num_test_samples = 100

# テストデータの生成
test_dataset = generate_brackets(num_test_samples, max_depth, min_len, max_len)

# テストデータの前処理と保存
preprocess_and_save_dataset(test_dataset, "test_bracket_dataset.json", max_seq_length=30)
print("テストデータセットが保存された場所:", os.path.join(dirs["original"], "test_bracket_dataset.json"))
