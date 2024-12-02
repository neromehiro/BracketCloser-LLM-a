
# optuna_data_generator.py
import os
import json
import random
from typing import List, Tuple
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 括弧の種類とキーワード
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
token2id = {token: i + 1 for i, token in enumerate(tokens)}
id2token = {i + 1: token for i, token in enumerate(tokens)}
BRACKETS = {'(': ')', '【': '】', '{': '}'}

def ensure_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"ディレクトリ {directory} を作成しました。")
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
    if (current_token):
        tokens.append(current_token)
    return tokens

def preprocess_and_save_dataset(dataset, base_dir, filename, max_seq_length=30):
    original_dir = os.path.join(base_dir, "dataset/original")
    tokenize_dir = os.path.join(base_dir, "dataset/tokenize")
    preprocessed_dir = os.path.join(base_dir, "dataset/preprocessed")

    for directory in [original_dir, tokenize_dir, preprocessed_dir]:
        ensure_dir(directory)

    original_path = os.path.join(original_dir, filename)
    try:
        with open(original_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print(f"{original_path} の保存に成功しました。")
    except Exception as e:
        print(f"{original_path} の保存に失敗しました。エラー: {e}")

    tokenized_dataset = [tokenize_string(data) for data in dataset]
    tokenize_path = os.path.join(tokenize_dir, filename)
    try:
        with open(tokenize_path, "w", encoding="utf-8") as f:
            json.dump(tokenized_dataset, f, ensure_ascii=False, indent=4)
        print(f"{tokenize_path} の保存に成功しました。")
    except Exception as e:
        print(f"{tokenize_path} の保存に失敗しました。エラー: {e}")

    preprocessed_dataset = [[token2id[token] for token in data if token in token2id] for data in tokenized_dataset]
    preprocessed_dataset = pad_sequences(preprocessed_dataset, maxlen=max_seq_length, padding='post', value=0).tolist()

    preprocessed_path = os.path.join(preprocessed_dir, filename)
    try:
        with open(preprocessed_path, "w", encoding="utf-8") as f:
            json.dump(preprocessed_dataset, f, ensure_ascii=False, indent=4)
        print(f"{preprocessed_path} の保存に成功しました。")
    except Exception as e:
        print(f"{preprocessed_path} の保存に失敗しました。エラー: {e}")

def generate_bracket_sequence(max_depth: int) -> str:
    if max_depth == 0:
        return ""
    
    sequence = ""
    stack = []
    for _ in range(random.randint(1, 20)):
        if len(stack) < max_depth and random.random() > 0.3:
            bracket = random.choice(list(BRACKETS.keys()))
            sequence += bracket
            stack.append(bracket)
        else:
            if stack:
                open_bracket = stack.pop()
                close_bracket = BRACKETS[open_bracket]
                sequence += close_bracket

    while stack:
        open_bracket = stack.pop()
        close_bracket = BRACKETS[open_bracket]
        sequence += close_bracket
    
    return sequence

def close_brackets(seq: str) -> str:
    stack = []
    output_seq = ""
    
    for char in seq:
        if char in BRACKETS.keys():
            stack.append(char)
        elif char in BRACKETS.values():
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
        dataset.append(f"input:{input_seq},output:{output_seq},")
    
    return dataset

def generate_test_data(num_samples: int = 1000, max_depth: int = 5, min_len: int = 5, max_len: int = 20) -> List[str]:
    return generate_brackets(num_samples, max_depth, min_len, max_len)

def extract_output(data: str) -> str:
    try:
        output_start = data.index('output:') + len('output:')
        output_end = data.rindex(',', output_start)
        output = data[output_start:output_end]
        return output
    except ValueError:
        return ''

def count_brackets(s: str) -> int:
    brackets = '(){}【】'
    return sum(1 for c in s if c in brackets)

def create_datasets(base_dir, epoch: int, num_samples: int = 1000, max_seq_length=30, learning_stage=None):
    filename = f"test_bracket_dataset_epoch_{epoch}.json"

    if learning_stage is None:
        # 従来通りの挙動
        dataset = generate_test_data(num_samples)
        preprocess_and_save_dataset(dataset, base_dir, filename, max_seq_length)
    else:
        # カリキュラム学習の実装
        if learning_stage == 1:
            required_counts = {1: num_samples}
        elif learning_stage == 2:
            half_samples = num_samples // 2
            required_counts = {1: half_samples, '2+': num_samples - half_samples}
        elif learning_stage == 3:
            required_counts = {'2+': num_samples}
        else:
            print(f"無効な learning_stage: {learning_stage}")
            return

        counts = {k: 0 for k in required_counts}
        dataset = []

        while any(counts[k] < required_counts[k] for k in counts):
            batch_size = max(1000, num_samples * 2)
            batch_data = generate_test_data(num_samples=batch_size)

            for data in batch_data:
                output = extract_output(data)
                bracket_count = count_brackets(output)

                if bracket_count == 0:
                    continue  # 出力がない場合はスキップ

                if bracket_count == 1 and 1 in required_counts and counts[1] < required_counts[1]:
                    dataset.append(data)
                    counts[1] += 1
                elif bracket_count >= 2 and '2+' in required_counts and counts['2+'] < required_counts['2+']:
                    dataset.append(data)
                    counts['2+'] += 1

                # 必要なデータが集まったらループを抜ける
                if all(counts[k] >= required_counts[k] for k in counts):
                    break

        # 必要なサンプル数に調整
        dataset = dataset[:num_samples]
        preprocess_and_save_dataset(dataset, base_dir, filename, max_seq_length)

if __name__ == "__main__":
    base_dir = "optuna_studies/hyper_transformer_1"
    epoch = 1  # デフォルトのepoch値、必要に応じて変更
    num_samples = 1000  # デフォルトで1000個のサンプルを生成
    learning_stage = 1  # カリキュラム学習の段階を指定（1, 2, 3）、未指定の場合は従来通り

    create_datasets(base_dir, epoch, num_samples, learning_stage=learning_stage)
    print("データセットの生成と保存が完了しました。")