# data_generator.py
import os
import json
import random
from typing import List, Tuple
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 語彙: BOS/SEP/EOS と括弧のみ
tokens = ["[BOS]", "[SEP]", "[EOS]", "(", ")", "【", "】", "{", "}"]

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
    except Exception as e:
        print(f"ディレクトリ {directory} の作成に失敗しました。エラー: {e}")

def tokenize_io(sample: str) -> List[str]:
    """input:xxx,output:yyy, -> [BOS] x x x [SEP] y y y [EOS]"""
    try:
        inp = sample.split("input:")[1].split(",output:")[0]
        out = sample.split(",output:")[1].rstrip(",")
    except Exception:
        return []
    return ["[BOS]"] + list(inp) + ["[SEP]"] + list(out) + ["[EOS]"]

def preprocess_and_save_dataset(dataset, filename, max_seq_length=30):
    for directory in dirs.values():
        ensure_dir(directory)

    original_path = os.path.join(dirs["original"], filename)
    try:
        with open(original_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print(f"{original_path} の保存に成功しました。")
    except Exception as e:
        print(f"{original_path} の保存に失敗しました。エラー: {e}")

    tokenized_dataset = [tokenize_io(data) for data in dataset]
    tokenize_path = os.path.join(dirs["tokenize"], filename)
    try:
        with open(tokenize_path, "w", encoding="utf-8") as f:
            json.dump(tokenized_dataset, f, ensure_ascii=False, indent=4)
        print(f"{tokenize_path} の保存に成功しました。")
    except Exception as e:
        print(f"{tokenize_path} の保存に失敗しました。エラー: {e}")

    preprocessed_dataset = [
        [token2id[token] for token in data if token in token2id]
        for data in tokenized_dataset
    ]
    preprocessed_dataset = pad_sequences(preprocessed_dataset, maxlen=max_seq_length, padding='post', value=0).tolist()

    preprocessed_path = os.path.join(dirs["preprocessed"], filename)
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
        # 末尾を動かしたあとでも正しい閉じ順を再計算する
        pos = random.randint(1, 3)
        input_seq = input_seq[:-pos]
        output_seq = close_brackets(input_seq)
    return input_seq, output_seq

def count_missing_brackets(input_seq: str) -> int:
    """開き括弧の未閉じ数をカウントする。"""
    stack = []
    for token in input_seq:
        if token in BRACKETS:
            stack.append(token)
        elif stack and BRACKETS.get(stack[-1]) == token:
            stack.pop()
    return len(stack)

def extract_input_part(sample: str) -> str:
    try:
        return sample.split("input:")[1].split(",output")[0]
    except Exception:
        return ""

def make_sample_with_missing(target_missing: int, max_depth: int, min_len: int, max_len: int) -> Tuple[str, int]:
    """指定した未閉じ数（3 もしくは 4+）のサンプルを生成する。"""
    attempts = 0
    sample = ""
    missing = 0
    while attempts < 50:
        open_tok = random.choice(list(BRACKETS.keys()))
        seq = open_tok * target_missing

        # 未閉じ数を変えないよう、別種の括弧でバランスしたペアを足して長さを稼ぐ
        other_opens = [k for k in BRACKETS.keys() if k != open_tok]
        while len(seq) < min_len and other_opens:
            extra_open = random.choice(other_opens)
            seq += extra_open + BRACKETS[extra_open]

        seq = seq[:max_len]

        output_seq = close_brackets(seq)
        seq, output_seq = adjust_output_position(seq, output_seq)
        missing = count_missing_brackets(seq)

        if (target_missing == 3 and missing == 3) or (target_missing >= 4 and missing >= 4):
            sample = f"input:{seq},output:{output_seq},"
            break
        attempts += 1

    if not sample:
        # フォールバック（最後に生成したものを採用）
        sample = f"input:{seq},output:{output_seq},"
    return sample, missing

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

def generate_test_data(num_samples: int, max_depth: int = 5, min_len: int = 5, max_len: int = 20) -> List[str]:
    target_three = 30
    target_four_plus = 30
    if target_three + target_four_plus > num_samples:
        raise ValueError(f"num_samples={num_samples} では 3回と4回以上を各{target_three}件確保できません。サンプル数を増やしてください。")

    dataset = generate_brackets(num_samples, max_depth, min_len, max_len)

    dataset_with_missing = []
    count_three = 0
    count_four = 0
    for item in dataset:
        missing = count_missing_brackets(extract_input_part(item))
        dataset_with_missing.append((item, missing))
        if missing == 3:
            count_three += 1
        elif missing >= 4:
            count_four += 1

    need_three = max(0, target_three - count_three)
    need_four = max(0, target_four_plus - count_four)

    extras = []
    for _ in range(need_three):
        sample, missing = make_sample_with_missing(3, max_depth, min_len, max_len)
        extras.append((sample, missing))
    for _ in range(need_four):
        sample, missing = make_sample_with_missing(4, max_depth, min_len, max_len)
        extras.append((sample, missing))

    dataset_with_missing.extend(extras)
    random.shuffle(dataset_with_missing)

    # 上限より多い場合は、ターゲット数を守りながら間引く
    count_three += sum(1 for _, m in extras if m == 3)
    count_four += sum(1 for _, m in extras if m >= 4)

    while len(dataset_with_missing) > num_samples:
        removed = False
        for idx, (_, missing) in enumerate(dataset_with_missing):
            if missing >= 4 and count_four > target_four_plus:
                dataset_with_missing.pop(idx)
                count_four -= 1
                removed = True
                break
            if missing == 3 and count_three > target_three:
                dataset_with_missing.pop(idx)
                count_three -= 1
                removed = True
                break
            if missing < 3:
                dataset_with_missing.pop(idx)
                removed = True
                break
        if not removed:
            # これ以上安全に削れない場合は末尾を落とす
            dataset_with_missing.pop()

    return [sample for sample, _ in dataset_with_missing[:num_samples]]

if __name__ == "__main__":
    # テストデータのサンプル数
    num_test_samples = 100

    # テストデータの生成
    test_dataset = generate_test_data(num_test_samples)

    # テストデータの前処理と保存
    preprocess_and_save_dataset(test_dataset, "test_bracket_dataset.json", max_seq_length=30)
    print("テストデータセットが保存された場所:", os.path.join(dirs["original"], "test_bracket_dataset.json"))
