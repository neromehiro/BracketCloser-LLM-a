import json

# 前処理されたデータのファイルパス
preprocessed_data_path = './components/dataset/preprocessed/test_bracket_dataset.json'

# データを読み込む
with open(preprocessed_data_path, 'r') as file:
    preprocessed_data = json.load(file)

# 各シーケンスの長さを取得
sequence_lengths = [len(seq) for seq in preprocessed_data]

# 最大シーケンス長を取得
max_seq_length = max(sequence_lengths)

print("各シーケンスの長さ:", sequence_lengths)
print("最大シーケンス長:", max_seq_length)
