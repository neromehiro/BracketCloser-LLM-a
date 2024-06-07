import json

# テストデータの保存パス
test_data_path = "./dataset/original/test_bracket_dataset.json"

def load_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def split_input_output(data):
    input_output_pairs = []
    for item in data:
        input_seq = item.split(",output:")[0] + ",output"
        output_seq = item.split(",output:")[1]
        input_output_pairs.append((input_seq, output_seq))
    return input_output_pairs

# テストデータのロード
test_data = load_dataset(test_data_path)

# 入力と出力を分割
input_output_pairs = split_input_output(test_data)

# 分割結果の確認
for idx, (input_seq, output_seq) in enumerate(input_output_pairs):
    print(f"問題{idx + 1}\n入力: {input_seq}\n出力: {output_seq}\n")

# 確認のため結果をファイルに保存
with open('input_output_pairs.txt', 'w', encoding='utf-8') as f:
    for idx, (input_seq, output_seq) in enumerate(input_output_pairs):
        f.write(f"問題{idx + 1}\n入力: {input_seq}\n出力: {output_seq}\n\n")
