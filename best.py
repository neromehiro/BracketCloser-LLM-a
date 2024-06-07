import optuna
import os
import json

# データベースファイルのパス
storage_path = "sqlite:///optuna_studies/hyper_transformer_2/optuna_study.db"

# study_nameの抽出（パスのディレクトリ名を取得）
study_name = os.path.basename(os.path.dirname(storage_path))

# ディレクトリが存在するか確認し、存在しない場合は作成
os.makedirs(os.path.dirname(storage_path), exist_ok=True)

# Studyオブジェクトのロード
try:
    study = optuna.load_study(study_name=study_name, storage=storage_path)
except Exception as e:
    print(f"Error loading study: {e}")
    exit(1)

# 最も性能の良かった試行の取得
best_trial = study.best_trial

# 結果の保存先ファイルパス
output_dir = f"optuna_studies/{study_name}"
output_path = os.path.join(output_dir, "best_para.json")

# ディレクトリが存在するか確認し、存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# データの整形
best_trial_data = {
    "value": best_trial.value,
    "params": best_trial.params,
    "user_attrs": best_trial.user_attrs  # ユーザー属性を含める
}

# JSON形式で保存
with open(output_path, 'w') as f:
    json.dump(best_trial_data, f, indent=4)

print(f"Best trial data has been saved to {output_path}")
