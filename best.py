import optuna
import os
import json

# データベースファイルのパス
storage_path = "sqlite:///optuna_studies/hyper_lstm_1/optuna_study.db"

# study_nameの抽出（パスのディレクトリ名を取得）
study_name = os.path.basename(os.path.dirname(storage_path))

# Studyオブジェクトのロード
study = optuna.load_study(study_name=study_name, storage=storage_path)

# 最も性能の良かった試行の取得
best_trial = study.best_trial

# 結果の保存先ファイルパス
output_path = "optuna_studies/hyper_lstm_1/best_para.json"

# データの整形
best_trial_data = {
    "value": best_trial.value,
    "params": best_trial.params
}

# JSON形式で保存
with open(output_path, 'w') as f:
    json.dump(best_trial_data, f, indent=4)

print(f"Best trial data has been saved to {output_path}")
