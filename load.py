import optuna

def load_study_and_get_best_params(study_name, storage_name):
    # スタディをロード
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    # 最適な試行の結果を取得
    best_trial = study.best_trial
    
    # 最適な試行のパラメータを表示
    print("Best trial number:", best_trial.number)
    print("Best value (loss):", best_trial.value)
    print("Best parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    return best_trial.params

# 使用例
study_name = "hyper_lstm"
storage_name = "sqlite:///optuna_studies/hyper_lstm/optuna_study.db"

best_params = load_study_and_get_best_params(study_name, storage_name)
print("Best hyperparameters:", best_params)
