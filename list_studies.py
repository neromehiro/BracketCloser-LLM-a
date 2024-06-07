import optuna

def list_studies(storage_name: str):
    # Studyの一覧を取得
    study_summaries = optuna.study.get_all_study_summaries(storage=storage_name)

    # Study名の表示
    print("Existing studies:")
    for study_summary in study_summaries:
        print(f" - {study_summary.study_name}")

if __name__ == "__main__":
    storage_name = "sqlite:///optuna_studies/hyper_gpt/optuna_study.db"  # 使用しているストレージのパス
    list_studies(storage_name)
