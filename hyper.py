import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from tqdm import tqdm
import glob
import wandb
from wandb.integration.keras import WandbCallback
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from dotenv import load_dotenv
from modules.setup import setup, parse_time_limit
from modules.objective import objective
from modules.utils import create_save_folder
from modules.data_utils import load_dataset, prepare_sequences, tokens
from modules.training_utils import train_model, plot_training_history, save_metadata, TrainingHistory
import optuna_data_generator

BEST_LOSS = float('inf')

# .envファイルの読み込み
load_dotenv()

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "disabled"

STORAGE_BASE_PATH = "./optuna_studies/"

def clean_up_files(save_path, keep_files=['best_model.h5', 'training_history.png', 'optuna_study.db']):
    files = glob.glob(os.path.join(save_path, '*'))
    for file in files:
        if os.path.basename(file) not in keep_files:
            os.remove(file)

def create_model(architecture, best_params, model_architecture_func, seq_length, vocab_size):
    model_params = {
        "embedding_dim": best_params["embedding_dim"],
        "dropout_rate": best_params["dropout_rate"],
        "learning_rate": best_params["learning_rate"]
    }
    if architecture in ["gru", "lstm"]:
        model_params["units"] = best_params[f"{architecture}_units"]
        model_params["recurrent_dropout_rate"] = best_params["recurrent_dropout_rate"]
    if architecture in ["transformer", "bert", "gpt"]:
        model_params["num_heads"] = best_params["num_heads"]
        model_params["ffn_units"] = best_params["ffn_units"]
    if architecture in ["lstm", "bert"]:
        model_params["num_layers"] = best_params["num_layers"]

    return model_architecture_func(seq_length, vocab_size + 1, **model_params)

def load_training_data(encode_dir_path, seq_length, num_files=10):
    all_input_sequences, all_target_tokens = [], []
    for dirpath, _, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            if encoded_tokens_list is None:
                continue
            for encoded_tokens in encoded_tokens_list:
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                all_input_sequences.extend(input_sequences)
                all_target_tokens.extend(target_tokens)
    return all_input_sequences, all_target_tokens

def save_best_model_info(model_path, metadata, save_path):
    json_path = os.path.join(save_path, "best_model_info.json")
    try:
        with open(json_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        print(f"Best model information saved to {json_path}")
    except IOError as e:
        print(f"Failed to save best model information: {e}")

def generate_training_data(encode_dir_path, seq_length, num_files=10):
    all_input_sequences, all_target_tokens = load_training_data(encode_dir_path, seq_length, num_files)
    return np.array(all_input_sequences), np.array(all_target_tokens)

def generate_datasets(base_dir, num_samples):
    valid_samples = [100, 300, 500, 800, 1000, 3000, 5000, 10000]
    if num_samples not in valid_samples:
        raise ValueError(f"Invalid number of samples. Choose from {valid_samples}")
    optuna_data_generator.create_datasets(base_dir, 1, num_samples)


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def initialize_wandb():
    wandb_api_key = os.getenv('WANDB_API_KEY')
    project_name = os.getenv('WANDB_PROJECT_NAME')
    run_name = os.getenv('WANDB_RUN_NAME')

    if not all([wandb_api_key, project_name, run_name]):
        raise ValueError("WANDB_API_KEY, WANDB_PROJECT_NAME, and WANDB_RUN_NAME must be set as environment variables.")

    wandb.login(key=wandb_api_key)
    wandb.init(project=project_name, name=run_name)

def resume_existing_study():
    global ENCODE_DIR_PATH, MODEL_SAVE_BASE_PATH

    studies = [f for f in os.listdir(STORAGE_BASE_PATH) if f.startswith("hyper_")]
    if not studies:
        print("No existing studies found. Starting a new study.")
        return None, None, None

    for i, study_folder in enumerate(studies):
        print(f"{i + 1}. {study_folder}")
    study_index = int(input("Enter the number of the study to resume: ").strip()) - 1
    study_folder = studies[study_index]
    study_name = study_folder
    architecture_name = study_folder.split('_')[1]
    model_architecture_func, architecture = setup(architecture_name)
    save_path = os.path.join(STORAGE_BASE_PATH, study_folder)
    metadata_path = os.path.join(save_path, 'best_model_metadata.json')
    training_history = TrainingHistory(model_path=os.path.join(save_path, 'best_model.h5'), model_architecture_func=model_architecture_func)
    training_history.load_best_loss(metadata_path)

    ENCODE_DIR_PATH = os.path.join(save_path, "dataset/preprocessed/")
    MODEL_SAVE_BASE_PATH = save_path

    return study_name, architecture, training_history

def start_new_study():
    global ENCODE_DIR_PATH, MODEL_SAVE_BASE_PATH, STORAGE_BASE_PATH

    architecture_name = input("Enter the model architecture (gru, transformer, lstm, bert, gpt): ").strip()
    model_architecture_func, architecture = setup(architecture_name)
    study_path = create_save_folder(STORAGE_BASE_PATH, architecture_name)
    study_name = os.path.basename(study_path)
    storage_name = f"sqlite:///{study_path}/optuna_study.db"
    training_history = TrainingHistory(model_path=os.path.join(study_path, 'best_model.h5'), model_architecture_func=model_architecture_func)

    ENCODE_DIR_PATH = os.path.join(study_path, "dataset/preprocessed/")
    MODEL_SAVE_BASE_PATH = study_path

    num_samples = int(input("Enter the number of samples for the dataset (100, 300, 500, 800, 1000, 3000, 5000, 10000): ").strip())
    generate_datasets(study_path, num_samples)

    return "2", study_name, model_architecture_func, study_path

def create_study_folder(model_save_base_path, architecture):
    base_folder_name = f"hyper_{architecture}"
    
    # フォルダが存在しない場合に作成
    if not os.path.exists(model_save_base_path):
        os.makedirs(model_save_base_path)
    
    # 既存のフォルダをチェックして連番を追加
    existing_folders = [f for f in os.listdir(model_save_base_path) if f.startswith(base_folder_name)]
    
    # 連番の最大値を取得
    max_index = 0
    for folder in existing_folders:
        try:
            index = int(folder.split('_')[-1])
            if index > max_index:
                max_index = index
        except ValueError:
            continue
    
    # 新しいフォルダのインデックス
    folder_index = max_index + 1
    folder_name = f"{base_folder_name}_{folder_index}"
    
    save_path = os.path.join(model_save_base_path, folder_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def create_trial_folder(base_path, trial_number):
    trial_folder_name = f"trial_{trial_number}"
    
    trial_path = os.path.join(base_path, trial_folder_name)
    os.makedirs(trial_path, exist_ok=True)

    return trial_path


def main():
    global STORAGE_BASE_PATH  # 追加
    initialize_wandb()

    architecture = None  # 追加
    training_history = None  # 追加

    option = input("Choose an option:\n1. Resume existing study\n2. Start a new study\nEnter 1 or 2: ").strip()
    
    if option == "2":
        option, study_name, model_architecture_func, save_path = start_new_study()
        architecture = study_name.split('_')[1]  # 追加
        training_history = TrainingHistory(model_path=os.path.join(save_path, 'best_model.h5'), model_architecture_func=model_architecture_func)  # 追加

    if option == "1":
        study_name, architecture, training_history = resume_existing_study()
        if study_name is None:
            option = "2"
            # 新しい試行を開始する場合も training_history を初期化する
            option, study_name, model_architecture_func, save_path = start_new_study()
            architecture = study_name.split('_')[1]
            training_history = TrainingHistory(model_path=os.path.join(save_path, 'best_model.h5'), model_architecture_func=model_architecture_func)

    # データベースファイルのディレクトリが存在することを確認
    db_path = os.path.join(STORAGE_BASE_PATH, study_name)
    ensure_directory_exists(db_path)

    time_limit_str = input("Enter the training time limit (e.g., '3min', '1hour', '5hour'): ").strip()
    time_limit = parse_time_limit(time_limit_str)
    start_time = datetime.now()

    progress_bar = tqdm(total=time_limit.total_seconds(), desc="Optimization Progress", unit="s")

    def callback(study, trial):
        elapsed_time = (datetime.now() - start_time).total_seconds()
        progress_bar.update(elapsed_time - progress_bar.n)
        if elapsed_time >= time_limit.total_seconds():
            progress_bar.close()
            print("Time limit exceeded, stopping optimization.")
            study.stop()
        wandb.log({'elapsed_time': elapsed_time, 'trial_number': trial.number, 'best_value': study.best_value})

    n_jobs = int(input("Enter the number of parallel jobs: ").strip())
    try:
        study = optuna.create_study(
            study_name=study_name, 
            direction="minimize", 
            storage=f"sqlite:///{os.path.join(STORAGE_BASE_PATH, study_name, 'optuna_study.db')}",  # 修正
            load_if_exists=True,
            sampler=TPESampler(),
            pruner=HyperbandPruner(min_resource=1, max_resource="auto")
        )
        study.optimize(
            lambda trial: objective(trial, architecture, training_history.best_loss, ENCODE_DIR_PATH, lambda trial_number: create_trial_folder(MODEL_SAVE_BASE_PATH, trial_number)[1], study, study_name),
            timeout=time_limit.total_seconds(), 
            n_jobs=n_jobs, 
            callbacks=[callback]
        )



        output_dir = os.path.join(STORAGE_BASE_PATH, study_name)
        output_path = os.path.join(output_dir, "best_para.json")

        with open(output_path, 'r') as f:
            best_trial_data = json.load(f)

        best_params = best_trial_data["params"]
        epochs = best_params["epochs"]
        batch_size = best_params["batch_size"]
        learning_rate = best_params["learning_rate"]
        seq_length = 30

        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            model = create_model(architecture, best_params, model_architecture_func, seq_length, len(tokens))

        model_path = os.path.join(MODEL_SAVE_BASE_PATH, "best_model.h5")
        plot_path = os.path.join(MODEL_SAVE_BASE_PATH, "training_history.png")

        all_input_sequences, all_target_tokens = generate_training_data(ENCODE_DIR_PATH, seq_length, num_files=10)
        
        history, dataset_size = train_model(
            model, 
            all_input_sequences, 
            all_target_tokens, 
            epochs=epochs, 
            batch_size=batch_size, 
            model_path=model_path, 
            num_files=10, 
            learning_rate=learning_rate, 
            architecture=architecture, 
            model_architecture_func=model_architecture_func,
            callbacks=[WandbCallback(), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), training_history],
            generate_data_func=lambda: generate_training_data(ENCODE_DIR_PATH, seq_length, num_files=10)  # 追加
        )

        if history:
            plot_training_history(history, save_path=plot_path, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, num_files=10, dataset_size=dataset_size)

        model_size = os.path.getsize(model_path) / (1024 * 1024)
        model_params = model.count_params()

        metadata = {
            "epochs": epochs,
            "batch_size": batch_size,
            "num_files": 10,
            "learning_rate": learning_rate,
            "dataset_size": dataset_size,
            "model_size_MB": model_size,
            "model_params": model_params,
            "model_architecture": model_architecture_func.__name__,
            "original_folder": MODEL_SAVE_BASE_PATH
        }
        save_metadata(model_path, metadata)
        save_best_model_info(model_path, metadata, MODEL_SAVE_BASE_PATH)
        clean_up_files(MODEL_SAVE_BASE_PATH, keep_files=['best_model.h5', 'training_history.png', 'optuna_study.db', 'best_model_info.json'])

        print(f"Training finished.")
        print(f"Model size: {model_size:.2f} MB")
        print(f"Model parameters: {model_params}")

    except Exception as e:
        print(f"An exception occurred during optimization: {e}")
    finally:
        progress_bar.close()
        wandb.finish()

if __name__ == "__main__":
    main()
