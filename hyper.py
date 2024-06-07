import os
import json
import random
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

# グローバル変数BEST_LOSSを設定
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
    embedding_dim = best_params["embedding_dim"]
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]

    if architecture == "gru":
        return model_architecture_func(seq_length, vocab_size + 1, learning_rate, embedding_dim, best_params["gru_units"], dropout_rate, best_params["recurrent_dropout_rate"])
    elif architecture == "transformer":
        return model_architecture_func(seq_length, vocab_size + 1, learning_rate, embedding_dim, best_params["num_heads"], best_params["ffn_units"], dropout_rate)
    elif architecture == "lstm":
        return model_architecture_func(seq_length, vocab_size + 1, learning_rate, embedding_dim, best_params["lstm_units"], dropout_rate, best_params["recurrent_dropout_rate"], best_params["num_layers"])
    elif architecture == "bert":
        return model_architecture_func(seq_length, vocab_size + 1, learning_rate, embedding_dim, best_params["num_heads"], best_params["ffn_units"], best_params["num_layers"], dropout_rate)
    elif architecture == "gpt":
        return model_architecture_func(seq_length, vocab_size + 1, learning_rate, embedding_dim, best_params["num_heads"], best_params["ffn_units"], dropout_rate)

def load_training_data(encode_dir_path, seq_length, num_files=10):
    all_input_sequences = []
    all_target_tokens = []

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

def generate_datasets(base_dir, num_samples):
    dataset_sizes = [100, 300, 500, 800, 1000, 3000, 5000, 10000]
    if num_samples not in dataset_sizes:
        raise ValueError(f"Invalid number of samples. Choose from {dataset_sizes}")
    
    num_datasets = 1  # You can modify this if you need multiple datasets
    optuna_data_generator.create_datasets(base_dir, num_datasets, num_samples)


def main():
    global STORAGE_BASE_PATH, ENCODE_DIR_PATH, MODEL_SAVE_BASE_PATH

    wandb_api_key = os.getenv('WANDB_API_KEY')
    project_name = os.getenv('WANDB_PROJECT_NAME')
    run_name = os.getenv('WANDB_RUN_NAME')

    if not wandb_api_key or not project_name or not run_name:
        raise ValueError("WANDB_API_KEY, WANDB_PROJECT_NAME, and WANDB_RUN_NAME must be set as environment variables.")

    wandb.login(key=wandb_api_key)
    wandb.init(project=project_name, name=run_name)

    option = input("Choose an option:\n1. Resume existing study\n2. Start a new study\nEnter 1 or 2: ").strip()
    
    if option == "1":
        studies = [f for f in os.listdir(STORAGE_BASE_PATH) if f.startswith("hyper_")]
        if not studies:
            print("No existing studies found. Starting a new study.")
            option = "2"
        else:
            for i, study_folder in enumerate(studies):
                print(f"{i + 1}. {study_folder}")
            study_index = int(input("Enter the number of the study to resume: ").strip()) - 1
            study_folder = studies[study_index]
            study_name = study_folder
            architecture_name = study_folder.split('_')[1]
            storage_name = f"sqlite:///{STORAGE_BASE_PATH}/{study_folder}/optuna_study.db"
            model_architecture_func, architecture = setup(architecture_name)
            save_path = os.path.join(STORAGE_BASE_PATH, study_folder)
            metadata_path = os.path.join(save_path, 'best_model_metadata.json')
            training_history = TrainingHistory(model_path=os.path.join(save_path, 'best_model.h5'), model_architecture_func=model_architecture_func)
            training_history.load_best_loss(metadata_path)

            ENCODE_DIR_PATH = os.path.join(save_path, "dataset/preprocessed/")
            MODEL_SAVE_BASE_PATH = save_path
            STORAGE_BASE_PATH = save_path

    if option == "2":
        architecture_name = input("Enter the model architecture (gru, transformer, lstm, bert, gpt): ").strip()
        model_architecture_func, architecture = setup(architecture_name)
        save_path = create_save_folder(STORAGE_BASE_PATH, architecture_name)
        study_name = os.path.basename(save_path)
        storage_name = f"sqlite:///{save_path}/optuna_study.db"
        training_history = TrainingHistory(model_path=os.path.join(save_path, 'best_model.h5'), model_architecture_func=model_architecture_func)
        
        ENCODE_DIR_PATH = os.path.join(save_path, "dataset/preprocessed/")
        MODEL_SAVE_BASE_PATH = save_path
        STORAGE_BASE_PATH = save_path
        
        num_samples = int(input("Enter the number of samples for the dataset (100, 300, 500, 800, 1000, 3000, 5000, 10000): ").strip())
        generate_datasets(save_path, num_samples)

    time_limit_str = input("Enter the training time limit (e.g., '3min', '1hour', '5hour'): ").strip()
    time_limit = parse_time_limit(time_limit_str)
    start_time = (datetime.now() + timedelta(hours=9))

    progress_bar = tqdm(total=time_limit.total_seconds(), desc="Optimization Progress", unit="s")

    def callback(study, trial):
        current_time = datetime.now() + timedelta(hours=9)
        elapsed_time = (current_time - start_time).total_seconds()
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
            storage=storage_name, 
            load_if_exists=True,
            sampler=TPESampler(),
            pruner=HyperbandPruner(min_resource=1, max_resource="auto")
        )
        study.optimize(
            lambda trial: objective(trial, architecture, training_history.best_loss, ENCODE_DIR_PATH, lambda: create_save_folder(save_path, architecture), study, study_name),
            timeout=time_limit.total_seconds(), 
            n_jobs=n_jobs, 
            callbacks=[callback]
        )
    except Exception as e:
        print(f"An exception occurred during optimization: {e}")
    finally:
        progress_bar.close()

    save_best_trial_to_json(study, study_name)
    output_dir = os.path.join("optuna_studies", study_name)
    output_path = os.path.join(output_dir, "best_para.json")

    with open(output_path, 'r') as f:
        best_trial_data = json.load(f)

    best_params = best_trial_data["params"]
    epochs = best_params["epochs"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    seq_length = 30

    model = create_model(architecture, best_params, model_architecture_func, seq_length, len(tokens))

    all_input_sequences, all_target_tokens = load_training_data(ENCODE_DIR_PATH, seq_length)

    if not all_input_sequences or not all_target_tokens:
        print("No data for training.")
        return

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = create_model(architecture, best_params, model_architecture_func, seq_length, len(tokens))

    model_path = os.path.join(save_path, "best_model.h5")
    plot_path = os.path.join(save_path, "training_history.png")

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
        callbacks=[WandbCallback(), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), training_history]
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
        "original_folder": save_path
    }
    save_metadata(model_path, metadata)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_best_model_info(model_path, metadata, save_path)

    clean_up_files(save_path, keep_files=['best_model.h5', 'training_history.png', 'optuna_study.db', 'best_model_info.json'])

    print(f"Training finished.")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Model parameters: {model_params}")

    wandb.finish()


if __name__ == "__main__":
    main()
