import os
import sys
import json
import numpy as np
from datetime import datetime
import pytz
from modules.data_utils import load_dataset, tokens, token2id
from modules.model_utils import (define_gru_model, define_transformer_model,
                                 define_lstm_model, define_bert_model, define_gpt_model)
from modules.training_utils import train_model_single, plot_training_history
from modules.custom_layers import CustomMultiHeadAttention
from tensorflow.keras.preprocessing.sequence import pad_sequences
import optuna_data_generator
from tensorflow.keras.optimizers import Adam
from modules.evaluate import main as evaluate_main


class Config:
    japan_timezone = pytz.timezone("Asia/Tokyo")
    dataset_base_dir = "./datasets/"
    model_save_path = "./models/"
    
    MODEL_ARCHITECTURES = {
        "gru": define_gru_model,
        "transformer": define_transformer_model,
        "lstm": define_lstm_model,
        "bert": define_bert_model,
        "gpt": define_gpt_model
    }
    
    SHORTCUTS = {
        "gru": "gru",
        "tra": "transformer",
        "lstm": "lstm",
        "ber": "bert",
        "gpt": "gpt"
    }
    
    TRAINING_MODES = {
        "1min": {"epochs": 1, "batch_size": 128, "num_files": 1, "learning_rate": 0.01},
        "10min": {"epochs": 3, "batch_size": 256, "num_files": 1, "learning_rate": 0.01},
        "1hour": {"epochs": 7, "batch_size": 512, "num_files": 1, "learning_rate": 0.001},
        "6hours": {"epochs": 20, "batch_size": 1024, "num_files": 1, "learning_rate": 0.001},
        "12hours": {"epochs": 40, "batch_size": 1024, "num_files": 1, "learning_rate": 0.001},
        "24hours": {"epochs": 80, "batch_size": 1024, "num_files": 1, "learning_rate": 0.0005},
        "2days": {"epochs": 160, "batch_size": 1024, "num_files": 1, "learning_rate": 0.0005},
        "4days": {"epochs": 320, "batch_size": 1024, "num_files": 1, "learning_rate": 0.0005},
        # "op": { # transformer用
        #     "learning_rate": 0.00023979260599979734,
        #     "batch_size": 1024,
        #     "regularizer_type": "l2",
        #     "regularizer_value": 1.0703148054547978e-06,
        #     "embedding_dim": 218,
        #     "num_heads": 3,
        #     "ffn_units": 202,
        #     "dropout_rate": 0.16009354834601996,
        #     "epochs": 300,
        #     "num_files": 1
        # }
        # "op": { # transformer 複雑度を上げた
        #     "learning_rate": 0.0001,  # 学習率を小さくして微細な調整を可能に
        #     "batch_size": 1024,       # バッチサイズを減らして勾配更新頻度を上げる
        #     "regularizer_type": "l2",
        #     "regularizer_value": 1.0e-07,  # 正則化を弱め、複雑なモデルを許容
        #     "embedding_dim": 512,    # 埋め込み次元を増やし、表現力を向上
        #     "num_heads": 8,          # Attentionヘッド数を増やして情報抽出能力を強化
        #     "ffn_units": 1024,       # フィードフォワードネットワークのユニット数を増やす
        #     "dropout_rate": 0.1,     # ドロップアウト率を下げて学習を促進
        #     "epochs": 500,           # エポック数を増やし、十分な学習を確保
        #     "num_files": 1           # 変更なし
        # }

        "op": { # gru用 optuna
            "learning_rate": 0.0001,
            "batch_size": 32,
            "regularizer_type": "l2",
            "regularizer_value": 1.2526981458684705e-06,
            "embedding_dim": 94,
            "gru_units": 177,
            "dropout_rate": 0.2330572493663566,
            "recurrent_dropout_rate": 0.1878114654182462,
            "num_files": 1, 
            "epochs": 300
        },
        # "op": { # gru用 複雑度を上げた
        #     "learning_rate": 0.0001,
        #     "batch_size": 32,
        #     "regularizer_type": "l2",
        #     "regularizer_value": 1.2526981458684701e-07, # e-6→e-7
        #     "embedding_dim": 256, # 94→256
        #     "gru_units": 177,
        #     "dropout_rate": 0.2330572493663566,
        #     "recurrent_dropout_rate": 0.1878114654182462,
        #     "num_files": 1, 
        #     "epochs": 300
        # },
    }
    
    @staticmethod
    def select_mode_and_architecture():
        modes = list(Config.TRAINING_MODES.keys())
        architectures = list(Config.SHORTCUTS.keys())
        choices = [f"{arch} {mode}" for arch in architectures for mode in modes]

        print("以下のモードとアーキテクチャから選んでください。選択肢は英語のまま入力してください：\n")
        
        for arch in architectures:
            print(f"{arch.upper()} アーキテクチャ:")
            for mode in modes:
                print(f"    - {mode}: {arch} {mode}")
                
        choice = input("\nあなたの選択: ")

        while choice not in choices:
            print("無効な選択です。もう一度選択してください。")
            choice = input("アーキテクチャとモード (例: gru 1min): ")

        arch, mode = choice.split()
        architecture = Config.SHORTCUTS[arch]
        return Config.MODEL_ARCHITECTURES[architecture], Config.TRAINING_MODES[mode], architecture


class DatasetHandler:
    @staticmethod
    def generate_datasets(base_dir, num_samples, learning_stage=None):
        print(f"Generating dataset with {num_samples} samples.")
        optuna_data_generator.create_datasets(base_dir, 1, num_samples, learning_stage=learning_stage)

    @staticmethod
    def load_encoded_tokens(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def prepare_sequences(encoded_tokens, seq_length):
        # 自己回帰LM: 入力=全系列、ターゲット=1トークン右シフト
        trimmed = [int(t) for t in encoded_tokens if t != 0]
        if len(trimmed) < 2:
            return np.empty((0, seq_length)), np.empty((0, seq_length)), np.empty((0, seq_length))
        decoder_input = trimmed[:-1]
        targets = trimmed[1:]
        input_sequences = pad_sequences([decoder_input], maxlen=seq_length, padding='post', value=0)[0]
        target_tokens = pad_sequences([targets], maxlen=seq_length, padding='post', value=0)[0]
        sample_weights = (target_tokens != 0).astype(np.float32)  # PADは損失無視
        return np.array([input_sequences]), np.array([target_tokens]), np.array([sample_weights], dtype=np.float32)


class ModelTrainer:
    def __init__(self, architecture, training_mode, num_samples):
        self.architecture = architecture
        self.training_mode = training_mode
        self.num_samples = num_samples
        self.start_time = datetime.now(Config.japan_timezone)
        self.max_seq_length = 30
        self.vocab_size = len(set(tokens)) + 1
        self.model = self._initialize_model()
        self.temp_save_dir = self._create_temp_save_dir()
        self.model_path = os.path.join(self.temp_save_dir, "best_model.h5")
        self.history = []
        self.training_info_path = os.path.join(self.temp_save_dir, "training_info.json")
        self.plot_data_path = os.path.join(self.temp_save_dir, "plot_data.json")
        self.bracket2_accuracies = []
        self.bracket3_accuracies = []
        self.bracket4plus_accuracies = []
        self.micro_accuracies = []
    

    def _initialize_model(self):
        model_architecture_func = Config.MODEL_ARCHITECTURES[self.architecture]
        params = {k: v for k, v in self.training_mode.items()
                  if k not in ["epochs", "batch_size", "num_files", "learning_rate"]}
        return model_architecture_func(self.max_seq_length, self.vocab_size, self.training_mode["learning_rate"], **params)

    
    def _create_temp_save_dir(self):
        # 学習中の保存先のディレクトリ名を「日時_アーキテクチャ」に変更
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        temp_save_dir = os.path.join(Config.model_save_path, f"{timestamp}_{self.architecture}")
        os.makedirs(temp_save_dir, exist_ok=True)
        return temp_save_dir

    def _finalize_training(self, complete_accuracies, partial_accuracies):
        end_time = datetime.now(Config.japan_timezone)
        training_duration = (end_time - self.start_time).total_seconds() / 60
        
        # 学習後の最終的なディレクトリ名も「日時_アーキテクチャ」に変更
        final_save_dir = os.path.join(Config.model_save_path, f"{self.start_time.strftime('%Y%m%d_%H%M%S')}_{self.architecture}")
        os.rename(self.temp_save_dir, final_save_dir)
        
        model_path = os.path.join(final_save_dir, "best_model.h5")
        metadata = {
            "training_duration_minutes": training_duration,
            "training_end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_size_MB": os.path.getsize(model_path) / (1024 * 1024)
        }
        self._save_metadata(metadata, final_save_dir)
    

    def train(self):
        all_input_sequences = []
        all_target_tokens = []
        all_sample_weights = []
        complete_accuracies = []
        partial_accuracies = []
        previous_accuracy = None
        full_history = []

        for epoch in range(self.training_mode["epochs"]):
            print(f"Starting epoch {epoch + 1}/{self.training_mode['epochs']}")

            # 学習段階の設定
            if 1 <= epoch + 1 <= 4:
                learning_stage = 1
            elif 5 <= epoch + 1 <= 8:
                learning_stage = 2
            else:
                learning_stage = 3

            # データセットの生成
            DatasetHandler.generate_datasets(self.temp_save_dir, self.num_samples, learning_stage)

            # データの読み込み
            all_input_sequences = []
            all_target_tokens = []
            all_sample_weights = []
            self._load_data(all_input_sequences, all_target_tokens, all_sample_weights)

            if not all_input_sequences or not all_target_tokens or not all_sample_weights:
                print("No data for training.")
                return

            all_input_sequences = np.concatenate(all_input_sequences, axis=0)
            all_target_tokens = np.concatenate(all_target_tokens, axis=0)
            all_sample_weights = np.concatenate(all_sample_weights, axis=0)

            # モデルの学習
            history, _ = train_model_single(
                self.model, all_input_sequences, all_target_tokens, epochs=1,
                batch_size=self.training_mode["batch_size"], model_path=self.model_path,
                num_files=self.training_mode["num_files"],
                learning_rate=self.training_mode["learning_rate"], architecture=self.architecture,
                model_architecture_func=Config.MODEL_ARCHITECTURES[self.architecture],
                sample_weights=all_sample_weights
            )

            # accuracyの評価と再試行の判定
            current_accuracy = self._evaluate_accuracy(history, previous_accuracy)
            if current_accuracy is None or (previous_accuracy and current_accuracy <= previous_accuracy - 0.10):
                print("Accuracy decreased by more than 10%, retrying epoch.")
                continue
            previous_accuracy = current_accuracy

            # エポック終了後に精度の評価（2回 / 3回 / 4回以上を個別集計、バッチ＆並列で高速化）
            metrics = evaluate_main(
                model_path=self.model_path,
                epoch_num=epoch + 1,
                num_test_samples=300,
                batch_size=256,
                num_workers=1,  # 単一プロセスでモデル再ロードなし・高速化
                evaluate_single=False,
                eval_bracket_buckets=(2, 3, 4),
                max_decode_steps=10,
            )
            complete_accuracy = metrics.get("bracket_1")  # evaluate_single=False なので実質 None/0
            partial_accuracy = metrics.get("micro")
            complete_accuracies.append(complete_accuracy)
            partial_accuracies.append(partial_accuracy)
            self.bracket2_accuracies.append(metrics.get("bracket_2"))
            self.bracket3_accuracies.append(metrics.get("bracket_3"))
            self.bracket4plus_accuracies.append(metrics.get("bracket_4plus"))
            self.micro_accuracies.append(metrics.get("micro"))

            # 学習履歴を更新
            self._update_full_history(full_history, history)

            # `training_info.json` の更新と保存
            self._save_training_info(epoch)

            # `plot_data.json` の更新と保存
            self._save_plot_data(full_history, complete_accuracies, partial_accuracies)

            # プロットの更新と保存
            self._update_plot(full_history, complete_accuracies, partial_accuracies)

    def _update_plot(self, full_history, complete_accuracies, partial_accuracies):
        """毎エポックでプロットを更新するメソッド"""
        plot_data = {
            'loss': [epoch_data['loss'] for epoch_data in full_history],
            'val_loss': [epoch_data['val_loss'] for epoch_data in full_history],
            'accuracy': [epoch_data['accuracy'] for epoch_data in full_history],
            'val_accuracy': [epoch_data['val_accuracy'] for epoch_data in full_history],
            'complete_accuracy': complete_accuracies,
            'partial_accuracy': partial_accuracies,
            'bracket_2': self.bracket2_accuracies,
            'bracket_3': self.bracket3_accuracies,
            'bracket_4plus': self.bracket4plus_accuracies,
            'micro': self.micro_accuracies
        }

        avg_complete_accuracy = (sum(acc for acc in complete_accuracies if acc is not None) 
                                / len(complete_accuracies)) if complete_accuracies else 0
        avg_partial_accuracy = (sum(acc for acc in partial_accuracies if acc is not None) 
                                / len(partial_accuracies)) if partial_accuracies else 0
        avg_bracket_2 = (sum(acc for acc in self.bracket2_accuracies if acc is not None)
                         / len(self.bracket2_accuracies)) if self.bracket2_accuracies else 0
        avg_bracket_3 = (sum(acc for acc in self.bracket3_accuracies if acc is not None)
                         / len(self.bracket3_accuracies)) if self.bracket3_accuracies else 0
        avg_bracket_4plus = (sum(acc for acc in self.bracket4plus_accuracies if acc is not None)
                             / len(self.bracket4plus_accuracies)) if self.bracket4plus_accuracies else 0
        avg_micro = (sum(acc for acc in self.micro_accuracies if acc is not None)
                     / len(self.micro_accuracies)) if self.micro_accuracies else 0
        dataset_size = len(plot_data['loss'])  # データサイズの推定

        plot_training_history(
            plot_data,
            save_path=os.path.join(self.temp_save_dir, "training_history.png"),
            epochs=self.training_mode["epochs"],
            batch_size=self.training_mode["batch_size"],
            learning_rate=self.training_mode["learning_rate"],
            num_files=self.training_mode["num_files"],
            dataset_size=dataset_size,
            avg_complete_accuracy=avg_complete_accuracy,
            avg_partial_accuracy=avg_partial_accuracy,
            initial_metadata={},
            bracket_metrics={
                "bracket_2": self.bracket2_accuracies,
                "bracket_3": self.bracket3_accuracies,
                "bracket_4plus": self.bracket4plus_accuracies,
                "micro": self.micro_accuracies,
                "avg_bracket_2": avg_bracket_2,
                "avg_bracket_3": avg_bracket_3,
                "avg_bracket_4plus": avg_bracket_4plus,
                "avg_micro": avg_micro,
            }
        )


    def input_additional_epochs(self):
        """追加のエポック数を入力するメソッド"""
        try:
            additional_epochs = int(input("追加で何epoch学習しますか？: "))
            if additional_epochs <= 0:
                raise ValueError
            return additional_epochs
        except ValueError:
            print("無効なエポック数が入力されました。プログラムを終了します。")
            sys.exit(1)

    def select_model_directory(self, model_dirs):
        """モデルディレクトリを選択するメソッド"""
        print("継続学習するモデルを選択してください（番号を入力）：")
        for idx, model_dir in enumerate(model_dirs):
            print(f"{idx + 1}: {model_dir}")
        try:
            selected_idx = int(input("番号: ")) - 1
            if selected_idx < 0 or selected_idx >= len(model_dirs):
                raise ValueError
            return model_dirs[selected_idx]
        except ValueError:
            print("無効な番号が選択されました。プログラムを終了します。")
            sys.exit(1)

    def get_model_directories(self, path):
        """モデルディレクトリを取得するメソッド"""
        return sorted(
            [
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and
                os.path.exists(os.path.join(path, d, "training_info.json"))
            ],
            reverse=True
        )
        
    def _save_training_info(self, epoch):
        # Define metadata for the current (latest) epoch
        training_info = {
            "epoch": epoch + 1,
            "training_duration_minutes": (datetime.now(Config.japan_timezone) - self.start_time).total_seconds() / 60,
            "epochs": self.training_mode["epochs"],
            "batch_size": self.training_mode["batch_size"],
            "num_files": self.training_mode["num_files"],
            "learning_rate": self.training_mode["learning_rate"],
            "dataset_size": self.num_samples,
            "model_size_MB": os.path.getsize(self.model_path) / (1024 * 1024),
            "model_params": self.model.count_params(),
            "model_architecture": self.architecture,
            "training_start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_end_time": datetime.now(Config.japan_timezone).strftime("%Y-%m-%d %H:%M:%S"),
            "max_seq_length": self.max_seq_length,
            "vocab_size": self.vocab_size
        }

        # Save as a flat JSON structure with only the latest epoch's info
        with open(self.training_info_path, "w") as info_file:
            json.dump(training_info, info_file, indent=4)

    def _save_metadata(self, metadata, save_dir):
        # 学習終了後に保存するメタデータ
        metadata.update({
            "epochs": self.training_mode["epochs"],
            "batch_size": self.training_mode["batch_size"],
            "num_files": self.training_mode["num_files"],
            "learning_rate": self.training_mode["learning_rate"],
            "dataset_size": self.num_samples,
            "model_size_MB": os.path.getsize(self.model_path) / (1024 * 1024),
            "model_params": self.model.count_params(),
            "model_architecture": self.architecture,
            "training_start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_end_time": datetime.now(Config.japan_timezone).strftime("%Y-%m-%d %H:%M:%S"),
            "max_seq_length": self.max_seq_length,
            "vocab_size": self.vocab_size
        })
        # ディレクトリの `training_info.json` に保存
        with open(os.path.join(save_dir, "training_info.json"), "w") as info_file:
            json.dump(metadata, info_file, indent=4)

    def _save_plot_data(self, full_history, complete_accuracies, partial_accuracies):
        plot_data = {
            'loss': [epoch_data['loss'] for epoch_data in full_history],
            'val_loss': [epoch_data['val_loss'] for epoch_data in full_history],
            'accuracy': [epoch_data['accuracy'] for epoch_data in full_history],
            'val_accuracy': [epoch_data['val_accuracy'] for epoch_data in full_history],
            'complete_accuracy': complete_accuracies,
            'partial_accuracy': partial_accuracies,
            'bracket_2': self.bracket2_accuracies,
            'bracket_3': self.bracket3_accuracies,
            'bracket_4plus': self.bracket4plus_accuracies,
            'micro': self.micro_accuracies
        }
        with open(self.plot_data_path, "w") as plot_file:
            json.dump(plot_data, plot_file, indent=4)

    def _update_full_history(self, full_history, history):
        for i in range(len(history["loss"])):
            epoch_data = {
                'loss': history["loss"][i],
                'val_loss': history["val_loss"][i] if i < len(history.get("val_loss", [])) else None,
                'accuracy': history["accuracy"][i] if i < len(history.get("accuracy", [])) else None,
                'val_accuracy': history["val_accuracy"][i] if i < len(history.get("val_accuracy", [])) else None
            }
            full_history.append(epoch_data)
            
    def _load_data(self, all_input_sequences, all_target_tokens, all_sample_weights):
        dataset_path = os.path.join(self.temp_save_dir, "dataset", "preprocessed")
        for dirpath, _, filenames in os.walk(dataset_path):
            for file in filenames[:self.training_mode["num_files"]]:
                file_path = os.path.join(dirpath, file)
                encoded_tokens_list = DatasetHandler.load_encoded_tokens(file_path)
                for encoded_tokens in encoded_tokens_list:
                    input_sequences, target_tokens, sample_weights = DatasetHandler.prepare_sequences(encoded_tokens, self.max_seq_length)
                    all_input_sequences.append(input_sequences)
                    all_target_tokens.append(target_tokens)
                    all_sample_weights.append(sample_weights)

    def _evaluate_accuracy(self, history, previous_accuracy):
        current_accuracy = history.get("accuracy")[0] if 'accuracy' in history and len(history["accuracy"]) > 0 else None
        if previous_accuracy is not None and current_accuracy and current_accuracy <= previous_accuracy - 0.10:
            return None
        self.history.append(history)
        return current_accuracy

    def _plot_training_history(self, complete_accuracies, partial_accuracies):
        plot_data = {
            'loss': [epoch['loss'] for epoch in self.history],
            'val_loss': [epoch.get('val_loss') for epoch in self.history],
            'accuracy': [epoch.get('accuracy') for epoch in self.history],
            'val_accuracy': [epoch.get('val_accuracy') for epoch in self.history],
            'complete_accuracy': complete_accuracies,
            'partial_accuracy': partial_accuracies
        }
        # 平均値の計算
        avg_complete_accuracy = (sum(acc for acc in complete_accuracies if acc is not None) 
                                 / len(complete_accuracies)) if complete_accuracies else 0
        avg_partial_accuracy = (sum(acc for acc in partial_accuracies if acc is not None) 
                                / len(partial_accuracies)) if partial_accuracies else 0
        dataset_size = len(plot_data['loss'])  # データサイズの推定

        plot_training_history(
            plot_data,
            save_path=os.path.join(self.temp_save_dir, "training_history.png"),
            epochs=self.training_mode["epochs"],
            batch_size=self.training_mode["batch_size"],
            learning_rate=self.training_mode["learning_rate"],
            num_files=self.training_mode["num_files"],
            dataset_size=dataset_size,
            avg_complete_accuracy=avg_complete_accuracy,
            avg_partial_accuracy=avg_partial_accuracy,
            initial_metadata={}  # 必要であれば、実際のメタデータを渡す
        )



if __name__ == "__main__":
    model_architecture_func, training_mode, architecture = Config.select_mode_and_architecture()
    num_samples = int(input("Enter the number of samples for the dataset (100, 300, 500, 800, 1000, 3000, 5000, 10000): ").strip())
    trainer = ModelTrainer(architecture, training_mode, num_samples)
    trainer.train()
