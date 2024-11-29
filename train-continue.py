# train-continue.py

import os
import sys
import json
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from train import Config, DatasetHandler, ModelTrainer
from modules.custom_layers import CustomMultiHeadAttention
from modules.evaluate import main as evaluate_main

class ContinueModelTrainer(ModelTrainer):
    def __init__(self):
        # 必要な初期化を行う前に、super().__init__()を呼び出さない

        # JSONファイルが存在する最新のモデルディレクトリを取得
        model_dirs = self.get_model_directories(Config.model_save_path)

        if not model_dirs:
            print("有効なモデルディレクトリが見つかりません。プログラムを終了します。")
            sys.exit(1)

        selected_model_dir = self.select_model_directory(model_dirs)
        self.continue_model_dir = os.path.join(Config.model_save_path, selected_model_dir)

        # 追加のエポック数を入力
        additional_epochs = self.input_additional_epochs()

        # 前回の学習情報を読み込み
        training_info = self.load_training_info(self.continue_model_dir)

        # モデルアーキテクチャと累計エポック数の設定
        self.architecture = training_info.get('model_architecture', 'default_architecture')
        self.previous_epochs = training_info.get("epoch", 0)
        self.training_mode = training_info.copy()
        self.training_mode['epochs'] = self.previous_epochs + additional_epochs  # 累計エポック数として保持

        print(f"前回までに終了しているエポック数: {self.previous_epochs}")

        # モデルパラメータとパス設定
        self.num_samples = training_info.get('dataset_size', 1000)
        self.max_seq_length = training_info.get('max_seq_length', 512)
        self.vocab_size = training_info.get('vocab_size', 30522)
        self.learning_rate = training_info.get('learning_rate', 0.001)
        self.batch_size = training_info.get('batch_size', 32)
        self.num_files = training_info.get('num_files', 1)
        self.start_time = datetime.now(Config.japan_timezone)

        # スーパークラスの__init__を必要な引数とともに呼び出す
        super().__init__(self.architecture, self.training_mode, self.num_samples)

        # モデルとプロットデータのロード
        self.model = self.load_existing_model(self.continue_model_dir)
        self.plot_data = self.load_existing_plot_data(self.continue_model_dir)

    def load_existing_plot_data(self, directory):
        """プロットデータを既存ファイルから読み込む（train.pyに実装）"""
        return super().load_plot_data(directory)

    def load_existing_model(self, directory):
        """既存モデルをファイルからロード（train.pyに実装）"""
        custom_objects = {'CustomMultiHeadAttention': CustomMultiHeadAttention}
        return super().load_model_from_path(directory, custom_objects)

    def train(self):
        """トレーニングロジックを実装"""
        # 既存のプロットデータと履歴をロード
        complete_accuracies = self.plot_data.get('complete_accuracy', [])
        partial_accuracies = self.plot_data.get('partial_accuracy', [])
        full_history = self.initialize_full_history(self.plot_data)
        previous_accuracy = None

        initial_metadata = self.load_training_info(self.continue_model_dir)

        for epoch in range(1, self.training_mode["epochs"] - self.previous_epochs + 1):
            current_epoch = self.previous_epochs + epoch
            print(f"Starting epoch {current_epoch} (累計)")

            # データセットを生成しロード
            self.generate_and_load_data(self.temp_save_dir, self.num_samples)

            if not self.dataset_ready():
                print("No data for training.")
                return

            # モデルの学習（エポックごとに1エポックのみ）
            history, _ = self.train_single_epoch(
                self.model, self.loaded_data, epochs=1,
                batch_size=self.training_mode["batch_size"], model_path=self.model_path,
                learning_rate=self.learning_rate
            )

            # 学習履歴を更新
            self.update_history(full_history, history)

            # 精度を評価し再試行の判定
            current_accuracy = self.evaluate_current_accuracy(history, previous_accuracy)
            if self.retry_epoch_needed(current_accuracy, previous_accuracy):
                print("Accuracy decreased significantly, retrying epoch.")
                continue
            previous_accuracy = current_accuracy

            # エポック終了後の精度評価とプロットの更新
            complete_accuracy, partial_accuracy = evaluate_main(self.model_path, current_epoch)
            complete_accuracies.append(complete_accuracy)
            partial_accuracies.append(partial_accuracy)

            # プロット更新
            self.update_plot(full_history, complete_accuracies, partial_accuracies, initial_metadata)
            self.save_plot_data(self.continue_model_dir, full_history, complete_accuracies, partial_accuracies)

            # `training_info.json` の更新
            self.save_training_info(current_epoch, self.continue_model_dir)

        # 最終的なトレーニング情報の保存
        self.finalize_training(complete_accuracies, partial_accuracies)

if __name__ == "__main__":
    trainer = ContinueModelTrainer()
    trainer.train()
