# modules/utils.py
import os
from datetime import datetime
import pytz

def create_save_folder(model_save_base_path, architecture):
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
