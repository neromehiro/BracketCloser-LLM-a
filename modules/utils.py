# modules/utils.py
import os
from datetime import datetime
import pytz

def create_save_folder(model_save_base_path, architecture):
    japan_timezone = pytz.timezone("Asia/Tokyo")
    now = datetime.now(japan_timezone)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # ベースフォルダ名
    base_folder_name = f"hyper_{architecture}"
    
    # 既存のフォルダをチェックして連番を追加
    existing_folders = [f for f in os.listdir(model_save_base_path) if f.startswith(base_folder_name)]
    folder_index = len(existing_folders) + 1
    folder_name = f"{base_folder_name}_{folder_index}"
    
    save_path = os.path.join(model_save_base_path, folder_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path
