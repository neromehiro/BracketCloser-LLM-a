# modules/setup.py
import os
from datetime import timedelta
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model,
    "lstm": define_lstm_model,
    "bert": define_bert_model,
    "gpt": define_gpt_model
}

def setup(architecture_name):
    if architecture_name in MODEL_ARCHITECTURES:
        model_architecture_func = MODEL_ARCHITECTURES[architecture_name]
        return model_architecture_func, architecture_name
    else:
        raise ValueError(f"Unsupported architecture: {architecture_name}")

def parse_time_limit(time_limit_str):
    """時間制限の文字列をtimedeltaに変換する"""
    if 'min' in time_limit_str:
        minutes = int(time_limit_str.replace('min', '').strip())
        return timedelta(minutes=minutes)
    elif 'hour' in time_limit_str:
        hours = int(time_limit_str.replace('hour', '').strip())
        return timedelta(hours=hours)
    else:
        raise ValueError("Unsupported time limit format. Use 'min' or 'hour'.")
