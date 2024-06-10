import sys
from modules.evaluate import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <model_save_path>")
    else:
        model_save_path = sys.argv[1]
        main(model_save_path)
