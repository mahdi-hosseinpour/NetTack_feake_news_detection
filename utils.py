# utils.py
import torch
import numpy as np
import random
import os
from pathlib import Path
from src.config import Config

def set_seed(seed=Config.SEED):
    """
    Set random seed for reproducibility across numpy, torch, and python random
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def save_model(model, path=Config.MODEL_DIR / "graphsage_best.pth"):
    """
    Save the model state dict
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path=Config.MODEL_DIR / "graphsage_best.pth"):
    """
    Load model state dict
    """
    model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
    model.eval()
    print(f"Model loaded from {path}")
    return model