# src/config.py
import torch
from pathlib import Path

class Config:
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RESULTS_DIR = ROOT_DIR / "results"
    MODEL_DIR = RESULTS_DIR / "models"
    LOG_DIR = RESULTS_DIR / "logs"

    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MAX_TFIDF_FEATURES = 5000
    TFIDF_NGRAM = (3, 3)

    PCA_COMPONENTS = 128

    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LR = 0.005
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 500
    PATIENCE = 30

    VAL_RATIO = 0.10
    TEST_RATIO = 0.20

    NUM_CLASSES = 6

    EPSILON_FGSM = 0.1
    EPSILON_PGD = 0.01
    ALPHA_PGD = 0.001
    PGD_STEPS = 40

    PRINT_EVERY = 50

