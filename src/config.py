
import torch


IMG_SIZE=224
PATCH_SIZE=16
HIDDEN_DIM=768
FILTER_SIZE=2048
NUM_HEADS=8
N_LAYERS=6
DROPOUT_RATE=0.1
NUM_CLASSES=10

LEARNING_RATE=1e-4

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")