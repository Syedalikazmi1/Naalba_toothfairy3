

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATASET_PATH = (
    "/content/drive/MyDrive/"
    "NeBLa_ToothFairy3_Project/"
    "data/training_dataset/"
    "final_nebla_dataset"
)


CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "NeBLa_ToothFairy3_Project/"
    "results/checkpoints"
)


EPOCHS = 10

BATCH_RAYS = 128

HIDDEN_DIM = 64

LR = 1e-3

