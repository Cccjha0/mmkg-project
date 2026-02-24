# src/config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "OpenBG-IMG")
IMG_DIR = os.path.join(DATA_ROOT, "images")

ENTITY2TEXT_PATH = os.path.join(DATA_ROOT, "entity2text.tsv")
TRAIN_PATH = os.path.join(DATA_ROOT, "train.tsv")
VALID_PATH = os.path.join(DATA_ROOT, "dev.tsv")   # 或 valid.tsv
TEST_PATH  = os.path.join(DATA_ROOT, "test.tsv")

# 其他超参...
BATCH_SIZE = 128
EPOCHS = 50