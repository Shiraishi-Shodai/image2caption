import kagglehub
import polars as pl
from matplotlib import pyplot as plt
import csv
import cv2
from torch.nn import functional as F
import torch
import glob
import cv2
from torchvision.transforms import v2
from torchvision.io import decode_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
from pathlib import Path
import japanize_matplotlib
import re
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.data import load_yaml
from src.models.model import Encoder
from torch.optim import Adam, SGD
import random
from src.dataset.build import create_dataloaders
from src.dataset.split import split_dataframe_by_image
from src.dataset.transforms import build_transforms
import torch.nn as nn


def main():

#######################
# yamlの読み込み
#######################
    DATASET_YAML = 'dataset.yaml'
    TRAIN_YAML = 'train.yaml'
    MODEL_YAML = 'model.yaml'

    dataset_config = load_yaml(f"./config/{DATASET_YAML}")
    DATA_DIR = dataset_config["DATA_DIR"]
    IMAGES_DIR = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["IMAGES"]}"
    CAPTIONS_CSV = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["CAPTIONS"]}"
    WORD_DICT_CSV = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["WORD_DICT"]}"
    START_WORD = dataset_config["SEPARATOR_KEYS"]["START"]
    END_WORD = dataset_config["SEPARATOR_KEYS"]["END"]
    PAD_WORD = dataset_config["SEPARATOR_KEYS"]["PAD"]

    train_config = load_yaml(f"./config/{TRAIN_YAML}")
    TRAIN_RATIO = train_config["TRAIN_RATIO"]
    VALIDATE_RATIO = train_config["VALIDATE_RATIO"]
    TEST_RATIO = train_config["TEST_RATIO"]
    BATCH_SIZE = train_config["BATCH_SIZE"]
    SHOW_SCORE_INVTERVAL = train_config["SHOW_SCORE_INVTERVAL"]
    MAX_EPOCH = train_config["MAX_EPOCH"]
    LR = train_config["LR"]

    model_config = load_yaml(f"./config/{MODEL_YAML}")

    

#######################
# その他変数定義
#######################
    df = pl.read_csv(CAPTIONS_CSV)
    word_dict_df = pl.read_csv(WORD_DICT_CSV)
    ignore_index = word_dict_df.filter(pl.col("word") == PAD_WORD)["id"].item()

    model = Seq2seq(model_config)
    # optim = SGD(
    #         # encoder.parameters(),
    #         lr=LR
    #     )

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

#######################
# 学習・検証・テストデータの作成
#######################
    
    # train_df, validate_df, test_df = split_dataframe_by_image(df, TRAIN_RATIO, VALIDATE_RATIO)
    # train_transform, eval_transform = build_transforms()
    # train_loader, validate_dataloader, test_loader = create_dataloaders(train_df, validate_df, test_df, IMAGES_DIR, BATCH_SIZE, train_transform, eval_transform)


# #######################
# # 学習
# #######################

#     for i, (images, caption) in enumerate(train_dataloader):
#         Encoder.forward(images)
        
        

if __name__ == "__main__":
    main()