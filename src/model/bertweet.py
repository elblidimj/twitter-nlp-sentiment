import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)


class OptimizedConfig:
    """Optimized configuration for big GPU"""

    TRAIN_POS = 'train_pos_full.txt'
    TRAIN_NEG = 'train_neg_full.txt'
    TEST_DATA = 'test_data.txt'

    MODEL_NAME = 'vinai/bertweet-base'
    MAX_LENGTH = 128
    NUM_LABELS = 2

    BATCH_SIZE = 128
    GRADIENT_ACCUMULATION = 2
    LEARNING_RATE = 3e-5
    EPOCHS = 3
    WARMUP_RATIO = 0.06
    WEIGHT_DECAY = 0.01
    VAL_SPLIT = 0.02

    MAX_GRAD_NORM = 1.0

    MODEL_SAVE_PATH = 'drive/MyDrive/bertweet_optimized.pth'
    SUBMISSION_PATH = 'submission.csv'
    LOG_CSV_PATH = 'drive/MyDrive/training_logs.csv'


def build_model(device):
    model = AutoModelForSequenceClassification.from_pretrained(
        'vinai/bertweet-base',
        num_labels=2
    )

    model.to(device)
    return model