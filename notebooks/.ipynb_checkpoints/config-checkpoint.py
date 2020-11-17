import os
import glob
import codecs
import csv
import pandas
import logging
import math
import numpy
import pickle
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm, tqdm_notebook, trange
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from transformers import (BertConfig, BertTokenizer, BertForSequenceClassification, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from sklearn.metrics import f1_score


train_articles = "/workspace/PTC/datasets/train-articles" 
test_articles = "/workspace/PTC/datasets/test-articles"
dev_articles = "/workspace/PTC/datasets/dev-articles"
train_SI_labels = "/workspace/PTC/datasets/train-labels-task-si"
train_TC_labels = "/workspace/PTC/datasets/train-labels-task-flc-tc"
dev_SI_labels = "/workspace/PTC/datasets/dev-labels-task-si"
dev_TC_labels = "/workspace/PTC/datasets/dev-labels-task-si/dev-labels-task-flc-tc"
# dev_TC_labels_file = "PTC/datasets/dev-task-TC.labels" # Multiple files
dev_TC_template = "/workspace/PTC/datasets/test-task-tc-template.out" # only .out file
techniques = "/workspace/PTC/tools/data/propaganda-techniques-names-semeval2020task11.txt"
PROP_TECH_TO_LABEL = {}
LABEL_TO_PROP_TECH = {}
label = 0
with open(techniques, "r") as f:
    for technique in f:
        PROP_TECH_TO_LABEL[technique.replace("\n", "")] = int(label)
        LABEL_TO_PROP_TECH[int(label)] = technique.replace("\n", "")
        label += 1
device = torch.device("cuda")
# device = torch.device("cpu")
n_gpu = torch.cuda.device_count()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LOG")
MODEL_CLASSES = {"bert": (BertConfig, BertForSequenceClassification, BertTokenizer)}
args = {"data_dir": "/workspace/PTC/datasets/",
        "model_type": "bert",
        "model_name": "bert-base-uncased",
        "output_dir": '/workspace/PTC/datasets/output/SI_output',
        "max_seq_length": 128,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_train_epochs": 1,
        "weight_decay": 0,
        "learning_rate": 4e-5,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "logging_steps": 50,
        "save_steps": 2000,
        "overwrite_output_dir": False}
