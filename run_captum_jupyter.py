import os
import json
import copy
import random
import _jsonnet
import argparse
import datetime
import itertools

import torch
import torch.optim as optim
import numpy as np

from src import utils
from src.models.model import IRNet
from captum_utils import view_captum
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

load_model = "logs/semql_simple_multi_spider/model/best_model.pt"
config_file = "train_config/tmp/semql_simple_multi.jsonnet"
# Load Training Info
H_PARAMS = json.loads(_jsonnet.evaluate_file(config_file))

# Set random seed
torch.manual_seed(H_PARAMS["seed"])
torch.cuda.manual_seed(H_PARAMS["seed"])
np.random.seed(H_PARAMS["seed"])
random.seed(H_PARAMS["seed"])

# Load dataset
train_datas, val_datas, table_data = utils.load_dataset(H_PARAMS, use_small=False)

# Set model
torch.cuda.set_device(0)
model = IRNet(H_PARAMS, is_qgm=H_PARAMS["is_qgm"], is_cuda=True)
model.cuda()


print("load pretrained model from {}".format(load_model))
pretrained_model = torch.load(load_model, map_location=lambda storage, loc: storage,)
pretrained_modeled = copy.deepcopy(pretrained_model)
for k in pretrained_model.keys():
    if k not in model.state_dict().keys():
        del pretrained_modeled[k]

model.load_state_dict(pretrained_modeled)

model.word_emb = (
    utils.load_word_emb(H_PARAMS["glove_embed_path"])
    if H_PARAMS["bert"] == -1
    else None
)


for data in train_datas:
    view_captum(
        model, train_datas[0], table_data, is_bert=False, is_qgm=H_PARAMS["is_qgm"]
    )

for data in val_datas:
    view_captum(
        model, val_datas[0], table_data, is_bert=False, is_qgm=H_PARAMS["is_qgm"]
    )
