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


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config", type=str, default="", help="Path for train config json file"
    )

    parser.add_argument("--load_model", type=str, default="", help="saved model path")
    parser.add_argument("--cuda", type=int, default="-1", help="GPU number")
    args = parser.parse_args()

    # Load Training Info
    H_PARAMS = json.loads(_jsonnet.evaluate_file(args.train_config))

    # Set random seed
    torch.manual_seed(H_PARAMS["seed"])
    if args.cuda:
        torch.cuda.manual_seed(H_PARAMS["seed"])
    np.random.seed(H_PARAMS["seed"])
    random.seed(H_PARAMS["seed"])

    # Load dataset
    train_datas, val_datas, table_data = utils.load_dataset(H_PARAMS, use_small=False)

    # Set model
    if args.cuda != -1:
        torch.cuda.set_device(args.cuda)
    model = IRNet(H_PARAMS, is_qgm=H_PARAMS["is_qgm"], is_cuda=args.cuda != -1)
    if args.cuda != -1:
        model.cuda()

    if args.load_model:
        print("load pretrained model from {}".format(args.load_model))
        pretrained_model = torch.load(
            args.load_model, map_location=lambda storage, loc: storage
        )
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

    # Log path
    # log_path = os.path.join(
    #     H_PARAMS["log_path"], "debugging" if args.debugging else H_PARAMS["log_key"]
    # )
    # log_model_path = os.path.join(log_path, "model")
    #
    # # Tensorboard
    # summary_writer = SummaryWriter(log_path)

    for data in train_datas:
        view_captum(
            model, train_datas[0], table_data, is_bert=False, is_qgm=H_PARAMS["is_qgm"]
        )

    for data in val_datas:
        view_captum(
            model, val_datas[0], table_data, is_bert=False, is_qgm=H_PARAMS["is_qgm"]
        )
