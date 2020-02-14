# -*- coding: utf-8 -*-
import os
import copy
import json
import torch
import random
import _jsonnet
import argparse
import numpy as np

from src import utils
from src.models.model import IRNet

if __name__ == "__main__":
    # Parse arguments
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

    train_datas, val_datas, table_data = utils.load_dataset(H_PARAMS=H_PARAMS)

    if args.cuda != -1:
        torch.cuda.set_device(args.cuda)

    model = IRNet(H_PARAMS, is_qgm=H_PARAMS["is_qgm"], is_cuda=args.cuda != -1)

    if args.cuda != -1:
        torch.cuda.set_device(args.cuda)
        model.cuda()

    # Load trained wieghts
    print("load pretrained model from {}".format(args.load_model))
    pretrained_model = torch.load(
        args.load_model, map_location=lambda storage, loc: storage
    )
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]
    model.load_state_dict(pretrained_modeled)

    # Load word embedding
    model.word_emb = (
        utils.load_word_emb(H_PARAMS["glove_embed_path"])
        if H_PARAMS["bert"] == -1
        else None
    )

    # Evaluation
    print("Evaluation:")

    (
        train_total_acc,
        train_is_correct,
        train_pred,
        train_gold,
        train_list,
    ) = utils.epoch_acc(
        model,
        H_PARAMS["batch_size"],
        train_datas[0],
        table_data,
        is_transformer=H_PARAMS["is_transformer"],
        is_qgm=H_PARAMS["is_qgm"],
        return_details=True,
    )

    dev_total_acc, dev_is_correct, dev_pred, dev_gold, dev_list = utils.epoch_acc(
        model,
        H_PARAMS["batch_size"],
        val_datas[0],
        table_data,
        is_transformer=H_PARAMS["is_transformer"],
        is_qgm=H_PARAMS["is_qgm"],
        return_details=True,
    )

    print("Train Acc: {}".format(train_total_acc["total"]))
    print("Dev Acc: {}".format(dev_total_acc["total"]))

    # Log path
    log_path = os.path.join(H_PARAMS["log_path"], H_PARAMS["log_key"])

    # Create path
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    train_out_path = os.path.join(log_path, "train.result")
    dev_out_path = os.path.join(log_path, "dev.result")

    use_col_set = not H_PARAMS['is_qgm']

    # Save outputs from train
    assert len(train_pred) == len(train_gold) and len(train_gold) == len(train_list)
    if H_PARAMS["is_transformer"]:
        # Format pred
        tmp = []
        for pred in train_pred:
            tmp += [' '.join(["{}({})".format(*item) for item in pred])]
        train_pred = tmp

    utils.write_eval_result_as(
        train_out_path,
        train_list,
        train_is_correct,
        train_total_acc,
        train_pred,
        train_gold,
        use_col_set=use_col_set
    )

    # Save outputs from dev
    assert len(dev_pred) == len(dev_gold) and len(dev_gold) == len(dev_list)
    if H_PARAMS["is_transformer"]:
        # Format pred
        tmp = []
        for pred in dev_pred:
            tmp += [' '.join(["{}({})".format(*item) for item in pred])]
        dev_pred = tmp
    utils.write_eval_result_as(
        dev_out_path, dev_list, dev_is_correct, dev_total_acc, dev_pred, dev_gold, use_col_set=use_col_set
    )
