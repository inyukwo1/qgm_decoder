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

    sql_data, table_data, val_sql_data, val_table_data = utils.load_dataset(
        H_PARAMS["data_path"], use_small=H_PARAMS["toy"]
    )

    # Filter data for bert
    if H_PARAMS["bert"] != -1:
        sql_data = [data for data in sql_data if data["db_id"] != "baseball_1"]
        val_sql_data = [data for data in val_sql_data if data["db_id"] != "baseball_1"]

    print("train data length: {}".format(len(sql_data)))
    print("dev data length: {}".format(len(val_sql_data)))

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

    train_total_acc, train_pred, train_gold, train_list = utils.epoch_acc(
        model,
        H_PARAMS["batch_size"],
        sql_data,
        table_data,
        is_qgm=H_PARAMS["is_qgm"],
        return_output=True,
    )

    dev_total_acc, dev_pred, dev_gold, dev_list = utils.epoch_acc(
        model,
        H_PARAMS["batch_size"],
        val_sql_data,
        val_table_data,
        is_qgm=H_PARAMS["is_qgm"],
        return_output=True,
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

    # Save outputs from train
    assert len(train_pred) == len(train_gold) and len(train_gold) == len(train_list)
    with open(train_out_path, "w") as f:
        f.write("Data len: {}\n\n".format(len(train_gold)))
        for key, value in train_total_acc.items():
            f.write("acc {}: {}\n".format(key, value))
        f.write("\n")
        for idx in range(len(train_gold)):
            pred = train_pred[idx]
            gold = train_gold[idx]
            sql_json = train_list[idx].sql_json

            if H_PARAMS["is_qgm"]:
                pred = sorted(pred)
                gold = sorted(gold)

            f.write("db_id: {}\n".format(sql_json["db_id"]))
            f.write("query: {}\n".format(sql_json["query"]))
            f.write("question: {}\n".format(str(sql_json["question_arg"])))
            f.write("que_type: {}\n".format(str(sql_json["question_arg_type"])))
            f.write("table   : {}\n".format(str(sql_json["table_names"])))
            f.write("column  : {}\n".format(str(sql_json["col_set"])))
            f.write("gold: {}\n".format(gold))
            f.write("pred: {}\n".format(pred))
            f.write("\n")

    # Save outputs from dev
    assert len(dev_pred) == len(dev_gold) and len(dev_gold) == len(dev_list)
    with open(dev_out_path, "w") as f:
        f.write("Data len: {}\n\n".format(len(dev_gold)))
        for key, value in dev_total_acc.items():
            f.write("acc {}: {}\n".format(key, value))
        f.write("\n")
        for idx in range(len(dev_gold)):
            pred = dev_pred[idx]
            gold = dev_gold[idx]
            sql_json = dev_list[idx].sql_json

            if H_PARAMS["is_qgm"]:
                pred = sorted(pred)
                gold = sorted(gold)

            f.write("db_id: {}\n".format(sql_json["db_id"]))
            f.write("query: {}\n".format(sql_json["query"]))
            f.write("question: {}\n".format(str(sql_json["question_arg"])))
            f.write("que_type: {}\n".format(str(sql_json["question_arg_type"])))
            f.write("table   : {}\n".format(str(sql_json["table_names"])))
            f.write("column  : {}\n".format(str(sql_json["col_set"])))
            f.write("gold: {}\n".format(gold))
            f.write("pred: {}\n".format(pred))
            f.write("\n")
