import os
import json
import copy
import random
import _jsonnet
import argparse
import datetime

import torch
import torch.optim as optim
import numpy as np

from src import utils
from src.models.model import IRNet
from torch.utils.tensorboard import SummaryWriter


def logging_to_tensorboard(summary_writer, prefix, dic, epoch):
    for key in dic.keys():
        summary_writer.add_scalar(prefix + key, dic[key], epoch)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config", type=str, default="", help="Path for train config json file"
    )
    parser.add_argument(
        "--debugging", action="store_true", help="Run in debugging mode"
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sql_data, table_data, val_sql_data, val_table_data = utils.load_dataset(
        H_PARAMS["data_path"],
        use_small=H_PARAMS["toy"],
        is_simple_query=H_PARAMS["is_simple_query"],
        is_single_table=H_PARAMS["is_single_table"],
    )

    # Filter data for bert
    if H_PARAMS["bert"] != -1:
        sql_data = [
            data
            for data in sql_data
            if data["db_id"] != "baseball_1"
            and data["db_id"] != "cre_Drama_Workshop_Groups"
            and data["db_id"] != "sakila_1"
            and data["db_id"] != "formula_1"
            and data["db_id"] != "soccer_1"
        ]
        val_sql_data = [data for data in val_sql_data if data["db_id"] != "baseball_1"]

    print("train data length: {}".format(len(sql_data)))
    print("dev data length: {}".format(len(val_sql_data)))

    model = IRNet(H_PARAMS, is_qgm=H_PARAMS["is_qgm"], is_cuda=args.cuda != -1)

    if args.cuda != -1:
        torch.cuda.set_device(args.cuda)
        model.cuda()

    # now get the optimizer
    optimizer_cls = eval("torch.optim.%s" % H_PARAMS["optimizer"])
    optimizer = optimizer_cls(model.without_bert_params, lr=H_PARAMS["lr"])
    if H_PARAMS["bert"] != -1:
        bert_optimizer = optimizer_cls(
            model.transformer_encoder.parameters(), lr=H_PARAMS["bert_lr"]
        )
    else:
        bert_optimizer = None
    print("Enable Learning Rate Scheduler: ", H_PARAMS["lr_scheduler"])
    if H_PARAMS["lr_scheduler"]:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=H_PARAMS["milestones"],
            gamma=H_PARAMS["lr_scheduler_gamma"],
        )
        scheduler_bert = (
            optim.lr_scheduler.MultiStepLR(
                bert_optimizer,
                milestones=H_PARAMS["milestones"],
                gamma=H_PARAMS["lr_scheduler_gamma"],
            )
            if bert_optimizer
            else None
        )
    else:
        scheduler = None
        scheduler_bert = None

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
    log_path = os.path.join(
        H_PARAMS["log_path"], "debugging" if args.debugging else H_PARAMS["log_key"]
    )
    log_model_path = os.path.join(log_path, "model")

    # Tensorboard
    summary_writer = SummaryWriter(log_path)

    # Log hyper-parameters
    with open(os.path.join(log_path, "config.jsonnet"), "w") as f:
        f.write(str(H_PARAMS))

    # Create Save directory
    if not os.path.exists(log_model_path):
        os.mkdir(log_model_path)

    best_dev_total_acc = 0
    for epoch in range(1, H_PARAMS["max_epoch"]):
        print(
            "\nEpoch: {}  lr: {:.2e}  step: {}  Time: {}".format(
                epoch,
                scheduler.optimizer.param_groups[0]["lr"],
                scheduler.optimizer._step_count,
                str(datetime.datetime.now()).split(".")[0],
            )
        )

        # Evaluation
        train_loss = utils.epoch_train(
            model,
            optimizer,
            bert_optimizer,
            H_PARAMS["batch_size"],
            sql_data,
            table_data,
            H_PARAMS["clip_grad"],
            is_qgm=H_PARAMS["is_qgm"],
        )
        if not epoch % H_PARAMS["eval_freq"] or epoch == H_PARAMS["epoch"]:
            print("Evaluation:")
            train_total_acc = utils.epoch_acc(
                model,
                H_PARAMS["batch_size"],
                sql_data,
                table_data,
                is_qgm=H_PARAMS["is_qgm"],
            )
            dev_loss = utils.epoch_train(
                model,
                optimizer,
                bert_optimizer,
                H_PARAMS["batch_size"],
                val_sql_data,
                val_table_data,
                H_PARAMS["clip_grad"],
                is_qgm=H_PARAMS["is_qgm"],
                is_train=False,
            )
            dev_total_acc = utils.epoch_acc(
                model,
                H_PARAMS["batch_size"],
                val_sql_data,
                val_table_data,
                is_qgm=H_PARAMS["is_qgm"],
            )
            print("Train Acc: {}".format(train_total_acc["total"]))
            print("Dev Acc: {}".format(dev_total_acc["total"]))

            # Logging to tensorboard
            logging_to_tensorboard(summary_writer, "train_loss/", train_loss, epoch)
            logging_to_tensorboard(summary_writer, "train_acc/", train_total_acc, epoch)
            logging_to_tensorboard(summary_writer, "dev_loss/", dev_loss, epoch)
            logging_to_tensorboard(summary_writer, "dev_acc/", dev_total_acc, epoch)

            # Save if total_acc is higher
            if best_dev_total_acc < dev_total_acc["total"]:
                best_dev_total_acc = dev_total_acc["total"]
                print("Saving new best model with acc: {}".format(best_dev_total_acc))
                torch.save(
                    model.state_dict(),
                    os.path.join(log_model_path, "{}_best_model.pt".format(epoch)),
                )

        # Change learning rate
        scheduler.step()
        if scheduler_bert:
            scheduler_bert.step()
