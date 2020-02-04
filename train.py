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
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config", type=str, default="", help="Path for train config json file"
    )
    parser.add_argument(
        "--debugging", action="store_true", help="Run in debugging mode"
    )
    parser.add_argument("--toy", action='store_true', help='If set, use small data; used for fast debugging.')
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
    train_datas, val_datas, table_data = utils.load_dataset(H_PARAMS, use_small=args.toy)

    # Set model
    model = IRNet(H_PARAMS, is_qgm=H_PARAMS["is_qgm"], is_cuda=args.cuda != -1)
    if args.cuda != -1:
        torch.cuda.set_device(args.cuda)
        model.cuda()

    # Set optimizer
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

    best_val_acc = 0
    for epoch in range(1, H_PARAMS["max_epoch"]):
        print(
            "\nEpoch: {}  lr: {:.2e}  step: {}  Time: {}".format(
                epoch,
                scheduler.optimizer.param_groups[0]["lr"],
                scheduler.optimizer._step_count,
                str(datetime.datetime.now()).split(".")[0],
            )
        )

        # Training
        train_loss = utils.epoch_train(
            model,
            optimizer,
            bert_optimizer,
            H_PARAMS["batch_size"],
            list(itertools.chain.from_iterable(train_datas)),
            table_data,
            H_PARAMS["clip_grad"],
            is_qgm=H_PARAMS["is_qgm"],
        )

        utils.logging_to_tensorboard(summary_writer, "Total_train_loss/", train_loss, epoch)

        # Evaluation
        if not epoch % H_PARAMS["eval_freq"] or epoch == H_PARAMS["epoch"]:
            print("Evaluation:")
            dataset_names = H_PARAMS['data_names']

            val_losses = []
            val_total_accs = []
            train_total_accs = []
            for idx, dataset_name in enumerate(dataset_names):
                train_data = train_datas[idx]
                val_data = val_datas[idx]

                train_total_accs += [utils.epoch_acc(
                    model,
                    H_PARAMS["batch_size"],
                    train_data,
                    table_data,
                    is_qgm=H_PARAMS["is_qgm"],
                )]
                val_losses += [utils.epoch_train(
                    model,
                    optimizer,
                    bert_optimizer,
                    H_PARAMS["batch_size"],
                    val_data,
                    table_data,
                    H_PARAMS["clip_grad"],
                    is_qgm=H_PARAMS["is_qgm"],
                    is_train=False,
                )]
                val_total_accs += [utils.epoch_acc(
                    model,
                    H_PARAMS["batch_size"],
                    val_data,
                    table_data,
                    is_qgm=H_PARAMS["is_qgm"],
                )]

                # Logging to tensorboard
                utils.logging_to_tensorboard(summary_writer, "{}_train_acc/".format(dataset_name), train_total_accs[idx], epoch)
                utils.logging_to_tensorboard(summary_writer, "{}_val_loss/".format(dataset_name), val_losses[idx], epoch)
                utils.logging_to_tensorboard(summary_writer, "{}_val_acc/".format(dataset_name), val_total_accs[idx], epoch)

            # Calculate Total Acc
            train_acc = utils.calculate_total_acc(train_total_accs, [len(datas) for datas in train_datas])
            val_acc = utils.calculate_total_acc(val_total_accs, [len(datas) for datas in val_datas])

            # Logging to tensorboard
            utils.logging_to_tensorboard(summary_writer, "Total_train_acc/", train_acc, epoch)
            utils.logging_to_tensorboard(summary_writer, "Total_val_acc/", val_acc, epoch)

            # Save if total_acc is higher
            if best_val_acc < val_acc['total']:
                best_val_acc = val_acc['total']
                print("Saving new best model with acc: {}".format(best_val_acc))
                torch.save(
                    model.state_dict(),
                    os.path.join(log_model_path, "best_model.pt"),
                )
                with open('best_model.log', 'a') as f:
                    f.write('Epoch: {} Train Acc: Val Acc:{}'.format(epoch, train_acc, best_val_acc))

            # Print Accuracy
            print("Total Train Acc: {}".format(train_acc['total']))
            for idx in range(len(dataset_names)):
                print("{}: {}".format(dataset_names[idx], train_total_accs[idx]["total"]))
            print("\nTotal Val Acc: {}".format(val_acc['total']))
            for idx in range(len(dataset_names)):
                print("{}: {}".format(dataset_names[idx], val_total_accs[idx]["total"]))
            print('\n')

        # Change learning rate
        scheduler.step()
        if scheduler_bert:
            scheduler_bert.step()
