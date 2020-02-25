# -*- coding: utf-8 -*-
import os
import copy
import torch
import hydra
import logging

from src import utils
from src.models.model import IRNet

log = logging.getLogger(__name__)


@hydra.main(config_path="config/config.yaml")
def evaluate(cfg):
    # Load Training Info
    log_path = os.getcwd()
    os.chdir("../../../")

    # Set random seed
    utils.set_randome_seed(cfg.seed)

    # Load dataset
    train_data, val_data, table_data = utils.load_dataset(
        cfg.toy, cfg.is_bert, cfg.dataset.path, cfg.query_type
    )
    # Set model
    if cfg.cuda != -1:
        torch.cuda.set_device(cfg.cuda)
    model = IRNet(cfg)
    if cfg.cuda != -1:
        model.cuda()

    # Load trained weights
    assert cfg.load_model
    log.info("Load pretrained model from {}".format(cfg.load_model))
    pretrained_model = torch.load(
        cfg.load_model, map_location=lambda storage, loc: storage
    )
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    # Load word embedding
    model.word_emb = None if cfg.is_bert else utils.load_word_emb(cfg.glove_embed_path)

    # Evaluation
    log.info("Evaluation:")

    (
        train_total_acc,
        train_is_correct,
        train_pred,
        train_gold,
        train_list,
    ) = utils.epoch_acc(
        model,
        cfg.batch_size,
        train_data,
        table_data,
        cfg.model_name,
        return_details=True,
    )

    dev_total_acc, dev_is_correct, dev_pred, dev_gold, dev_list = utils.epoch_acc(
        model,
        cfg.batch_size,
        val_data,
        table_data,
        cfg.model_name,
        return_details=True,
    )

    print("Train Acc: {}".format(train_total_acc["total"]))
    print("Dev Acc: {}".format(dev_total_acc["total"]))

    train_out_path = os.path.join(log_path, "train.result")
    dev_out_path = os.path.join(log_path, "dev.result")

    use_col_set = "preprocess" in cfg.model_name

    # Save outputs from train
    assert len(train_pred) == len(train_gold) and len(train_gold) == len(train_list)
    if cfg.model_name == "transformer":
        # Format pred
        tmp = []
        for pred in train_pred:
            tmp += [" ".join(["{}({})".format(*item) for item in pred])]
        train_pred = tmp

    utils.write_eval_result_as(
        train_out_path,
        train_list,
        train_is_correct,
        train_total_acc,
        train_pred,
        train_gold,
        use_col_set=use_col_set,
    )

    # Save outputs from dev
    assert len(dev_pred) == len(dev_gold) and len(dev_gold) == len(dev_list)
    if cfg.model_name == "transformer":
        # Format pred
        tmp = []
        for pred in dev_pred:
            tmp += [" ".join(["{}({})".format(*item) for item in pred])]
        dev_pred = tmp
    utils.write_eval_result_as(
        dev_out_path,
        dev_list,
        dev_is_correct,
        dev_total_acc,
        dev_pred,
        dev_gold,
        use_col_set=use_col_set,
    )


if __name__ == "__main__":
    evaluate()
