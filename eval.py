import os
import copy
import torch
import hydra
import logging

from src import utils
from models.wrapper_model import EncoderDecoderModel

log = logging.getLogger(__name__)


@hydra.main(config_path="config/config.yaml")
def evaluate(cfg):
    # Load Training Info
    assert cfg.load_model, "Model path not defined!"
    log_path = cfg.load_model.split("/model/")[0]
    os.chdir("../../../")

    # Set random seed
    utils.set_random_seed(cfg.seed)

    # Set model
    if cfg.cuda != -1:
        torch.cuda.set_device(cfg.cuda)
    model = EncoderDecoderModel(cfg)
    model.load_model(cfg.load_model)
    if cfg.cuda != -1:
        model.cuda()

    # load word embedding
    model.word_emb = None if cfg.is_bert else utils.load_word_emb(cfg.glove_embed_path)

    # Load dataset
    train_data, val_data, table_data = utils.load_dataset(
        cfg.toy, cfg.is_bert, cfg.dataset.path, cfg.query_type, cfg.use_down_schema,
    )

    # Evaluation
    log.info("Evaluation:")

    (
        dev_total_acc,
        dev_is_correct,
        dev_pred,
        dev_gold,
        dev_list,
        details_lists,
    ) = utils.epoch_acc(model, cfg.batch_size, val_data, return_details=True)

    # print("Train Acc: {}".format(train_total_acc["total"]))
    print("Dev Acc: {}".format(dev_total_acc["total"]))

    # train_out_path = os.path.join(log_path, "train.result")
    out_file_tag = "_down_schema" if cfg.use_down_schema else ""
    dev_out_path = os.path.join(log_path, "dev{}.result".format(out_file_tag))

    utils.write_eval_result_as(
        dev_out_path, dev_list, dev_is_correct, dev_total_acc, dev_pred, dev_gold,
    )

    utils.analyze_regarding_schema_size(
        dev_list, dev_is_correct, dev_pred, dev_gold, table_data
    )


if __name__ == "__main__":
    evaluate()
