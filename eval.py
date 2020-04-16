import os
import copy
import torch
import hydra
import logging

from src import utils
from rule.semql.semql import SemQL
from models.wrapper_model import EncoderDecoderModel
from models.ensemble import EnsembleWrapper

log = logging.getLogger(__name__)


@hydra.main(config_path="config/config.yaml")
def evaluate(cfg):
    # Load Training Info
    log_path = os.getcwd()
    os.chdir("../../../")

    # Set random seed
    utils.set_random_seed(cfg.seed)

    # Set model
    if cfg.cuda != -1:
        torch.cuda.set_device(cfg.cuda)
    model = EnsembleWrapper(cfg)
    model.load_models(cfg.load_model)
    if cfg.cuda != -1:
        model.cuda()

    # Load dataset
    train_data, val_data, table_data = utils.load_dataset(
        cfg.toy, cfg.is_bert, cfg.dataset.path, cfg.query_type, cfg.use_down_schema,
    )

    # Evaluation
    log.info("Evaluation:")

    if True:
        (
            total_acc_pred,
            total_acc_refined,
            total_acc_arbitrated,
            total_acc_init_pred,
        ) = utils.epoch_acc(model, cfg.batch_size, val_data, cfg.decoder_name,)
        print("total acc pred: {}".format(total_acc_pred))
        print("total acc refined: {}".format(total_acc_refined))
        print("total acc arbitrated: {}".format(total_acc_arbitrated))
        print("total acc init pred: {}".format(total_acc_init_pred))
        return None
    else:
        dev_total_acc, dev_is_correct, dev_pred, dev_gold, dev_list = utils.epoch_acc(
            model, cfg.batch_size, val_data, cfg.decoder_name, return_details=True,
        )

    # print("Train Acc: {}".format(train_total_acc["total"]))
    print("Dev Acc: {}".format(dev_total_acc["total"]))

    # train_out_path = os.path.join(log_path, "train.result")
    dev_out_path = os.path.join(log_path, "dev.result")

    utils.write_eval_result_as(
        dev_out_path,
        dev_list,
        dev_is_correct,
        dev_total_acc,
        dev_pred,
        dev_gold,
        use_col_set=cfg.is_col_set,
    )

    # dev_is_correct
    indices = [idx for idx, item in enumerate(dev_is_correct) if item == True]
    print(indices)

    utils.analyze_regarding_schema_size(
        dev_list, dev_is_correct, dev_pred, dev_gold, table_data
    )

if __name__ == "__main__":
    evaluate()
