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

    # Append anw
    # train_data = utils.append_ground_truth(SemQL, train_data)
    # val_data = utils.append_ground_truth(SemQL, val_data)
    #
    # # Load encoder
    # if cfg.decoder_name == "ensemble":
    #     # Load trained weight
    #     log.info("Load pretrained model from {}".format(cfg.model_path))
    #     model.load_model()
    # else:
    #     # Load trained weights
    #     log.info("Load pretrained model from {}".format(cfg.load_model))
    #     pretrained_model = torch.load(
    #         cfg.load_model, map_location=lambda storage, loc: storage
    #     )
    #     # Load
    #     model.load_state_dict(pretrained_model)

    # # Load word embedding
    # model.word_emb = None if cfg.is_bert else utils.load_word_emb(cfg.glove_embed_path)

    # Evaluation
    log.info("Evaluation:")

    # (
    #     train_total_acc,
    #     train_is_correct,
    #     train_pred,
    #     train_gold,
    #     train_list,
    # ) = utils.epoch_acc(
    #     model,
    #     cfg.batch_size,
    #     train_data,
    #     table_data,
    #     cfg.decoder_name,
    #     cfg.is_col_set,
    #     return_details=True,
    # )

    if True:
        total_acc_pred, total_acc_refined, total_acc_arbitrated, total_acc_init_pred = utils.epoch_acc(
            model,
            cfg.batch_size,
            val_data,
            cfg.decoder_name,
        )
        print("total acc pred: {}".format(total_acc_pred))
        print("total acc refined: {}".format(total_acc_refined))
        print("total acc arbitrated: {}".format(total_acc_arbitrated))
        print("total acc init pred: {}".format(total_acc_init_pred))
        return None
    else:
        dev_total_acc, dev_is_correct, dev_pred, dev_gold, dev_list = utils.epoch_acc(
            model,
            cfg.batch_size,
            val_data,
            cfg.decoder_name,
            return_details=True,
        )

    # print("Train Acc: {}".format(train_total_acc["total"]))
    print("Dev Acc: {}".format(dev_total_acc["total"]))

    # train_out_path = os.path.join(log_path, "train.result")
    dev_out_path = os.path.join(log_path, "dev.result")

    # # Save outputs from train
    # assert len(train_pred) == len(train_gold) and len(train_gold) == len(train_list)
    # if cfg.model_name == "transformer":
    #     # Format pred
    #     tmp = []
    #     for pred in train_pred:
    #         tmp += [" ".join(["{}({})".format(*item) for item in pred])]
    #     train_pred = tmp

    # utils.write_eval_result_as(
    #     train_out_path,
    #     train_list,
    #     train_is_correct,
    #     train_total_acc,
    #     train_pred,
    #     train_gold,
    #     use_col_set=cfg.is_col_set,
    # )

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
