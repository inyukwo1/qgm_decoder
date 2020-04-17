import os
import copy
import hydra
import logging
import datetime

import torch
import torch.optim as optim
from radam import RAdam
from torch.utils.tensorboard import SummaryWriter

from src import utils
from rule.semql.semql import SemQL
from models.wrapper_model import EncoderDecoderModel

# from models.LSTMEncoderQGMTransformerDecoder import LSTMEncoderQGMTransformerDecoder

log = logging.getLogger(__name__)


@hydra.main(config_path="config/config.yaml")
def train(cfg):
    # Logging path
    log_path = os.getcwd()
    os.chdir("../../../")

    # Set random seed
    utils.set_random_seed(cfg.seed)

    # Set model
    if cfg.cuda != -1:
        torch.cuda.set_device(cfg.cuda)
    model = EncoderDecoderModel(cfg)
    if cfg.cuda != -1:
        model.cuda()

    # Load dataset
    train_data, val_data, table_data = utils.load_dataset(
        cfg.toy, cfg.is_bert, cfg.dataset.path, cfg.query_type, cfg.use_down_schema
    )

    # Set optimizer
    optimizer_cls = (
        RAdam if cfg.optimizer == "radam" else eval("torch.optim.%s" % cfg.optimizer)
    )
    optimizer = optimizer_cls(model.without_bert_params, lr=cfg.lr)
    if cfg.is_bert:
        bert_optimizer = optimizer_cls(
            model.transformer_encoder.parameters(), lr=cfg.bert_lr
        )
    else:
        bert_optimizer = None
    log.info("Enable Learning Rate Scheduler: {}".format(cfg.lr_scheduler))
    if cfg.lr_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.milestones, gamma=cfg.lr_scheduler_gamma,
        )
        scheduler_bert = (
            optim.lr_scheduler.MultiStepLR(
                bert_optimizer, milestones=cfg.milestones, gamma=cfg.lr_scheduler_gamma,
            )
            if bert_optimizer
            else None
        )
    else:
        scheduler = None
        scheduler_bert = None

    if cfg.load_model:
        log.info("Load pretrained model from {}".format(cfg.load_model))
        pretrained_model = torch.load(
            cfg.load_model, map_location=lambda storage, loc: storage
        )
        pretrained_modeled = copy.deepcopy(pretrained_model)
        for k in pretrained_model.keys():
            if k not in model.state_dict().keys():
                del pretrained_modeled[k]
        model.load_state_dict(pretrained_modeled)

    # model.load_word_emb()
    model.word_emb = None if cfg.is_bert else utils.load_word_emb(cfg.glove_embed_path)

    # Log path
    log_model_path = os.path.join(log_path, "model")

    # Tensorboard
    summary_writer = SummaryWriter(log_path)

    # Create Save directory
    if not os.path.exists(log_model_path):
        os.mkdir(log_model_path)

    best_val_acc = 0
    for epoch in range(1, cfg.max_epoch):
        log.info(
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
            cfg.batch_size,
            train_data,
            cfg.clip_grad,
            cfg.decoder_name,
            optimize_freq=cfg.optimize_freq,
        )

        dataset_name = cfg.dataset.name

        utils.logging_to_tensorboard(
            summary_writer, "{}_train_loss/".format(dataset_name), train_loss, epoch
        )

        # Evaluation
        if not epoch % cfg.eval_freq or epoch == cfg.max_epoch:
            log.info("Evaluation:")
            val_loss = utils.epoch_train(
                model,
                optimizer,
                bert_optimizer,
                cfg.batch_size,
                val_data,
                cfg.clip_grad,
                cfg.decoder_name,
                is_train=False,
                optimize_freq=cfg.optimize_freq,
            )

            if cfg.decoder_name == "transformer":
                train_acc_pred = utils.epoch_acc(
                    model, cfg.batch_size, train_data,
                )
                val_acc_pred = utils.epoch_acc(
                    model, cfg.batch_size, val_data,
                )
                # Logging to tensorboard
                utils.logging_to_tensorboard(
                    summary_writer,
                    "{}_train_pred_acc/".format(dataset_name),
                    train_acc_pred,
                    epoch,
                )
                utils.logging_to_tensorboard(
                    summary_writer,
                    "{}_val_loss/".format(dataset_name),
                    val_loss,
                    epoch,
                )
                utils.logging_to_tensorboard(
                    summary_writer,
                    "{}_val_pred_acc/".format(dataset_name),
                    val_acc_pred,
                    epoch,
                )
                # Print Accuracy
                log.info("Total Train Acc: {}\n".format(train_acc_pred["total"],))
                log.info("Total Val Acc: {}\n".format(val_acc_pred["total"],))

                # Save if total_acc is higher
                if best_val_acc <= val_acc_pred["total"]:
                    best_val_acc = val_acc_pred["total"]
                    log.info("Saving new best model with acc: {}".format(best_val_acc))
                    torch.save(
                        model.state_dict(),
                        os.path.join(log_model_path, "best_model.pt"),
                    )
                    with open(os.path.join(log_path, "best_model.log"), "a") as f:
                        f.write(
                            "Epoch: {} Train Acc: {} Val Acc:{}".format(
                                epoch, train_acc_pred, best_val_acc
                            )
                        )
            else:
                train_acc = utils.epoch_acc(
                    model, cfg.batch_size, train_data,
                )
                val_acc = utils.epoch_acc(
                    model, cfg.batch_size, val_data,
                )

                # Logging to tensorboard
                utils.logging_to_tensorboard(
                    summary_writer,
                    "{}_train_acc/".format(dataset_name),
                    train_acc,
                    epoch,
                )
                utils.logging_to_tensorboard(
                    summary_writer,
                    "{}_val_loss/".format(dataset_name),
                    val_loss,
                    epoch,
                )
                utils.logging_to_tensorboard(
                    summary_writer, "{}_val_acc/".format(dataset_name), val_acc, epoch,
                )
                # Print Accuracy
                log.info("Total Train Acc: {}".format(train_acc["total"]))
                log.info("Total Val Acc: {}\n".format(val_acc["total"]))
                # Save if total_acc is higher
                if best_val_acc <= val_acc["total"]:
                    best_val_acc = val_acc["total"]
                    log.info("Saving new best model with acc: {}".format(best_val_acc))
                    torch.save(
                        model.state_dict(),
                        os.path.join(log_model_path, "best_model.pt"),
                    )
                    with open(os.path.join(log_path, "best_model.log"), "a") as f:
                        f.write(
                            "Epoch: {} Train Acc: {} Val Acc:{}".format(
                                epoch, train_acc, best_val_acc
                            )
                        )

        # Change learning rate
        scheduler.step()
        if scheduler_bert:
            scheduler_bert.step()


if __name__ == "__main__":
    train()
