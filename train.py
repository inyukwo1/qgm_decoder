import os
import copy
import hydra
import logging
import datetime
import random
import math

import torch
import torch.optim as optim
from radam import RAdam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src import utils
from rule.semql.semql import SemQL
from models.wrapper_model import EncoderDecoderModel
# from models.LSTMEncoderQGMTransformerDecoder import LSTMEncoderQGMTransformerDecoder
from optimizer import *


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
    train_data, val_data, table_data = utils.load_dataset(model,
        cfg.toy, cfg.is_bert, cfg.dataset.path, cfg.query_type, cfg.use_down_schema, cfg.remove_punctuation_marks, cfg
    )

    # Set optimizer
    optimizer_cls = (
        RAdam if cfg.optimizer == "radam" else eval("torch.optim.%s" % cfg.optimizer)
    )
    optimizer = optimizer_cls(model.without_bert_params, lr=cfg.lr)
    if cfg.encoder_name == "bert" and cfg.train_bert:
        bert_optimizer = optimizer_cls(
            model.encoder.parameters(), lr=cfg.bert_lr
        )
    else:
        bert_optimizer = None
    log.info("Enable Learning Rate Scheduler: {}".format(cfg.lr_scheduler))
    if cfg.lr_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.milestones, gamma=cfg.lr_scheduler_gamma,
        )
        # scheduler = WarmUpMultiStepLR(optimizer,
        #                               warmup_steps=cfg.max_step_cnt/20,
        #                               milestones=cfg.milestones,
        #                               gamma=cfg.lr_scheduler_gamma,
        #                               )
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

    model.word_emb = None if cfg.is_bert else utils.load_word_emb(cfg.glove_embed_path)

    if cfg.train_glove:
        # Filter
        tmp = {}
        for datas in [train_data, val_data]:
            for data in datas:
                # src_sents, tab_cols, table_names
                for words_list in [data.src_sent, data.tab_cols, data.table_names]:
                    for words in words_list:
                        for word in words:
                            if word in model.word_emb:
                                tmp[word] = model.word_emb[word]
        tmp["unk"] = model.word_emb["unk"]
        model.mapping = {}
        model.word_emb = torch.nn.ParameterList([])
        for idx, (key, value) in enumerate(tmp.items()):
            model.mapping[key] = idx
            model.word_emb.append(torch.nn.Parameter(torch.tensor(value)))

        model.cuda()

    if cfg.encoder_name == "lstm":
        model.encoder.word_emb = model.word_emb

    # Log path
    log_model_path = os.path.join(log_path, "model")

    # Tensorboard
    summary_writer = SummaryWriter(log_path)

    # Create Save directory
    if not os.path.exists(log_model_path):
        os.mkdir(log_model_path)

    # Sort
    train_data.sort(key=lambda item: len(item.src_sent) + item.table_len + item.col_num)

    best_val_acc = 0
    cnts_per_epoch = math.ceil(len(train_data) / cfg.batch_size)
    for cnt in tqdm(range(0, cfg.max_step_cnt * cnts_per_epoch)):
        epoch = int(cnt / cnts_per_epoch) + 1
        step_cnt = scheduler.optimizer._step_count
        is_to_step = (cnt % cfg.optimize_freq) == 0 and cnt

        log.info(
            "\nEpoch: {}  lr: {:.3e} cnt:{} step: {}  Time: {}".format(
                epoch,
                scheduler.optimizer.param_groups[0]["lr"],
                cnt,
                step_cnt,
                str(datetime.datetime.now()).split(".")[0],
                )
        )

        # Shuffle
        if cnt % cnts_per_epoch == 0:
            # shuffle
            def chunks(lst, n):
                for i in range(0, len(lst), n):
                    yield lst[i: i + n]
            new_train_data = []
            for train_data_chunk in chunks(train_data, cfg.batch_size * 3):
                random.shuffle(train_data_chunk)
                new_train_data += [train_data_chunk]
            random.shuffle(new_train_data)
            shuffled_train_data = []
            for train_data_chunk in new_train_data:
                shuffled_train_data += train_data_chunk

        # Training
        # create sub train data
        batch_front = cnt % cnts_per_epoch
        if (cnt+1) % cnts_per_epoch or not len(train_data) % cfg.batch_size:
            batch_rear = batch_front + cfg.batch_size
        else:
            batch_rear = batch_front + (len(train_data) % cfg.batch_size)
        print("front: {} rear:{}".format(batch_front, batch_rear))
        mini_batch = shuffled_train_data[batch_front:batch_rear]

        train_loss, train_loss_dic = utils.train(
            model,
            mini_batch,
            cfg.decoder_name,
        )

        train_loss.backward()

        if is_to_step:
            if cfg.clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if bert_optimizer:
                bert_optimizer.step()
                bert_optimizer.zero_grad()
                scheduler_bert.step()

        dataset_name = cfg.dataset.name
        utils.logging_to_tensorboard(
            summary_writer, "{}_train_loss/".format(dataset_name), train_loss_dic, step_cnt
        )
        # Evaluation
        if (cnt % (cnts_per_epoch * cfg.eval_freq)) == 0 and cnt or step_cnt == cfg.max_step_cnt:
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
            train_acc = utils.epoch_acc(
                model, cfg.batch_size, train_data
            )
            val_acc = utils.epoch_acc(
                model, cfg.batch_size, val_data
            )
            utils.logging_to_tensorboard(
                summary_writer,
                "lr",
                scheduler.optimizer.param_groups[0]["lr"],
                step_cnt,
            )
            # Logging to tensorboard
            utils.logging_to_tensorboard(
                summary_writer,
                "{}_train_acc/".format(dataset_name),
                train_acc,
                step_cnt,
            )
            utils.logging_to_tensorboard(
                summary_writer,
                "{}_val_loss/".format(dataset_name),
                val_loss,
                step_cnt,
            )
            utils.logging_to_tensorboard(
                summary_writer, "{}_val_acc/".format(dataset_name), val_acc, step_cnt,
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
                        "Step: {} Train Acc: {} Val Acc:{}".format(
                            step_cnt, train_acc, best_val_acc
                        )
                    )

if __name__ == "__main__":
    train()
