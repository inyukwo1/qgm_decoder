# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : train.py
# @Software: PyCharm
"""

import time
import traceback

import os
import torch
import torch.optim as optim
import tqdm
import copy

from src import args as arg
from src import utils
from src.models.model import IRNet
from src.rule import semQL


def train(args):
    """
    :param args:
    :return:
    """

    grammar = semQL.Grammar()
    sql_data, table_data, val_sql_data,\
    val_table_data= utils.load_dataset(args.dataset, use_small=args.toy)

    model = IRNet(args, grammar)

    if args.cuda: model.cuda()

    # now get the optimizer
    optimizer_cls = eval('torch.optim.%s' % args.optimizer)
    optimizer = optimizer_cls(model.without_bert_params, lr=args.lr)
    if args.bert != -1:
        bert_optimizer = optimizer_cls(model.transformer_encoder.parameters(), lr=args.bert_lr)
    else:
        bert_optimizer = None
    print('Enable Learning Rate Scheduler: ', args.lr_scheduler)
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_scheduler_gammar)
        scheduler_bert = optim.lr_scheduler.MultiStepLR(bert_optimizer,
                            milestones=args.milestones, gamma=args.lr_scheduler_gammar) if bert_optimizer else None
    else:
        scheduler = None
        scheduler_bert = None

    print('Loss epoch threshold: %d' % args.loss_epoch_threshold)
    print('Sketch loss coefficient: %f' % args.sketch_loss_coefficient)

    if args.load_model:
        print('load pretrained model from %s'% (args.load_model))
        pretrained_model = torch.load(args.load_model,
                                         map_location=lambda storage, loc: storage)
        pretrained_modeled = copy.deepcopy(pretrained_model)
        for k in pretrained_model.keys():
            if k not in model.state_dict().keys():
                del pretrained_modeled[k]

        model.load_state_dict(pretrained_modeled)

    model.word_emb = utils.load_word_emb(args.glove_embed_path)
    #model.word_emb = None
    # begin train

    model_save_path = utils.init_log_checkpoint_path(args)
    utils.save_args(args, os.path.join(model_save_path, 'config.json'))
    best_dev_acc = .0
    #
    # try:
    with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:
        for epoch in tqdm.tqdm(range(args.epoch)):
            if args.lr_scheduler:
                scheduler.step()
            if scheduler_bert:
                scheduler_bert.step()
            epoch_begin = time.time()
            loss = utils.epoch_train(model, optimizer, bert_optimizer, args.batch_size, sql_data, table_data, args, epoch,
                               loss_epoch_threshold=args.loss_epoch_threshold,
                               sketch_loss_coefficient=args.sketch_loss_coefficient)
            epoch_end = time.time()
            if epoch % 3 == 0:
                json_datas = utils.epoch_acc(model, args.batch_size, val_sql_data, val_table_data,
                                             beam_size=args.beam_size)
                acc, sketch_acc = utils.eval_acc(json_datas, val_sql_data)

                if acc > best_dev_acc:
                    utils.save_checkpoint(model, os.path.join(model_save_path, 'best_model.model'))
                    best_dev_acc = acc
                utils.save_checkpoint(model, os.path.join(model_save_path, '{%s}_{%s}.model') % (epoch, acc))

                log_str = 'Epoch: %d, Loss: %f, Sketch Acc: %f, Acc: %f, time: %f\n' % (
                    epoch + 1, loss, sketch_acc, acc, epoch_end - epoch_begin)
                tqdm.tqdm.write(log_str)
                epoch_fd.write(log_str)
                epoch_fd.flush()
    # except Exception as e:
    #     # Save model
    #     utils.save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))
    #     print(e)
    #     tb = traceback.format_exc()
    #     print(tb)
    # else:
    #     utils.save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))
    #     json_datas = utils.epoch_acc(model, args.batch_size, val_sql_data, val_table_data,
    #                                  beam_size=args.beam_size)
    #     acc = utils.eval_acc(json_datas, val_sql_data)
    #
    #     print("Sketch Acc: %f, Acc: %f, Beam Acc: %f" % (acc, acc, acc,))


if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    print(args)
    train(args)