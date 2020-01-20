# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""

import copy
import random

import src.rule.semQL as define_rule
from src.models import nn_utils
from src import utils

class Example:
    """

    """
    def __init__(self, src_sent, vis_seq=None, tab_cols=None, tab_iter=None, col_num=None, sql=None,
                 one_hot_type=None, col_hot_type=None, tab_hot_type=None, schema_len=None, tab_ids=None,
                 table_names=None, table_len=None, col_table_dict=None, cols=None,
                 table_col_name=None, table_col_len=None,
                  col_pred=None, tokenized_src_sent=None, qgm=None
        ):

        self.src_sent = src_sent
        self.tokenized_src_sent = tokenized_src_sent
        self.vis_seq = vis_seq
        self.tab_cols = tab_cols
        self.tab_iter = tab_iter
        self.col_num = col_num
        self.sql = sql
        self.one_hot_type=one_hot_type
        self.col_hot_type = col_hot_type
        self.tab_hot_type = tab_hot_type
        self.schema_len = schema_len
        self.tab_ids = tab_ids
        self.table_names = table_names
        self.table_len = table_len
        self.col_table_dict = col_table_dict
        self.cols = cols
        self.table_col_name = table_col_name
        self.table_col_len = table_col_len
        self.col_pred = col_pred
        self.qgm = qgm


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Batch(object):
    def __init__(self, examples, grammar=None, is_cuda=False):
        self.examples = examples

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]
        self.tokenized_src_sents = [e.tokenized_src_sent for e in self.examples]
        self.tokenized_src_sents_len = [len(e.tokenized_src_sent) for e in examples]
        self.src_sents_word = [e.src_sent for e in self.examples]
        self.table_sents_word = [[" ".join(x) for x in e.tab_cols] for e in self.examples]

        self.schema_sents_word = [[" ".join(x) for x in e.table_names] for e in self.examples]

        self.src_type = [e.one_hot_type for e in self.examples]
        self.col_hot_type = [e.col_hot_type for e in self.examples]
        self.tab_hot_type = [e.tab_hot_type for e in self.examples]
        self.table_sents = [e.tab_cols for e in self.examples]
        self.table_names_iter = [e.tab_iter for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]
        self.tab_ids = [e.tab_ids for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_table_dict = [e.col_table_dict for e in examples]
        self.table_col_name = [e.table_col_name for e in examples]
        self.table_col_len = [e.table_col_len for e in examples]
        self.col_pred = [e.col_pred for e in examples]
        self.qgm = [e.qgm for e in examples]

        self.grammar = grammar
        self.cuda = is_cuda

    def __len__(self):
        return len(self.examples)


    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    @cached_property
    def pred_col_mask(self):
        return nn_utils.pred_col_mask(self.col_pred, self.col_num)

    @cached_property
    def schema_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_unk_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda, value=None)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    cuda=self.cuda)
