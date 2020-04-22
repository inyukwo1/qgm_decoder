# -*- coding: utf-8 -*-

# import copy
# import preprocess.rule.semQL as define_rule
from src.models import nn_utils


class Example:
    def __init__(
        self,
        src_sent,
        tgt_actions=None,
        tab_cols=None,
        col_num=None,
        sql=None,
        table_names=None,
        table_len=None,
        col_tab_dic=None,
        tab_col_dic=None,
        cols=None,
        qgm=None,
        relation=None,
        gt=None,
        db_id=None,
        db=None,
        data=None,
        vis_seq=None,
        tab_iter=None,
        one_hot_type=None,
        col_hot_type=None,
        tab_hot_type=None,
        schema_len=None,
        tab_ids=None,
        qgm_action=None,
        table_col_name=None,
        table_col_len=None,
        col_pred=None,
        tokenized_src_sent=None,
        used_table_set=None,
    ):
        self.src_sent = src_sent
        self.tab_cols = tab_cols
        self.col_num = col_num
        self.sql = sql
        self.table_names = table_names
        self.table_len = table_len
        self.col_tab_dic = col_tab_dic
        self.tab_col_dic = tab_col_dic
        self.cols = cols
        self.tgt_actions = tgt_actions
        self.qgm = qgm
        self.relation = relation
        self.gt = gt
        self.db_id = db_id
        self.db = db
        self.data = data
        # self.tokenized_src_sent = tokenized_src_sent
        # self.vis_seq = vis_seq
        # self.tab_iter = tab_iter
        self.one_hot_type = one_hot_type
        self.col_hot_type = col_hot_type
        self.tab_hot_type = tab_hot_type
        self.used_table_set = list(used_table_set)
        self.used_col_set = list(set([action[1] for action in gt if action[0] == "C"]))
        # self.schema_len = schema_len
        # self.tab_ids = tab_ids
        # self.table_col_name = table_col_name
        # self.table_col_len = table_col_len
        # self.col_pred = col_pred
        # self.truth_actions = (
        #     copy.deepcopy(tgt_actions)
        #     if tgt_actions
        #     else copy.deepcopy(qgm_action).split(" ")
        # )
        # self.qgm_action = qgm_action
        # self.sketch = list()
        # if self.truth_actions:
        #     for ta in self.truth_actions:
        #         if (
        #             isinstance(ta, define_rule.C)
        #             or isinstance(ta, define_rule.T)
        #             or isinstance(ta, define_rule.A)
        #         ):
        #             continue
        #         self.sketch.append(ta)


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
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
        self.table_sents = [e.tab_cols for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_tab_dic = [e.col_tab_dic for e in examples]
        self.tab_col_dic = [e.tab_col_dic for e in examples]
        self.qgm = [e.qgm for e in examples]
        self.relation = [e.relation for e in examples]
        self.gt = [e.gt for e in examples]
        # if examples[0].tgt_actions:
        #     self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
        #     self.max_sketch_num = max(len(e.sketch) for e in self.examples)
        # self.tokenized_src_sents = [e.tokenized_src_sent for e in self.examples]
        # self.tokenized_src_sents_len = [len(e.tokenized_src_sent) for e in examples]
        # self.src_sents_word = [e.src_sent for e in self.examples]
        # self.table_sents_word = [
        #     [" ".join(x) for x in e.tab_cols] for e in self.examples
        # ]
        # self.schema_sents_word = [
        #     [" ".join(x) for x in e.table_names] for e in self.examples
        # ]
        # self.src_type = [e.one_hot_type for e in self.examples]
        self.col_hot_type = [e.col_hot_type for e in self.examples]
        self.tab_hot_type = [e.tab_hot_type for e in self.examples]
        # self.table_names_iter = [e.tab_iter for e in self.examples]
        # self.tab_ids = [e.tab_ids for e in self.examples]
        # self.table_col_name = [e.table_col_name for e in examples]
        # self.table_col_len = [e.table_col_len for e in examples]
        # self.col_pred = [e.col_pred for e in examples]
        # self.qgm_action = [e.qgm_action for e in examples]
        self.used_table_set = [e.used_table_set for e in self.examples]
        self.used_table_gt = [
            (
                [("Z", len(e.used_table_set) - 1)]
                + [("T", table) for table in e.used_table_set]
                + [("Y", len(e.used_col_set) - 1)]
                + [("C", col) for col in e.used_col_set]
            )
            for e in self.examples
        ]

        self.grammar = grammar
        self.cuda = is_cuda

    def __len__(self):
        return len(self.examples)

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(
            self.table_len, table_dict, cuda=self.cuda
        )

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
        return nn_utils.length_array_to_mask_tensor(
            self.col_num, cuda=self.cuda, value=None
        )

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len, cuda=self.cuda)
