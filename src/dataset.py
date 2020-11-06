# import copy
# import preprocess.rule.semQL as define_rule
import torch
import numpy as np


class Example:
    def __init__(
        self,
        src_sent,
        src_sent_type=None,
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
    ):
        self.src_sent = src_sent
        self.src_sent_type = src_sent_type
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
        self.col_hot_type = col_hot_type
        self.bert_input = None
        self.bert_input_indices = None
        # self.one_hot_type = one_hot_type
        # self.tab_hot_type = tab_hot_type
        # self.tokenized_src_sent = tokenized_src_sent
        # self.vis_seq = vis_seq
        # self.tab_iter = tab_iter
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
    def __init__(self, examples, grammar=None, is_cuda=False, use_bert_cache=False):
        self.examples = examples

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_type = [e.src_sent_type for e in self.examples]
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
        self.db = [e.db for e in examples]
        self.bert_input_indices = [e.bert_input_indices for e in examples]
        self.sql = [e.sql for e in examples]
        if use_bert_cache:
            self.bert_input = [e.bert_input for e in examples]
        else:
            self.bert_input = self._pad_bert_input(examples)
        # if examples[0].tgt_actions:
        #     self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
        #     self.max_sketch_num = max(len(e.sketch) for e in self.examples)
        # self.tokenized_src_sents = [e.tokenized_src_sent for e in self.examples]
        # self.tokenized_src_sents_len = [len(e.tokenized_src_sent) for e in examples]
        self.src_sents_word = [e.src_sent for e in self.examples]
        # self.table_sents_word = [
        #     [" ".join(x) for x in e.tab_cols] for e in self.examples
        # ]
        # self.schema_sents_word = [
        #     [" ".join(x) for x in e.table_names] for e in self.examples
        # ]
        # self.src_type = [e.one_hot_type for e in self.examples]
        self.col_hot_type = [e.col_hot_type for e in self.examples]
        # self.tab_hot_type = [e.tab_hot_type for e in self.examples]
        # self.table_names_iter = [e.tab_iter for e in self.examples]
        # self.tab_ids = [e.tab_ids for e in self.examples]
        # self.table_col_name = [e.table_col_name for e in examples]
        # self.table_col_len = [e.table_col_len for e in examples]
        # self.col_pred = [e.col_pred for e in examples]
        # self.qgm_action = [e.qgm_action for e in examples]

        self.grammar = grammar
        self.cuda = is_cuda

    def _pad_bert_input(self, examples):
        if not examples[0].bert_input:
            return None
        lens = [item.bert_input_indices[-1][-1][-1] for item in examples]
        max_len = max(lens)

        bert_inputs = []
        for idx, length in enumerate(lens):
            bert_inputs += [examples[idx].bert_input + (" [PAD]" * (max_len - length))]

        return bert_inputs

    def __len__(self):
        return len(self.examples)

    def table_dict_mask(self, table_dict):
        return table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    @cached_property
    def pred_col_mask(self):
        return pred_col_mask(self.col_pred, self.col_num)

    @cached_property
    def schema_token_mask(self):
        return length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return length_array_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_appear_mask(self):
        return appear_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_unk_mask(self):
        return length_array_to_mask_tensor(self.col_num, cuda=self.cuda, value=None)

    @cached_property
    def src_token_mask(self):
        return length_array_to_mask_tensor(self.src_sents_len, cuda=self.cuda)


def table_dict_to_mask_tensor(length_array, table_dict, cuda=False):
    max_len = max(length_array)
    batch_size = len(table_dict)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, ta_val in enumerate(table_dict):
        for tt in ta_val:
            mask[i][tt] = 0

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def pred_col_mask(value, max_len):
    max_len = max(max_len)
    batch_size = len(value)
    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for v_ind, v_val in enumerate(value):
        for v in v_val:
            mask[v_ind][v] = 0
    mask = torch.ByteTensor(mask)
    return mask.cuda()


def length_array_to_mask_tensor(length_array, cuda=False, value=None):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    if value != None:
        for b_id in range(len(value)):
            for c_id, c in enumerate(value[b_id]):
                if value[b_id][c_id] == [3]:
                    mask[b_id][c_id] = 1

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def appear_to_mask_tensor(length_array, cuda=False, value=None):
    max_len = max(length_array)
    batch_size = len(length_array)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)
    return mask
