# -*- coding: utf-8 -*-
import torch.nn as nn
from typing import List
from models.encoder.irnet.encoder import IRNetLSTMEncoder
from models.decoder.transformer_framework.decoder import TransformerDecoderFramework
from rule.semql.semql import SemQL


class LSTMEncoderQGMTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMEncoderQGMTransformerDecoder, self).__init__()
        self.encoder = IRNetLSTMEncoder(cfg)
        self.decoder = TransformerDecoderFramework(cfg)

    def forward(self, examples):

        results = self.encoder(examples)
        # TODO make this code cleaner

        encoded_src = [result["src_encoding"] for result in results]
        encoded_col = [result["col_encoding"] for result in results]
        encoded_tab = [result["tab_encoding"] for result in results]
        col_tab_dic = [example.col_table_dict for example in examples]
        tab_col_dic = []
        for b_idx in range(len(col_tab_dic)):
            b_tmp = []
            tab_len = len(col_tab_dic[b_idx][0])
            for t_idx in range(tab_len):
                tab_tmp = [
                    idx
                    for idx in range(len(col_tab_dic[b_idx]))
                    if t_idx in col_tab_dic[b_idx][idx]
                ]
                b_tmp += [tab_tmp]
            tab_col_dic += [b_tmp]

        if self.training:
            golds_str: List[str] = [
                SemQL.semql.create_data(example.qgm) for example in examples
            ]
            golds = [
                [
                    SemQL.semql.str_to_action(action_str)
                    for action_str in gold.split(" ")
                ]
                for gold in golds_str
            ]
        else:
            golds = None
        return self.decoder(
            encoded_src, encoded_col, encoded_tab, col_tab_dic, tab_col_dic, golds
        )
