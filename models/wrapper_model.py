import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils

from src.dataset import Batch
from decoder.qgm.qgm_decoder import QGM_Decoder

from decoder.lstm.decoder import LSTM_Decoder
from decoder.transformer_framework.decoder import TransformerDecoderFramework
from decoder.semql.semql_decoder import SemQL_Decoder
from decoder.semql_framework.decoder import SemQLDecoderFramework
from decoder.ra_transformer_framework.decoder import RATransformerDecoder
from decoder.ensemble.decoder import EnsembleDecoder

from encoder.ra_transformer.encoder import RA_Transformer_Encoder
from encoder.transformer.encoder import Transformer_Encoder
from encoder.lstm.encoder import LSTMEncoder
from encoder.bert.encoder import BERT
from encoder.irnet.encoder import IRNetLSTMEncoder
import src.relation as relation


class EncoderDecoderModel(nn.Module):
    def __init__(self, cfg):
        super(EncoderDecoderModel, self).__init__()
        self.cfg = cfg
        self.is_bert = cfg.is_bert
        self.is_cuda = cfg.cuda != -1
        self.use_col_set = cfg.is_col_set
        self.encoder_name = cfg.encoder_name
        self.decoder_name = cfg.decoder_name
        self.embed_size = 1024 if self.is_bert else 300

        # Decoder
        if self.decoder_name == "transformer":
            self.decoder = TransformerDecoderFramework(cfg)
        elif self.decoder_name == "lstm":
            self.decoder = LSTM_Decoder(cfg)
        elif self.decoder_name == "qgm":
            self.decoder = QGM_Decoder(cfg)
        elif self.decoder_name == "semql":
            self.decoder = SemQLDecoderFramework(cfg)
        elif self.decoder_name == "ra_transformer":
            self.decoder = RATransformerDecoder(cfg)
        elif self.decoder_name == "ensemble":
            self.decoder = EnsembleDecoder(cfg)
        else:
            raise RuntimeError("Unsupported decoder name: {}".format(self.decoder_name))

        self.without_bert_params = list(self.parameters(recurse=True))

        # Encoder
        if self.encoder_name == "bert":
            self.encoder = BERT(cfg)
        elif self.encoder_name == "lstm":
            self.encoder = LSTMEncoder(cfg)
            #self.encoder = IRNetLSTMEncoder(cfg)
        elif self.encoder_name == "transformer":
            self.encoder = Transformer_Encoder(cfg)
        elif self.encoder_name == "ra_transformer":
            self.encoder = RA_Transformer_Encoder(cfg)
        else:
            raise RuntimeError("Unsupported encoder name")

        # Key embeddings
        self.key_emb = nn.Embedding(4, 300).double()


        if self.encoder_name != "bert":
            self.without_bert_params = list(self.parameters(recurse=True))

    def gen_x_batch(self, q):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        is_list = False
        if type(q[0][0]) == list:
            is_list = True
        for i, one_q in enumerate(q):
            if not is_list:
                q_val = []
                for w in one_q:
                    if w == "[db]":
                        q_val.append(self.key_emb[torch.tensor(0)])
                    elif w == "[table]":
                        q_val.append(self.key_emb[torch.tensor(1)])
                    elif w == "[column]":
                        q_val.append(self.key_emb[torch.tensor(3)])
                    elif w == "[value]":
                        q_val.append(self.key_emb[torch.tensor(4)])
                    else:
                        q_val.append(self.word_emb.get(w, self.word_emb["unk"]))

                q_val2 = list(
                    map(
                        lambda x: self.word_emb.get(
                            x, np.zeros(self.embed_size, dtype=np.float32),
                        ),
                        one_q,
                    )
                )
                print("Warning!")
                raise RuntimeWarning("Check this logic")
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        if w == "[db]":
                            emb_list.append(self.key_emb.weight[0])
                        elif w == "[table]":
                            emb_list.append(self.key_emb.weight[1])
                        elif w == "[column]":
                            emb_list.append(self.key_emb.weight[2])
                        elif w == "[value]":
                            emb_list.append(self.key_emb.weight[3])
                        else:
                            numpy_emb = self.word_emb.get(w, self.word_emb["unk"])
                            emb_list.append(torch.tensor(numpy_emb).cuda())
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        q_val.append(sum(emb_list) / float(ws_len))

            val_embs.append(q_val)
            val_len[i] = len(q_val)
        max_len = max(val_len)

        val_emb_array = torch.zeros((B, max_len, self.embed_size)).cuda()
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        return val_emb_array

    def forward(self, examples):
        # now should implement the examples
        batch = Batch(examples, is_cuda=self.is_cuda)
        if self.encoder_name == "ra_transformer":
            src = self.gen_x_batch(batch.src_sents)
            col = self.gen_x_batch(batch.table_sents)
            tab = self.gen_x_batch(batch.table_names)

            src_len = batch.src_sents_len
            col_len = [len(item) for item in batch.table_sents]
            tab_len = [len(item) for item in batch.table_names]

            src_mask = batch.src_token_mask
            col_mask = batch.table_token_mask
            tab_mask = batch.schema_token_mask

            relation_matrix = relation.create_batch(batch.relation)

            src_encodings, table_embeddings, schema_embeddings = \
                self.encoder(src, col, tab, src_len, col_len, tab_len, src_mask, col_mask, tab_mask, relation_matrix)
            enc_last_cell = None
        elif self.encoder_name == "transformer":
            (
                src_encodings,
                table_embeddings,
                schema_embeddings,
                _,
            ) = self.lstm_encode(batch)
            src_mask = batch.src_token_mask
            col_mask = batch.table_token_mask
            tab_mask = batch.schema_token_mask
            (
                src_encodings,
                table_embeddings,
                schema_embeddings
            ) = self.encoder(src_encodings, table_embeddings, schema_embeddings, src_mask, col_mask, tab_mask)
            enc_last_cell = None
        elif self.encoder_name == "bert":
            (
                src_encodings,
                table_embeddings,
                schema_embeddings,
                last_cell,
            ) = self.encoder(batch)
            if src_encodings is None:
                return None, None
            enc_last_cell = last_cell
        elif self.encoder_name == "lstm":
            encoded_out = self.encoder(batch)
            #encoded_out = self.encoder(examples)
            src_encodings = [item["src_encoding"] for item in encoded_out]
            table_embeddings = [item["col_encoding"] for item in encoded_out]
            schema_embeddings = [item["tab_encoding"] for item in encoded_out]
            enc_last_cell = torch.stack([item["last_cell"] for item in encoded_out])
        else:
            raise RuntimeError("Unsupported encoder name")

        if self.decoder_name == "lstm":
            src_mask = batch.src_token_mask
            col_mask = batch.table_token_mask
            tab_mask = batch.schema_token_mask
            col_tab_dic = batch.col_table_dict
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
            golds = [self.decoder.grammar.create_data(item) for item in batch.qgm]
            tmp = []
            for gold in golds:
                tmp += [
                    [
                        self.decoder.grammar.str_to_action(item)
                        for item in gold.split(" ")
                    ]
                ]
            golds = tmp
            losses, pred = self.decoder(
                enc_last_cell,
                src_encodings,
                table_embeddings,
                schema_embeddings,
                src_mask,
                col_mask,
                tab_mask,
                col_tab_dic,
                tab_col_dic,
                golds,
            )
            return losses
        elif self.decoder_name == "transformer":
            col_tab_dic = batch.col_table_dict
            golds = [self.decoder.grammar.create_data(item) for item in batch.qgm]
            tmp = []
            for gold in golds:
                tmp += [
                    [
                        self.decoder.grammar.str_to_action(item)
                        for item in gold.split(" ")
                    ]
                ]
            golds = tmp
            losses = self.decoder(
                src_encodings,
                table_embeddings,
                schema_embeddings,
                batch.src_sents_len,
                batch.col_num,
                batch.table_len,
                col_tab_dic,
                golds,
            )

            return losses
        elif self.decoder_name == "qgm":
            src_mask = batch.src_token_mask
            col_mask = batch.table_token_mask
            tab_mask = batch.schema_token_mask
            col_tab_dic = batch.col_table_dict
            b_indices = torch.arange(len(batch)).cuda()

            self.decoder.set_variables(
                self.is_bert,
                src_encodings,
                table_embeddings,
                schema_embeddings,
                src_mask,
                col_mask,
                tab_mask,
                col_tab_dic,
            )
            _, losses, pred_boxes = self.decoder.decode(
                b_indices, None, enc_last_cell, prev_box=None, gold_boxes=batch.qgm
            )
            return losses, pred_boxes
        elif self.decoder_name == "semql":
            col_tab_dic = batch.col_table_dict
            golds = [self.decoder.grammar.create_data(item) for item in batch.qgm]
            tmp = []
            for gold in golds:
                tmp += [[
                        self.decoder.grammar.str_to_action(item)
                        for item in gold.split(" ")
                        ]]
            golds = tmp
            losses = self.decoder(
                enc_last_cell,
                src_encodings,
                table_embeddings,
                schema_embeddings,
                col_tab_dic,
                golds,
            )
            return losses
        elif self.decoder_name == "ra_transformer":
            col_tab_dic = batch.col_table_dict
            golds = [self.decoder.grammar.create_data(item) for item in batch.qgm]
            tmp = []
            for gold in golds:
                tmp += [
                    [
                        self.decoder.grammar.str_to_action(item)
                        for item in gold.split(" ")
                    ]
                ]
            golds = tmp
            losses = self.decoder(src_encodings, table_embeddings, schema_embeddings, col_tab_dic, golds)
            return losses
        else:
            raise RuntimeError("Unsupported Decoder Name")

    def parse(self, examples):
        with torch.no_grad():
            batch = Batch(examples, is_cuda=self.is_cuda)
            if self.encoder_name == "ra_transformer":
                src = self.gen_x_batch(batch.src_sents)
                col = self.gen_x_batch(batch.table_sents)
                tab = self.gen_x_batch(batch.table_names)

                src_len = batch.src_sents_len
                col_len = [len(item) for item in batch.table_sents]
                tab_len = [len(item) for item in batch.table_names]

                src_mask = batch.src_token_mask
                col_mask = batch.table_token_mask
                tab_mask = batch.schema_token_mask

                relation_matrix = relation.create_batch(batch.relation)

                src_encodings, table_embeddings, schema_embeddings = \
                    self.encoder(src, col, tab, src_len, col_len, tab_len, src_mask, col_mask, tab_mask,
                                 relation_matrix)
            elif self.encoder_name == "transformer":
                (
                    src_encodings,
                    table_embeddings,
                    schema_embeddings,
                    _,
                ) = self.lstm_encode(batch)
                src_mask = batch.src_token_mask
                col_mask = batch.table_token_mask
                tab_mask = batch.schema_token_mask
                (
                    src_encodings,
                    table_embeddings,
                    schema_embeddings
                ) = self.encoder(src_encodings, table_embeddings, schema_embeddings, src_mask, col_mask, tab_mask)
            elif self.encoder_name == "bert":
                (
                    src_encodings,
                    table_embeddings,
                    schema_embeddings,
                    last_cell,
                ) = self.encoder(batch)
                if src_encodings is None:
                    return None, None
                enc_last_cell = last_cell
            elif self.encoder_name == "lstm":
                encoded_out = self.encoder(batch)
                #encoded_out = self.encoder(examples)
                src_encodings = [item["src_encoding"] for item in encoded_out]
                table_embeddings = [item["col_encoding"] for item in encoded_out]
                schema_embeddings = [item["tab_encoding"] for item in encoded_out]
                enc_last_cell = torch.stack([item["last_cell"] for item in encoded_out])
            else:
                raise RuntimeError("Unsupported encoder name")

            if self.decoder_name == "lstm":
                src_mask = batch.src_token_mask
                col_mask = batch.table_token_mask
                tab_mask = batch.schema_token_mask
                col_tab_dic = batch.col_table_dict
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
                golds = [self.decoder.grammar.create_data(item) for item in batch.qgm]
                tmp = []
                for gold in golds:
                    tmp += [
                        self.decoder.grammar.str_to_action(item)
                        for item in gold.split(" ")
                    ]
                golds = tmp
                losses, pred = self.decoder(
                    enc_last_cell,
                    src_encodings,
                    table_embeddings,
                    schema_embeddings,
                    src_mask,
                    col_mask,
                    tab_mask,
                    col_tab_dic,
                    tab_col_dic,
                    golds,
                )
                return pred
            elif self.decoder_name == "transformer":
                col_tab_dic = batch.col_table_dict
                pred = self.decoder(
                    src_encodings,
                    table_embeddings,
                    schema_embeddings,
                    batch.src_sents_len,
                    batch.col_num,
                    batch.table_len,
                    col_tab_dic,
                    golds=None,
                )

                return pred

            elif self.decoder_name == "qgm":
                src_mask = batch.src_token_mask
                col_mask = batch.table_token_mask
                tab_mask = batch.schema_token_mask
                col_tab_dic = batch.col_table_dict
                b_indices = torch.arange(len(batch)).cuda()
                self.decoder.set_variables(
                    self.is_bert,
                    src_encodings,
                    table_embeddings,
                    schema_embeddings,
                    src_mask,
                    col_mask,
                    tab_mask,
                    col_tab_dic,
                )
                _, losses, pred_boxes = self.decoder.decode(
                    b_indices, None, enc_last_cell, prev_box=None, gold_boxes=None
                )
                return pred_boxes
            elif self.decoder_name == "semql":
                col_tab_dic = batch.col_table_dict
                pred = self.decoder(
                    enc_last_cell,
                    src_encodings,
                    table_embeddings,
                    schema_embeddings,
                    col_tab_dic,
                    golds=None,
                )
                return pred
            elif self.decoder_name == "ra_transformer":
                col_tab_dic = batch.col_table_dict
                pred = self.decoder(src_encodings, table_embeddings, schema_embeddings, col_tab_dic, golds=None)
                return pred
            elif self.decoder_name == "ensemble":
                #Pass batch
                src = self.gen_x_batch(batch.src_sents)
                col = self.gen_x_batch(batch.table_sents)
                tab = self.gen_x_batch(batch.table_names)

                src_len = batch.src_sents_len
                col_len = [len(item) for item in batch.table_sents]
                tab_len = [len(item) for item in batch.table_names]

                src_mask = batch.src_token_mask
                col_mask = batch.table_token_mask
                tab_mask = batch.schema_token_mask

                relation_matrix = relation.create_batch(batch.relation)
                col_tab_dic = batch.col_table_dict

                pred = self.decoder(
                    src,
                    col,
                    tab,
                    src_len,
                    col_len,
                    tab_len,
                    src_mask,
                    col_mask,
                    tab_mask,
                    relation_matrix,
                    col_tab_dic,
                    batch.gt,
                )
                return pred
            else:
                raise RuntimeError("Unsupported decoder name")
