# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable

from src.dataset import Batch
from src.models.basic_model import BasicModel
from decoder.qgm.qgm_decoder import QGM_Decoder
#from decoder.semql.semql_decoder import SemQL_Decoder
from transformers import *
from decoder.lstm.decoder import LSTM_Decoder
from decoder.transformer.decoder import Transformer_Decoder
from encoder.transformer.encoder import Transformer_Encoder
from encoder.ra_transformer.encoder import RA_Transformer_Encoder
import src.relation as relation

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [
    (BertModel, BertTokenizer, "bert-large-uncased", 1024),
    (OpenAIGPTModel, OpenAIGPTTokenizer, "openai-gpt"),
    (GPT2Model, GPT2Tokenizer, "gpt2"),
    (CTRLModel, CTRLTokenizer, "ctrl"),
    (TransfoXLModel, TransfoXLTokenizer, "transfo-xl-wt103"),
    (XLNetModel, XLNetTokenizer, "xlnet-large-cased", 1024),
    (XLNetModel, XLNetTokenizer, "xlnet-base-cased", 768),
    (XLMModel, XLMTokenizer, "xlm-mlm-enfr-1024"),
    (DistilBertModel, DistilBertTokenizer, "distilbert-base-uncased"),
    (RobertaModel, RobertaTokenizer, "roberta-large", 1024),
    (RobertaModel, RobertaTokenizer, "roberta-base"),
]


class IRNet(BasicModel):
    def __init__(self, cfg):
        super(IRNet, self).__init__()
        self.cfg = cfg
        self.is_bert = cfg.is_bert
        self.is_cuda = cfg.cuda != -1
        self.encoder_name = cfg.encoder_name
        self.decoder_name = cfg.decoder_name

        if self.is_cuda:
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_tensor = torch.FloatTensor

        if self.is_bert:
            model_class, tokenizer_class, pretrained_weight, dim = MODELS[cfg.bert]
            cfg.att_vec_size = dim
            self.embed_size = dim
        else:
            self.embed_size = 300

        hidden_size = cfg.hidden_size

        self.encoder_lstm = nn.LSTM(
            self.embed_size, hidden_size // 2, bidirectional=True, batch_first=True,
        )

        self.decoder_cell_init = nn.Linear(hidden_size, hidden_size)

        self.col_type = nn.Linear(9, hidden_size)
        self.tab_type = nn.Linear(5, hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.captum_iden = nn.Dropout(0.000001)

        # Decoder
        if self.decoder_name == "transformer":
            self.decoder = Transformer_Decoder(cfg)
        elif self.decoder_name == "lstm":
            self.decoder = LSTM_Decoder(cfg)
        elif self.decoder_name == "qgm":
            self.decoder = QGM_Decoder(cfg)
        elif self.decoder_name == "semql":
            #self.decoder = SemQL_Decoder(cfg)
            pass
        else:
            raise RuntimeError("Unsupported decoder name")

        self.without_bert_params = list(self.parameters(recurse=True))

        # Encoder
        if self.encoder_name == "bert":
            self.encoder = None
        elif self.encoder_name == "lstm":
            self.encoder = None
        elif self.encoder_name == "transformer":
            self.encoder = Transformer_Encoder(cfg)
        elif self.encoder_name == "ra_transformer":
            self.encoder = RA_Transformer_Encoder(cfg)
        else:
            raise RuntimeError("Unsupported encoder name")

        if self.encoder_name != "bert":
            self.without_bert_params = list(self.parameters(recurse=True))

    def lstm_encode(self, batch, src_embedding=None):
        src_encodings, (last_state, last_cell) = self.encode(
            batch.src_sents, batch.src_sents_len, None, src_token_embed=src_embedding
        )

        src_encodings = self.dropout(src_encodings)
        if src_embedding is None:
            src_embedding = self.gen_x_batch(batch.src_sents)

        table_embedding = self.gen_x_batch(batch.table_sents)
        schema_embedding = self.gen_x_batch(batch.table_names)

        # get emb differ
        embedding_differ = self.embedding_cosine(
            src_embedding=src_embedding,
            table_embedding=table_embedding,
            table_unk_mask=batch.table_unk_mask,
        )

        schema_differ = self.embedding_cosine(
            src_embedding=src_embedding,
            table_embedding=schema_embedding,
            table_unk_mask=batch.schema_token_mask,
        )

        tab_ctx = (src_encodings.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
        schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        table_embedding = table_embedding + tab_ctx

        schema_embedding = schema_embedding + schema_ctx

        col_type = self.input_type(batch.col_hot_type)

        col_type_var = self.col_type(col_type)

        tab_type = self.input_type(batch.tab_hot_type)

        tab_type_var = self.tab_type(tab_type)

        table_embedding = table_embedding + col_type_var

        schema_embedding = schema_embedding + tab_type_var
        return src_encodings, table_embedding, schema_embedding, last_cell


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

            src_encodings, table_embedding, schema_embedding = \
                self.encoder(src, col, tab, src_len, col_len, tab_len, src_mask, col_mask, tab_mask, relation_matrix)

        elif self.encoder_name == "transformer":
            (
                src_encodings,
                table_embedding,
                schema_embedding,
                _,
            ) = self.lstm_encode(batch)
            src_mask = batch.src_token_mask
            col_mask = batch.table_token_mask
            tab_mask = batch.schema_token_mask
            (
                src_encodings,
                table_embeddings,
                schema_embeddings
            ) = self.encoder(src_encodings, table_embedding, schema_embedding, src_mask, col_mask, tab_mask)
        elif self.encoder_name == "bert":
            (
                src_encodings,
                table_embedding,
                schema_embedding,
                last_cell,
            ) = self.encoder(batch)
            if src_encodings is None:
                return None, None
            dec_init_vec = self.init_decoder_state(last_cell)
        elif self.encoder_name == "lstm":
            (
                src_encodings,
                table_embedding,
                schema_embedding,
                last_cell,
            ) = self.lstm_encode(batch)
            dec_init_vec = self.init_decoder_state(last_cell)
        else:
            raise RuntimeError("Unsupported encoder name")
        src_encodings = self.captum_iden(src_encodings)

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
                tmp += [[self.decoder.grammar.str_to_action(item) for item in gold.split(" ")]]
            golds = tmp
            losses, pred = self.decoder(
                                dec_init_vec,
                                src_encodings,
                                table_embedding,
                                schema_embedding,
                                src_mask,
                                col_mask,
                                tab_mask,
                                col_tab_dic,
                                tab_col_dic,
                                golds,
                                )
            return losses
        elif self.decoder_name == "transformer":
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
                tmp += [[self.decoder.grammar.str_to_action(item) for item in gold.split(" ")]]
            golds = tmp
            losses, pred = self.decoder(
                src_encodings,
                table_embedding,
                schema_embedding,
                src_mask,
                col_mask,
                tab_mask,
                col_tab_dic,
                tab_col_dic,
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
                table_embedding,
                schema_embedding,
                src_mask,
                col_mask,
                tab_mask,
                col_tab_dic,
            )
            _, losses, pred_boxes = self.decoder.decode(
                b_indices, None, dec_init_vec, prev_box=None, gold_boxes=batch.qgm
            )
            return losses, pred_boxes
        elif self.decoder_name == "preprocess":
            sketch_prob_var, lf_prob_var = self.decoder.decode_forward(
                examples,
                batch,
                src_encodings,
                table_embedding,
                schema_embedding,
                dec_init_vec,
            )
            return sketch_prob_var, lf_prob_var
        else:
            raise RuntimeError("Unsupported Decoder Name")

    def parse(self, examples):
        with torch.no_grad():
            batch = Batch(examples, is_cuda=self.is_cuda)
            if self.encoder_name == "ra_transformer":
                src = batch.src_sents
                col = self.gen_x_batch(batch.table_sents)
                tab = self.gen_x_batch(batch.table_names)

                src_len = batch.src_sents_len
                col_len = [len(item) for item in batch.table_sents]
                tab_len = [len(item) for item in batch.table_names]

                src_mask = batch.src_token_mask
                col_mask = batch.table_token_mask
                tab_mask = batch.schema_token_mask

                relation_matrix = relation.create_batch(batch.relation)

                src_encodings, table_embedding, schema_embedding = \
                    self.encoder(src, col, tab, src_len, col_len, tab_len, src_mask, col_mask, tab_mask,
                                 relation_matrix)
            elif self.encoder_name == "transformer":
                (
                    src_encodings,
                    table_embedding,
                    schema_embedding,
                    _,
                ) = self.lstm_encode(batch)
                src_mask = batch.src_token_mask
                col_mask = batch.table_token_mask
                tab_mask = batch.schema_token_mask
                (
                    src_encodings,
                    table_embeddings,
                    schema_embeddings
                ) = self.encoder(src_encodings, table_embedding, schema_embedding, src_mask, col_mask, tab_mask)
            elif self.encoder_name == "bert":
                (
                    src_encodings,
                    table_embedding,
                    schema_embedding,
                    last_cell,
                ) = self.encoder(batch)
                if src_encodings is None:
                    return None, None
                dec_init_vec = self.init_decoder_state(last_cell)
            elif self.encoder_name == "lstm":
                src_encodings, (last_state, last_cell) = self.encode(
                    batch.src_sents, batch.src_sents_len, None
                )

                src_encodings = self.dropout(src_encodings)

                table_embedding = self.gen_x_batch(batch.table_sents)
                src_embedding = self.gen_x_batch(batch.src_sents)
                schema_embedding = self.gen_x_batch(batch.table_names)
                # get emb differ
                embedding_differ = self.embedding_cosine(
                    src_embedding=src_embedding,
                    table_embedding=table_embedding,
                    table_unk_mask=batch.table_unk_mask,
                )

                schema_differ = self.embedding_cosine(
                    src_embedding=src_embedding,
                    table_embedding=schema_embedding,
                    table_unk_mask=batch.schema_token_mask,
                )

                tab_ctx = (
                    src_encodings.unsqueeze(1) * embedding_differ.unsqueeze(3)
                ).sum(2)
                schema_ctx = (
                    src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)
                ).sum(2)

                table_embedding = table_embedding + tab_ctx

                schema_embedding = schema_embedding + schema_ctx

                col_type = self.input_type(batch.col_hot_type)

                col_type_var = self.col_type(col_type)

                tab_type = self.input_type(batch.tab_hot_type)

                tab_type_var = self.tab_type(tab_type)

                table_embedding = table_embedding + col_type_var

                schema_embedding = schema_embedding + tab_type_var

                dec_init_vec = self.init_decoder_state(last_cell)
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
                    tmp += [self.decoder.grammar.str_to_action(item) for item in gold.split(" ")]
                golds = tmp
                losses, pred = self.decoder(dec_init_vec,
                                            src_encodings,
                                            table_embedding,
                                            schema_embedding,
                                            src_mask,
                                            col_mask,
                                            tab_mask,
                                            col_tab_dic,
                                            tab_col_dic,
                                            golds,
                                            )
                return pred
            elif self.decoder_name == "transformer":
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
                    tmp += [self.decoder.grammar.str_to_action(item) for item in gold.split(" ")]
                golds = tmp
                losses, pred = self.decoder(
                    src_encodings,
                    table_embedding,
                    schema_embedding,
                    src_mask,
                    col_mask,
                    tab_mask,
                    col_tab_dic,
                    tab_col_dic,
                    golds,
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
                    table_embedding,
                    schema_embedding,
                    src_mask,
                    col_mask,
                    tab_mask,
                    col_tab_dic,
                )
                _, losses, pred_boxes = self.decoder.decode(
                    b_indices, None, dec_init_vec, prev_box=None, gold_boxes=None
                )

                return pred_boxes
            elif self.decoder_name == "preprocess":
                completed_beams, _ = self.decoder.decode_parse(
                    batch,
                    src_encodings,
                    table_embedding,
                    schema_embedding,
                    dec_init_vec,
                    beam_size=5,
                )
                highest_prob_actions = (
                    completed_beams[0].actions if completed_beams else []
                )
                return highest_prob_actions
            else:
                raise RuntimeError("Unsupported decoder name")

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())



    def forward_until_step(self, src_embeddings, examples, step):
        batch = Batch(examples, is_cuda=self.is_cuda)

        if self.is_bert:
            (
                src_encodings,
                table_embedding,
                schema_embedding,
                last_cell,
            ) = self.bert_encode(batch)
        else:
            (
                src_encodings,
                table_embedding,
                schema_embedding,
                last_cell,
            ) = self.lstm_encode(batch, src_embeddings)
            if src_encodings is None:
                return None, None
        src_encodings = self.captum_iden(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)

        if self.decoder_name == "transformer":
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

            gold_action, pred_action, gold_score, pred_score = self.decoder(
                src_encodings,
                table_embedding,
                schema_embedding,
                src_mask,
                col_mask,
                tab_mask,
                tab_col_dic,
                batch.qgm_action,
                step,
                batch.table_sents,
                batch.table_names_iter,
            )

        elif self.decoder_name == "qgm":
            src_mask = batch.src_token_mask
            col_mask = batch.table_token_mask
            tab_mask = batch.schema_token_mask
            col_tab_dic = batch.col_table_dict
            b_indices = torch.arange(len(batch)).cuda()

            self.decoder.set_variables(
                self.is_bert,
                src_encodings,
                table_embedding,
                schema_embedding,
                src_mask,
                col_mask,
                tab_mask,
                col_tab_dic,
                step,
            )
            pred_action, pred_score = self.decoder.decode_until_step(
                b_indices, None, dec_init_vec, prev_box=None, gold_boxes=batch.qgm
            )
        elif self.decoder_name == "preprocess":
            (
                gold_action,
                pred_action,
                gold_score,
                pred_score,
            ) = self.decoder.decode_forward(
                examples,
                batch,
                src_encodings,
                table_embedding,
                schema_embedding,
                dec_init_vec,
                step,
            )
        else:
            raise RuntimeError("Unsupported Decoder name")

        return gold_action, pred_action, gold_score, pred_score