# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable

from src.dataset import Batch
from src.models.basic_model import BasicModel
from qgm.qgm_decoder import QGM_Decoder
from semql.semql_decoder import SemQL_Decoder
from transformers import *
from qgm_transformer.decoder import QGM_Transformer_Decoder

# from encoder.encoder import transformer_encoder


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
        self.decoder_name = cfg.model_name

        if self.is_cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
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
        # self.table_linear = nn.Linear(self.embed_size, hidden_size)
        # self.schema_linear = nn.Linear(self.embed_size, hidden_size)

        self.decoder_cell_init = nn.Linear(hidden_size, hidden_size)

        self.col_type = nn.Linear(9, hidden_size)
        self.tab_type = nn.Linear(5, hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.captum_iden = nn.Dropout(0.000001)

        # QGM Decoder
        if self.decoder_name == "qgm_transformer":
            self.decoder = QGM_Transformer_Decoder(cfg)
        elif self.decoder_name == "qgm":
            self.decoder = QGM_Decoder(cfg)
        elif self.decoder_name == "semql":
            self.decoder = SemQL_Decoder(cfg)
        else:
            raise RuntimeError("Unsupported decoder name")

        self.without_bert_params = list(self.parameters(recurse=True))
        if self.is_bert:
            model_class, tokenizer_class, pretrained_weight, dim = MODELS[cfg.bert]
            self.bert_encoder = model_class.from_pretrained(pretrained_weight)
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
            # self.tokenizer.add_special_tokens({"additional_special_tokens": ["[table]", "[column]", "[value]"]})
            self.transformer_dim = dim
            self.col_lstm = torch.nn.LSTM(
                dim, dim // 2, batch_first=True, bidirectional=True
            )
            self.tab_lstm = torch.nn.LSTM(
                dim, dim // 2, batch_first=True, bidirectional=True
            )

    def bert_encode(self, batch: Batch):
        B = len(batch)
        sentences = batch.src_sents
        col_sets = batch.table_sents
        table_sets = batch.table_names_iter

        questions = []
        question_lens = []
        word_start_end_batch = []
        col_start_end_batch = []
        tab_start_end_batch = []
        col_types = []
        for b in range(B):
            word_start_ends = []
            question = "[CLS]"
            # question = "<cls>"
            for word in sentences[b]:
                start = len(self.tokenizer.tokenize(question))
                for one_word in word:
                    question += " " + one_word
                end = len(self.tokenizer.tokenize(question))
                word_start_ends.append((start, end))
            col_start_ends = []
            for cols in col_sets[b]:
                start = len(self.tokenizer.tokenize(question))
                question += " [SEP]"
                # question += " <sep>"
                for one_word in cols:
                    question += " " + one_word
                end = len(self.tokenizer.tokenize(question))
                col_start_ends.append((start, end))
            tab_start_ends = []
            for tabs in table_sets[b]:
                start = len(self.tokenizer.tokenize(question))
                question += " [SEP]"
                # question += "<sep>"
                for one_word in tabs:
                    question += " " + one_word
                end = len(self.tokenizer.tokenize(question))
                tab_start_ends.append((start, end))
            if end >= self.tokenizer.max_len:
                print("xxxxxxxxxx")
                continue
            col_types.append(batch.col_hot_type[b])
            question_lens.append(end)
            questions.append(question)
            word_start_end_batch.append(word_start_ends)
            col_start_end_batch.append(col_start_ends)
            tab_start_end_batch.append(tab_start_ends)
        if not questions:
            return None, None, None
        for idx, question_len in enumerate(question_lens):
            questions[idx] = questions[idx] + (" " + self.tokenizer.pad_token) * (
                max(question_lens) - question_len
            )
        encoded_questions = [
            self.tokenizer.encode(question, add_special_tokens=False)
            for question in questions
        ]
        encoded_questions = torch.tensor(encoded_questions)
        if torch.cuda.is_available():
            encoded_questions = encoded_questions.cuda()
        embedding = self.bert_encoder(encoded_questions)[0]
        src_encodings = []
        table_embedding = []
        schema_embedding = []
        for b in range(len(questions)):
            one_q_encodings = []
            for st, ed in word_start_end_batch[b]:
                sum_tensor = torch.zeros_like(embedding[b][st])
                for i in range(st, ed):
                    sum_tensor = sum_tensor + embedding[b][i]
                sum_tensor = sum_tensor / (ed - st)
                one_q_encodings.append(sum_tensor)
            src_encodings.append(one_q_encodings)
            one_col_encodings = []
            for st, ed in col_start_end_batch[b]:
                inputs = embedding[b, st:ed].unsqueeze(0)
                lstm_out = self.col_lstm(inputs)[0].view(
                    ed - st, 2, self.transformer_dim // 2
                )
                col_encoding = torch.cat((lstm_out[-1, 0], lstm_out[0, 1]))
                one_col_encodings.append(col_encoding)
            table_embedding.append(one_col_encodings)
            one_tab_encodings = []
            for st, ed in tab_start_end_batch[b]:
                inputs = embedding[b, st:ed].unsqueeze(0)
                lstm_out = self.tab_lstm(inputs)[0].view(
                    ed - st, 2, self.transformer_dim // 2
                )
                tab_encoding = torch.cat((lstm_out[-1, 0], lstm_out[0, 1]))
                one_tab_encodings.append(tab_encoding)
            schema_embedding.append(one_tab_encodings)
        max_src_len = max([len(one_q_encodings) for one_q_encodings in src_encodings])
        max_col_len = max(
            [len(one_col_encodings) for one_col_encodings in table_embedding]
        )
        max_tab_len = max(
            [len(one_tab_encodings) for one_tab_encodings in schema_embedding]
        )
        for b in range(len(questions)):
            src_encodings[b] += [torch.zeros_like(src_encodings[b][0])] * (
                max_src_len - len(src_encodings[b])
            )
            src_encodings[b] = torch.stack(src_encodings[b])
            table_embedding[b] += [torch.zeros_like(table_embedding[b][0])] * (
                max_col_len - len(table_embedding[b])
            )
            table_embedding[b] = torch.stack(table_embedding[b])
            schema_embedding[b] += [torch.zeros_like(schema_embedding[b][0])] * (
                max_tab_len - len(schema_embedding[b])
            )
            schema_embedding[b] = torch.stack(schema_embedding[b])
        src_encodings = torch.stack(src_encodings)
        table_embedding = torch.stack(table_embedding)
        schema_embedding = torch.stack(schema_embedding)

        col_type = self.input_type(col_types)
        col_type_var = self.col_type(col_type)
        table_embedding = table_embedding + col_type_var

        return src_encodings, table_embedding, schema_embedding, embedding[:, 0, :]

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

        # table_embedding = self.table_linear(table_embedding)
        # schema_embedding = self.schema_linear(schema_embedding)

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

        if self.decoder_name == "qgm_transformer":
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
        elif self.decoder_name == "semql":
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

    def forward(self, examples):
        # now should implement the examples
        batch = Batch(examples, is_cuda=self.is_cuda)

        src = self.gen_x_batch(batch.src_sents)
        col = self.gen_x_batch(batch.table_sents)
        tab = self.gen_x_batch(batch.table_names_iter)
        # src_encodings, table_embedding, schema_embedding = transformer_encoder(
        #     src,
        #     col,
        #     tab,
        #     batch.src_token_mask,
        #     batch.table_token_mask,
        #     batch.schema_token_mask,
        # )

        if self.is_bert:
            (
                src_encodings,
                table_embedding,
                schema_embedding,
                last_cell,
            ) = self.bert_encode(batch)
            if src_encodings is None:
                return None, None
        else:
            (
                src_encodings,
                table_embedding,
                schema_embedding,
                last_cell,
            ) = self.lstm_encode(batch)
        src_encodings = self.captum_iden(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)

        if self.decoder_name == "qgm_transformer":
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

            losses, pred = self.decoder(
                dec_init_vec,
                src_encodings,
                table_embedding,
                schema_embedding,
                src_mask,
                col_mask,
                tab_mask,
                tab_col_dic,
                batch.qgm_action,
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
        elif self.decoder_name == "semql":
            lf_prob_var, _ = self.decoder.decode_forward(
                examples,
                batch,
                src_encodings,
                table_embedding,
                schema_embedding,
                dec_init_vec,
            )
            return lf_prob_var
        else:
            raise RuntimeError("Unsupported Decoder Name")

    def parse(self, examples):
        with torch.no_grad():
            batch = Batch(examples, is_cuda=self.is_cuda)
            if self.is_bert:
                (
                    src_encodings,
                    table_embedding,
                    schema_embedding,
                    last_cell,
                ) = self.bert_encode(batch)
                if src_encodings is None:
                    return None, None
            else:
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

                # table_embedding = self.table_linear(table_embedding)
                # schema_embedding = self.schema_linear(schema_embedding)

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

            if self.decoder_name == "qgm_transformer":
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

                losses, pred = self.decoder(
                    dec_init_vec,
                    src_encodings,
                    table_embedding,
                    schema_embedding,
                    src_mask,
                    col_mask,
                    tab_mask,
                    tab_col_dic,
                    batch.qgm_action,
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
            elif self.decoder_name == "semql":
                _, highest_prob_actions = self.decoder.decode_forward(
                    None,
                    batch,
                    src_encodings,
                    table_embedding,
                    schema_embedding,
                    dec_init_vec,
                )
                return highest_prob_actions[0]
            else:
                raise RuntimeError("Unsupported decoder name")

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())
