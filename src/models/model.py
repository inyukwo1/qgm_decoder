# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : model.py
# @Software: PyCharm
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable

from src.beam import Beams, ActionInfo
from src.dataset import Batch
from src.models import nn_utils
from src.models.basic_model import BasicModel
from src.models.pointer_net import PointerNet
from src.rule import semQL as define_rule
from transformers import *
from qgm.qgm_decoder import QGM_Decoder


# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       'bert-large-uncased', 1024),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-large-cased', 1024),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased', 768),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-large', 1024),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]


class IRNet(BasicModel):
    
    def __init__(self, args, grammar):
        super(IRNet, self).__init__()
        self.args = args
        self.grammar = grammar
        self.use_column_pointer = args.column_pointer
        self.use_sentence_features = args.sentence_features

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor
        if args.bert != -1:
            model_class, tokenizer_class, pretrained_weight, dim = MODELS[args.bert]
            args.hidden_size = dim
            args.col_embed_size = dim
            args.embed_size = dim
            args.att_vec_size = dim
        self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size // 2, bidirectional=True,
                                    batch_first=True)


        input_dim = args.action_embed_size + \
                    args.att_vec_size + \
                    args.type_embed_size
        # previous action
        # input feeding
        # pre type embedding

        self.lf_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        self.att_sketch_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        self.prob_att = nn.Linear(args.att_vec_size, 1)
        self.prob_len = nn.Linear(1, 1)

        self.col_type = nn.Linear(9, args.col_embed_size)
        self.tab_type = nn.Linear(5, args.col_embed_size)
        self.sketch_encoder = nn.LSTM(args.action_embed_size, args.action_embed_size // 2, bidirectional=True,
                                      batch_first=True)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        self.att_project = nn.Linear(args.hidden_size + args.type_embed_size, args.hidden_size)

        self.N_embed = nn.Embedding(len(define_rule.N._init_grammar()), args.action_embed_size)

        self.read_out_act = F.tanh if args.readout == 'non_linear' else nn_utils.identity

        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        self.q_att = nn.Linear(args.hidden_size, args.embed_size)

        self.column_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.column_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        self.table_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        # QGM Decoer
        self.decoder = QGM_Decoder()

        self.without_bert_params = list(self.parameters(recurse=True))
        if args.bert != -1:
            model_class, tokenizer_class, pretrained_weight, dim = MODELS[args.bert]
            self.transformer_encoder = model_class.from_pretrained(pretrained_weight)
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
            # self.tokenizer.add_special_tokens({"additional_special_tokens": ["[table]", "[column]", "[value]"]})
            self.transformer_dim = dim
            self.col_lstm = torch.nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
            self.tab_lstm = torch.nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
            args.hidden_size = dim
            args.col_embed_size = dim

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.N_embed.weight.data)
        print('Use Column Pointer: ', True if self.use_column_pointer else False)
        
    def forward(self, examples):
        args = self.args
        # now should implement the examples
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        table_appear_mask = batch.table_appear_mask

        if args.bert == -1:
            src_encodings, (last_state, last_cell) = self.encode(batch.src_sents, batch.src_sents_len, None)

            src_encodings = self.dropout(src_encodings)

            table_embedding = self.gen_x_batch(batch.table_sents)
            src_embedding = self.gen_x_batch(batch.src_sents)
            schema_embedding = self.gen_x_batch(batch.table_names)
            # get emb differ
            embedding_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=table_embedding,
                                                     table_unk_mask=batch.table_unk_mask)

            schema_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=schema_embedding,
                                                  table_unk_mask=batch.schema_token_mask)

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
        else:
            src_encodings, table_embedding, schema_embedding, last_cell = self.transformer_encode(batch)
            if src_encodings is None:
                return None, None

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)

        tmp = self.decoder.decode(src_encodings, table_embedding, schema_embedding, dec_init_vec)
        return tmp

    def transformer_encode(self, batch: Batch):
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
            #question = "[CLS]"
            question = "<cls>"
            for word in sentences[b]:
                start = len(self.tokenizer.tokenize(question))
                for one_word in word:
                    question += " " + one_word
                end = len(self.tokenizer.tokenize(question))
                word_start_ends.append((start, end))
            col_start_ends = []
            for cols in col_sets[b]:
                start = len(self.tokenizer.tokenize(question))
                #question += " [SEP]"
                question += " <sep>"
                for one_word in cols:
                    question += " " + one_word
                end = len(self.tokenizer.tokenize(question))
                col_start_ends.append((start, end))
            tab_start_ends = []
            for tabs in table_sets[b]:
                start = len(self.tokenizer.tokenize(question))
                #question += " [SEP]"
                question += "<sep>"
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
            questions[idx] = questions[idx] + (" " + self.tokenizer.pad_token) * (max(question_lens) - question_len)
        encoded_questions = [self.tokenizer.encode(question, add_special_tokens=False) for question in questions]
        encoded_questions = torch.tensor(encoded_questions)
        if torch.cuda.is_available():
            encoded_questions = encoded_questions.cuda()
        embedding = self.transformer_encoder(encoded_questions)[0]
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
                lstm_out = self.col_lstm(inputs)[0].view(ed - st, 2, self.transformer_dim // 2)
                col_encoding = torch.cat((lstm_out[-1, 0], lstm_out[0, 1]))
                one_col_encodings.append(col_encoding)
            table_embedding.append(one_col_encodings)
            one_tab_encodings = []
            for st, ed in tab_start_end_batch[b]:
                inputs = embedding[b, st:ed].unsqueeze(0)
                lstm_out = self.tab_lstm(inputs)[0].view(ed - st, 2, self.transformer_dim // 2)
                tab_encoding = torch.cat((lstm_out[-1, 0], lstm_out[0, 1]))
                one_tab_encodings.append(tab_encoding)
            schema_embedding.append(one_tab_encodings)
        max_src_len = max([len(one_q_encodings) for one_q_encodings in src_encodings])
        max_col_len = max([len(one_col_encodings) for one_col_encodings in table_embedding])
        max_tab_len = max([len(one_tab_encodings) for one_tab_encodings in schema_embedding])
        for b in range(len(questions)):
            src_encodings[b] += [torch.zeros_like(src_encodings[b][0])] * (max_src_len - len(src_encodings[b]))
            src_encodings[b] = torch.stack(src_encodings[b])
            table_embedding[b] += [torch.zeros_like(table_embedding[b][0])] * (max_col_len - len(table_embedding[b]))
            table_embedding[b] = torch.stack(table_embedding[b])
            schema_embedding[b] += [torch.zeros_like(schema_embedding[b][0])] * (max_tab_len - len(schema_embedding[b]))
            schema_embedding[b] = torch.stack(schema_embedding[b])
        src_encodings = torch.stack(src_encodings)
        table_embedding = torch.stack(table_embedding)
        schema_embedding = torch.stack(schema_embedding)

        col_type = self.input_type(col_types)
        col_type_var = self.col_type(col_type)
        table_embedding = table_embedding + col_type_var

        return src_encodings, table_embedding, schema_embedding, embedding[:,0,:]

    def parse(self, examples, beam_size=5):
        """
        one example a time
        :param examples:
        :param beam_size:
        :return:
        """
        batch = Batch([examples], self.grammar, cuda=self.args.cuda)
        if self.args.bert == -1:
            src_encodings, (last_state, last_cell) = self.encode(batch.src_sents, batch.src_sents_len, None)

            src_encodings = self.dropout(src_encodings)

            table_embedding = self.gen_x_batch(batch.table_sents)
            src_embedding = self.gen_x_batch(batch.src_sents)
            schema_embedding = self.gen_x_batch(batch.table_names)
            # get emb differ
            embedding_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=table_embedding,
                                                     table_unk_mask=batch.table_unk_mask)

            schema_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=schema_embedding,
                                                  table_unk_mask=batch.schema_token_mask)

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
        else:
            src_encodings, table_embedding, schema_embedding, last_cell = self.transformer_encode(batch)
            if src_encodings is None:
                return None, None

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)

        # Begin Decoding
        tmp = self.decoder.decode(src_encodings, table_embedding, schema_embedding, dec_init_vec)
        return tmp

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None,
             return_att_weight=False):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = decoder(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        att_t = F.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = F.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())
