import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from transformers import *

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


class BERT(nn.Module):
    def __init__(self, cfg):
        super(BERT, self).__init__()
        self.cfg = cfg

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
        self.col_type = nn.Linear(9, self.transformer_dim)

    def input_type(self, values_list):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros(
            (B, max_len, values_list[0].shape[1]), dtype=np.float32
        )
        for i in range(B):
            val_emb_array[i, : val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array).cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def forward(self, batch):
        B = len(batch)
        sentences = batch.src_sents
        col_sets = batch.table_sents
        table_sets = batch.table_names

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
                # question += " <sep>"
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
