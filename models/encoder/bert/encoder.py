import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from transformers import *

import os
import pickle
from tqdm import tqdm

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
        self.use_bert_cache = cfg.use_bert_cache
        self.bert_cache = {}

        self.reduce_dim = cfg.reduce_dim
        if self.reduce_dim:
            self.linear_layer = nn.Linear(dim, 300)

    def create_cache(self, examples_list):
        if not self.use_bert_cache:
            return None

        for examples in examples_list:
            for idx in tqdm(range(len(examples))):
                example = examples[idx]
                question = example.bert_input
                encoded_question = self.tokenizer.encode(question, add_special_tokens=False)
                encoded_question = torch.tensor(encoded_question).unsqueeze(0).cuda()
                embedding = self.bert_encoder(encoded_question)[0].squeeze(0).cpu().detach().numpy()

                # indices
                word_start_end, col_start_end, tab_start_end = example.bert_input_indices

                # split question
                question_encodings = [sum(embedding[st:ed]) / (ed-st) for st, ed in word_start_end]

                # split col
                cols = [np.expand_dims(embedding[st:ed], axis=0)for st, ed in col_start_end]

                # Split tab
                tabs = [np.expand_dims(embedding[st:ed], axis=0) for st, ed in tab_start_end]

                self.bert_cache[question] = [question_encodings, cols, tabs]

        cache_path = "./data"
        cache_name = "bert_cache.pkl"
        save_path = os.path.join(cache_path, cache_name)
        if cache_name in os.listdir("./data"):
            with open(save_path, "rb") as f:
                self.bert_cache = pickle.load(f)
        else:
            with open(save_path, "wb") as f:
                pickle.dump(self.bert_cache, f)

    def input_type(self, values_list):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array).cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def forward(self, batch):
        B = len(batch)
        sentences = batch.src_sents
        col_sets = batch.table_sents
        table_sets = batch.table_names

        # Process
        use_preprocessed = True
        if use_preprocessed:
            questions = batch.bert_input
            word_start_end_batch = [item[0] for item in batch.bert_input_indices]
            col_start_end_batch = [item[1] for item in batch.bert_input_indices]
            tab_start_end_batch = [item[2] for item in batch.bert_input_indices]
        else:
            questions = []
            question_lens = []
            word_start_end_batch = []
            col_start_end_batch = []
            tab_start_end_batch = []
            for b in range(B):
                word_start_ends = []
                question = "[CLS]"
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
                    for one_word in cols:
                        question += " " + one_word
                    end = len(self.tokenizer.tokenize(question))
                    col_start_ends.append((start, end))
                tab_start_ends = []
                for tabs in table_sets[b]:
                    start = len(self.tokenizer.tokenize(question))
                    question += " [SEP]"
                    for one_word in tabs:
                        question += " " + one_word
                    end = len(self.tokenizer.tokenize(question))
                    tab_start_ends.append((start, end))
                if end >= self.tokenizer.max_len:
                    print("xxxxxxxxxx")
                    exit(-1)
                question_lens.append(end)
                questions.append(question)
                word_start_end_batch.append(word_start_ends)
                col_start_end_batch.append(col_start_ends)
                tab_start_end_batch.append(tab_start_ends)
            if not questions:
                return None, None, None, None
            for idx, question_len in enumerate(question_lens):
                questions[idx] = questions[idx] + (" " + self.tokenizer.pad_token) * (
                    max(question_lens) - question_len
                )
        # Encode
        src_encodings = []
        table_embedding = []
        schema_embedding = []
        if self.use_bert_cache:
            embedding = []
            for b_idx, question in enumerate(questions):
                # question_without_pad = question.replace("[PAD]", "").strip()
                # one_q_encodings, col_encodings, tab_encodings = self.bert_cache[question_without_pad]
                one_q_encodings, col_encodings, tab_encodings = self.bert_cache[question]

                one_q_encodings = torch.tensor(one_q_encodings).cuda()
                col_encodings = [torch.tensor(item).cuda() for item in col_encodings]
                tab_encodings = [torch.tensor(item).cuda() for item in tab_encodings]

                # question
                src_encodings.append(list(torch.unbind(one_q_encodings, 0)))

                # column
                one_col_encodings = []
                for item in col_encodings:
                    output, (h_n, c_n) = self.col_lstm(item)
                    one_col_encodings.append(h_n.view(-1))
                table_embedding.append(one_col_encodings)

                # table
                one_tab_encodings = []
                for item in tab_encodings:
                    output, (h_n, c_n) = self.tab_lstm(item)
                    one_tab_encodings.append(h_n.view(-1))
                schema_embedding.append(one_tab_encodings)

        else:
            encoded_questions = [
                self.tokenizer.encode(question, add_special_tokens=False)
                for question in questions
            ]
            encoded_questions = torch.tensor(encoded_questions)
            if torch.cuda.is_available():
                encoded_questions = encoded_questions.cuda()
            embedding = self.bert_encoder(encoded_questions)[0]
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
                    inputs = embedding[b][st:ed].unsqueeze(0)
                    lstm_out = self.col_lstm(inputs)[0].view(
                        ed - st, 2, self.transformer_dim // 2
                    )
                    col_encoding = torch.cat((lstm_out[-1, 0], lstm_out[0, 1]))
                    one_col_encodings.append(col_encoding)
                table_embedding.append(one_col_encodings)
                one_tab_encodings = []
                for st, ed in tab_start_end_batch[b]:
                    inputs = embedding[b][st:ed].unsqueeze(0)
                    lstm_out = self.tab_lstm(inputs)[0].view(
                        ed - st, 2, self.transformer_dim // 2
                    )
                    tab_encoding = torch.cat((lstm_out[-1, 0], lstm_out[0, 1]))
                    one_tab_encodings.append(tab_encoding)
                schema_embedding.append(one_tab_encodings)

        # Batch padding
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

        col_type = self.input_type(batch.col_hot_type)
        col_type_var = self.col_type(col_type)
        table_embedding = table_embedding + col_type_var

        if self.reduce_dim:
            src_encodings = self.linear_layer(src_encodings)
            table_embedding = self.linear_layer(table_embedding)
            schema_embedding = self.linear_layer(schema_embedding)

        return src_encodings, table_embedding, schema_embedding, [item[0, :] for item in embedding]
