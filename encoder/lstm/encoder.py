import torch
import numpy as np
import torch.nn as nn

from src import utils

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMEncoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMEncoder, self).__init__()
        self.cfg = cfg
        self.embed_size = 300
        self.hidden_size = 300
        self.cuda = cfg.cuda != -1
        self.dropout = nn.Dropout(0.3)
        self.col_type = nn.Linear(9, self.embed_size)
        self.encoder_lstm = nn.LSTM(
            self.embed_size, self.hidden_size // 2, bidirectional=True, batch_first=True
        )

        self.word_emb = None
        self._import_word_emb()

    def set_word_emb(self, word_emb):
        self.word_emb = word_emb

    def _import_word_emb(self):
        self.word_emb = utils.load_word_emb(self.cfg.glove_embed_path)

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

        val_inp = torch.from_numpy(val_emb_array)
        if self.cuda:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def gen_x_batch(self, q):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        is_list = False
        if type(q[0][0]) == list:
            is_list = True
        for i, one_q in enumerate(q):
            if not is_list:
                q_val = list(
                    map(
                        lambda x: self.word_emb.get(
                            x, np.zeros(self.embed_size, dtype=np.float32)
                        ),
                        one_q,
                    )
                )
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_emb.get(w, self.word_emb["unk"]))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        q_val.append(sum(emb_list) / float(ws_len))

            val_embs.append(q_val)
            val_len[i] = len(q_val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.embed_size), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.cuda:
            val_inp = val_inp.cuda()
        return val_inp

    def embedding_cosine(self, src_embedding, table_embedding, table_unk_mask):
        embedding_differ = []
        for i in range(table_embedding.size(1)):
            one_table_embedding = table_embedding[:, i, :]
            one_table_embedding = one_table_embedding.unsqueeze(1).expand(
                table_embedding.size(0), src_embedding.size(1), table_embedding.size(2)
            )

            topk_val = F.cosine_similarity(one_table_embedding, src_embedding, dim=-1)

            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)
        embedding_differ.data.masked_fill_(
            table_unk_mask.unsqueeze(2)
            .expand(
                table_embedding.size(0),
                table_embedding.size(1),
                embedding_differ.size(2),
            )
            .bool(),
            0,
        )

        return embedding_differ

    def encode(self, src_sents_var, src_sents_len, q_onehot_project=None):
        src_token_embed = self.gen_x_batch(src_sents_var)

        if q_onehot_project is not None:
            src_token_embed = torch.cat([src_token_embed, q_onehot_project], dim=-1)

        packed_src_token_embed = pack_padded_sequence(
            src_token_embed, src_sents_len, batch_first=True
        )
        src_encodings, (last_state, last_cell) = self.encoder_lstm(
            packed_src_token_embed
        )
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def forward(self, batch):
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
        tab_ctx = (src_encodings.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
        schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        table_embedding = table_embedding + tab_ctx
        schema_embedding = schema_embedding + schema_ctx
        col_type = self.input_type(batch.col_hot_type)
        col_type_var = self.col_type(col_type)
        table_embedding = table_embedding + col_type_var

        src_encoding_list = list(torch.unbind(src_encodings))
        col_encoding_list = list(torch.unbind(table_embedding))
        tab_encoding_list = list(torch.unbind(schema_embedding))
        last_cell_lsit = list(torch.unbind(last_state))

        return [
            {
                "src_encoding": item1,
                "col_encoding": item2,
                "tab_encoding": item3,
                "last_cell": item4,
            }
            for item1, item2, item3, item4 in zip(
                src_encoding_list, col_encoding_list, tab_encoding_list, last_cell_lsit
            )
        ]
