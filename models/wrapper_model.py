import numpy as np

# import random
import copy
import torch
import torch.nn as nn
import torch.nn.utils

# from src import utils
from src.dataset import Batch
from models.decoder.qgm.qgm_decoder import QGM_Decoder

from models.decoder.lstm.decoder import LSTM_Decoder
from models.decoder.transformer_framework.decoder import TransformerDecoderFramework

# from decoder.semql.semql_decoder import SemQL_Decoder
from models.decoder.semql_framework.decoder import SemQLDecoderFramework
from models.decoder.ra_transformer_framework.decoder import RATransformerDecoder
from models.decoder.ensemble.decoder import EnsembleDecoder

from models.encoder.ra_transformer.encoder import RA_Transformer_Encoder
from models.encoder.transformer.encoder import Transformer_Encoder
from models.encoder.lstm.encoder import LSTMEncoder
from models.encoder.bert.encoder import BERT

# from encoder.irnet.encoder import IRNetLSTMEncoder
import src.relation as relation


class EncoderDecoderModel(nn.Module):
    def __init__(self, cfg):
        super(EncoderDecoderModel, self).__init__()
        self.cfg = cfg
        self.is_cuda = cfg.cuda != -1
        self.is_bert = cfg.is_bert
        self.encoder_name = cfg.encoder_name
        self.decoder_name = cfg.decoder_name
        self.embed_size = 1024 if self.encoder_name == "bert" else 300

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
            # self.encoder = IRNetLSTMEncoder(cfg)
        elif self.encoder_name == "transformer":
            self.encoder = Transformer_Encoder(cfg)
        elif self.encoder_name == "ra_transformer":
            self.encoder = RA_Transformer_Encoder(cfg)
            if self.is_bert:
                self.bert = BERT(cfg)
        else:
            raise RuntimeError("Unsupported encoder name")

        # Key embeddings
        self.key_embs = nn.Embedding(6, 300).double().cuda()

        if self.encoder_name != "bert":
            self.without_bert_params = list(self.parameters(recurse=True))

    def load_model(self, pretrained_model_path):
        pretrained_model = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        pretrained_modeled = copy.deepcopy(pretrained_model)
        for k in pretrained_model.keys():
            if k not in self.state_dict().keys():
                del pretrained_modeled[k]
        self.load_state_dict(pretrained_model)

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
                        emb_list.append(self.key_embs.weight[0])
                    elif w == "[table]":
                        emb_list.append(self.key_embs.weight[1])
                    elif w == "[column]":
                        emb_list.append(self.key_embs.weight[2])
                    elif w == "[value]":
                        emb_list.append(self.key_embs.weight[3])
                    elif w == "[most]":
                        emb_list.append(self.key_embs.weight[4])
                    elif w == "[more]":
                        emb_list.append(self.key_embs.weight[5])
                    else:
                        if self.train_glove:
                            q_val.append(self.word_emb[self.mapping[w if w in self.mapping else "unk"]])
                        else:
                            q_val.append(self.word_emb.get(w, self.word_emb["unk"]))

                print("Warning!")
                raise RuntimeWarning("Check this logic")
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        if w == "[db]":
                            emb_list.append(self.key_embs.weight[0])
                        elif w == "[table]":
                             emb_list.append(self.key_embs.weight[1])
                        elif w == "[column]":
                            emb_list.append(self.key_embs.weight[2])
                        elif w == "[value]":
                            emb_list.append(self.key_embs.weight[3])
                        elif w == "[most]":
                            emb_list.append(self.key_embs.weight[4])
                        elif w == "[more]":
                            emb_list.append(self.key_embs.weight[5])
                        else:
                            if self.train_glove:
                                torch_emb = self.word_emb[self.mapping[w if w in self.mapping else "unk"]]
                                emb_list.append(torch_emb)
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

    def encode(self, batch, return_details=False):
        enc_last_cell = None
        if self.encoder_name == "ra_transformer":
            if self.is_bert:
                (
                    src,
                    col,
                    tab,
                    last_cell,
                ) = self.bert(batch)
            else:
                src = self.gen_x_batch(batch.src_sents)
                col = self.gen_x_batch(batch.table_sents)
                tab = self.gen_x_batch(batch.table_names)

            relation_matrix = relation.create_batch(batch.relation)

            output = self.encoder(
                src,
                col,
                tab,
                batch.src_sents_len,
                [len(item) for item in batch.table_sents],
                [len(item) for item in batch.table_names],
                batch.src_token_mask,
                batch.table_token_mask,
                batch.schema_token_mask,
                relation_matrix,
                return_details=return_details,
            )
            if return_details:
                src_encodings, table_embeddings, schema_embeddings, qk_weights_list, qk_relation_weights_list = output
            else:
                src_encodings, table_embeddings, schema_embeddings = output
        elif self.encoder_name == "transformer":
            (
                pre_src_encodings,
                pre_table_embeddings,
                pre_schema_embeddings,
                _,
            ) = self.lstm_encode(batch)
            (src_encodings, table_embeddings, schema_embeddings) = self.encoder(
                pre_src_encodings,
                pre_table_embeddings,
                pre_schema_embeddings,
                batch.src_token_mask,
                batch.table_token_mask,
                batch.schema_token_mask,
            )
        elif self.encoder_name == "bert":
            (
                src_encodings,
                table_embeddings,
                schema_embeddings,
                last_cell,
            ) = self.encoder(batch)
            enc_last_cell = last_cell
            qk_weights_list = qk_relation_weights_list = []
        elif self.encoder_name == "lstm":
            out = self.encoder(batch)
            src_encodings = [item["src_encoding"] for item in out]
            table_embeddings = [item["col_encoding"] for item in out]
            schema_embeddings = [item["tab_encoding"] for item in out]
            enc_last_cell = torch.stack([item["last_cell"] for item in out])
        else:
            raise RuntimeError("Unsupported encoder name")
        if return_details:
            return src_encodings, table_embeddings, schema_embeddings, enc_last_cell, qk_weights_list, qk_relation_weights_list
        else:
            return src_encodings, table_embeddings, schema_embeddings, enc_last_cell

    def decode(
        self,
        batch,
        src_encodings,
        table_embeddings,
        schema_embeddings,
        enc_last_cell,
        is_train=False,
        return_details=False,
    ):
        if self.decoder_name == "lstm":
            output = self.decoder(
                enc_last_cell,
                src_encodings,
                table_embeddings,
                schema_embeddings,
                batch.src_token_mask,
                batch.table_token_mask,
                batch.schema_token_mask,
                batch.col_tab_dic,
                batch.tab_col_dic,
                batch.gt if is_train else None,
            )
            return output
        elif self.decoder_name == "transformer":
            b_size = len(batch)
            output = self.decoder(
                b_size,
                src_encodings,
                table_embeddings,
                schema_embeddings,
                batch.src_sents_len,
                batch.col_num,
                batch.table_len,
                batch.col_tab_dic,
                batch.gt if is_train else None,
                return_details=return_details,
            )
            return output
        elif self.decoder_name == "semql":
            output = self.decoder(
                enc_last_cell,
                src_encodings,
                table_embeddings,
                schema_embeddings,
                batch.col_tab_dic,
                batch.gt if is_train else None,
            )
            return output
        elif self.decoder_name == "ra_transformer":
            output = self.decoder(
                src_encodings,
                table_embeddings,
                schema_embeddings,
                batch.col_tab_dic,
                batch.gt if is_train else None,
            )
            return output
        else:
            raise RuntimeError("Unsupported Decoder Name")
        return output

    def forward(self, examples, is_train=False, return_details=False):
        batch = Batch(examples, is_cuda=self.is_cuda, use_bert_cache=self.cfg.use_bert_cache)
        # Encode
        encoder_output = self.encode(
            batch, return_details=return_details,
        )
        if return_details:
            src_encodings, table_embeddings, schema_embeddings, enc_last_cell, qk_weights_list, qk_relation_weights_list = encoder_output
        else:
            src_encodings, table_embeddings, schema_embeddings, enc_last_cell = encoder_output
        # Decode
        decoder_output = self.decode(
            batch,
            src_encodings,
            table_embeddings,
            schema_embeddings,
            enc_last_cell,
            is_train=is_train,
            return_details=return_details,
        )
        if return_details:
            output, probs_list = decoder_output
            details = [{
                "qk_weights": qk_weights_list,
                "qk_relation_weights": qk_relation_weights_list,
                "probs": probs_list[0]
            }]
            return output, details
        else:
            return decoder_output

    def parse(self, examples, return_details=False):
        with torch.no_grad():
            pred = self.forward(examples, is_train=False, return_details=return_details)
            return pred
