import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable

from src.dataset import Batch
from src.models.basic_model import BasicModel
from decoder.qgm.qgm_decoder import QGM_Decoder

from decoder.lstm.decoder import LSTM_Decoder
from decoder.transformer_framework.decoder import TransformerDecoderFramework

from encoder.ra_transformer.encoder import RA_Transformer_Encoder
from encoder.transformer.encoder import Transformer_Encoder
import src.relation as relation


class IRNet(BasicModel):
    def __init__(self, cfg):
        super(IRNet, self).__init__()
        self.cfg = cfg
        self.is_bert = cfg.is_bert
        self.is_cuda = cfg.cuda != -1
        self.use_col_set = cfg.is_col_set
        self.encoder_name = cfg.encoder_name
        self.decoder_name = cfg.decoder_name
        self.embed_size = 1024 if self.is_bert else 300
        hidden_size = cfg.hidden_size

        if self.is_cuda:
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_tensor = torch.FloatTensor

        self.decoder_cell_init = nn.Linear(hidden_size, hidden_size)

        self.col_type = nn.Linear(9, hidden_size)
        self.tab_type = nn.Linear(5, hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

        self.decoder_cell_init = nn.Linear(hidden_size, hidden_size)

        self.col_type = nn.Linear(9, hidden_size)
        self.tab_type = nn.Linear(5, hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

        # Decoder
        if self.decoder_name == "transformer":
            self.decoder = TransformerDecoderFramework(cfg)
        elif self.decoder_name == "lstm":
            self.decoder = LSTM_Decoder(cfg)
        elif self.decoder_name == "qgm":
            self.decoder = QGM_Decoder(cfg)
        elif self.decoder_name == "semql":
            #self.decoder = SemQL_Decoder(cfg)
            raise RuntimeError("Not yet")
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
                table_embedding,
                schema_embedding,
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
                    tmp += [
                        self.decoder.grammar.str_to_action(item)
                        for item in gold.split(" ")
                    ]
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
                return pred
            elif self.decoder_name == "transformer":
                col_tab_dic = batch.col_table_dict
                pred = self.decoder(
                    src_encodings,
                    table_embedding,
                    schema_embedding,
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
