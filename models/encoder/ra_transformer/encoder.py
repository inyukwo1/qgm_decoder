import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from src.ra_transformer.ra_transformer_encoder import (
    RATransformerEncoder,
    RATransformerEncoderLayer,
)
from src.relation import N_RELATIONS, RELATION_LIST


class RA_Transformer_Encoder(nn.Module):
    def __init__(self, cfg):
        super(RA_Transformer_Encoder, self).__init__()
        self.cfg = cfg
        nhead = cfg.nhead
        layer_num = cfg.layer_num
        hidden_size = cfg.hidden_size
        dim = hidden_size
        self.is_bert = cfg.is_bert
        self.use_nl_lstm = cfg.use_nl_lstm_encoder
        self.use_col_lstm = cfg.use_col_lstm_encoder
        self.use_tab_lstm = cfg.use_tab_lstm_encoder

        self.use_nl_rat = cfg.use_nl_rat_encoder
        self.use_schema_rat = cfg.use_schema_rat_encoder
        self.use_guided_attention = cfg.use_guided_attention

        # layer num
        nl_rat_num_layer = cfg.nl_rat_num_layer
        schema_rat_num_layer = cfg.schema_rat_num_layer

        if self.use_nl_lstm:
            self.sen_lstm = nn.LSTM(
                dim, hidden_size // 2, bidirectional=True, batch_first=True
            )
            self.tab_lstm = nn.LSTM(
                dim, hidden_size // 2, bidirectional=True, batch_first=True
            )
        if self.use_nl_rat:
            # Separated
            nl_encoder_layer = RATransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                nrelation=N_RELATIONS,
                change_relation_contribution=cfg.change_relation_contribution,
                explicit_relation_feature=cfg.explicit_relation_feature,
            )
            self.nl_ra_transformer_encoder = RATransformerEncoder(
                nl_encoder_layer, num_layers=nl_rat_num_layer,
            )
        if self.use_schema_rat:
            schema_encoder_layer = RATransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                nrelation=N_RELATIONS,
                change_relation_contribution=cfg.change_relation_contribution,
                explicit_relation_feature=cfg.explicit_relation_feature,
            )
            self.schema_ra_transformer_encoder = RATransformerEncoder(
                schema_encoder_layer, num_layers=schema_rat_num_layer
            )

        # Combined
        encoder_layer = RATransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            nrelation=N_RELATIONS,
            change_relation_contribution=cfg.change_relation_contribution,
            explicit_relation_feature=cfg.explicit_relation_feature,
        )
        self.ra_transformer_encoder = RATransformerEncoder(
            encoder_layer, num_layers=layer_num
        )

    def forward(
        self,
        sen,
        col,
        tab,
        sen_len,
        col_len,
        tab_len,
        sen_mask,
        col_mask,
        tab_mask,
        relation,
        return_details=False,
    ):
        # LSTM
        if self.use_lstm:
            sen = self.encode_with_lstm(sen, sen_len, self.sen_lstm)
            tab = self.encode_with_lstm(tab, tab_len, self.tab_lstm)

        # Get len
        sen_max_len = sen.shape[1]
        col_max_len = col.shape[1]
        tab_max_len = tab.shape[1]

        # Change shape
        sen = sen.transpose(0, 1)
        col = col.transpose(0, 1)
        tab = tab.transpose(0, 1)

        if self.use_nl_rat:
            # encode NL
            nl_relation = relation[:, :sen_max_len, :sen_max_len]
            nl_src = sen
            nl_mask = sen_mask.bool()
            sen = self.nl_ra_transformer_encoder(
                nl_src, nl_src, nl_relation, src_key_padding_mask=nl_mask
            )

        if self.use_schema_rat:
            # encode schema
            schema_relation = relation[:, sen_max_len:, sen_max_len:]
            schema_src = torch.cat([col, tab], dim=0)
            schema_mask = torch.cat([col_mask, tab_mask], dim=1).bool()
            schema_out = self.schema_ra_transformer_encoder(
                schema_src,
                schema_src,
                schema_relation,
                src_key_padding_mask=schema_mask,
            )
            col = schema_out[:col_max_len, :, :]
            tab = schema_out[col_max_len:, :, :]

        # Combine
        src = torch.cat([sen, col, tab], dim=0)
        src_key_padding_mask = torch.cat([sen_mask, col_mask, tab_mask], dim=1).bool()

        output = self.ra_transformer_encoder(
            src,
            relation,
            src_key_padding_mask=src_key_padding_mask,
            return_details=return_details,
        )
        if return_details:
            encoded_src, qk_weights_list, qk_relation_weights_list = output
        else:
            src = torch.cat([sen, col, tab], dim=0)
            src_key_padding_mask = torch.cat(
                [sen_mask, col_mask, tab_mask], dim=1
            ).bool()

            output = self.ra_transformer_encoder(
                src,
                src,
                relation,
                src_key_padding_mask=src_key_padding_mask,
                return_details=return_details,
            )
            if return_details:
                encoded_src, qk_weights_list, qk_relation_weights_list = output
            else:
                encoded_src = output
            encoded_src = encoded_src.transpose(0, 1)

            # Get split points
            sen_idx = sen_max_len
            col_idx = sen_idx + col_max_len
            tab_idx = col_idx + tab_max_len
            assert tab_idx == src.shape[0], "Size doesn't match {} {}".format(
                tab_idx, src.shape
            )

            # Split
            encoded_sen = encoded_src[:, :sen_idx, :]
            encoded_col = encoded_src[:, sen_idx:col_idx, :]
            encoded_tab = encoded_src[:, col_idx:tab_idx, :]

        if return_details:
            return (
                encoded_sen,
                encoded_col,
                encoded_tab,
                qk_weights_list,
                qk_relation_weights_list,
            )
        else:
            return encoded_sen, encoded_col, encoded_tab

    def encode_with_lstm(self, src_emb, src_len, lstm):
        # Sort
        sorted_len, sorted_data_indices = torch.tensor(src_len).sort(0, descending=True)
        sorted_src_emb = src_emb[sorted_data_indices]

        # Encode
        packed_src_emb = pack_padded_sequence(
            sorted_src_emb, sorted_len, batch_first=True
        )
        packed_src_encodings, (last_state, last_cell) = lstm(packed_src_emb)
        packed_src_encodings, _ = pad_packed_sequence(
            packed_src_encodings, batch_first=True
        )

        # Back to original order
        new_idx = list(range(len(src_len)))
        new_idx.sort(key=lambda e: sorted_data_indices[e])
        encoded_src = packed_src_encodings[new_idx]

        return encoded_src
