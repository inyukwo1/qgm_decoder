import math
import torch
import torch.nn as nn
from src.transformer.transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


class Transformer_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Transformer_Encoder, self).__init__()
        self.cfg = cfg
        nhead = cfg.nhead
        layer_num = 1
        hidden_size = cfg.hidden_size
        dim = hidden_size

        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=layer_num
        )
        self._init_positional_embedding(dim)

    def _init_positional_embedding(self, d_model, dropout=0.1, max_len=100):
        self.pos_dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def pos_encode(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.pos_dropout(x)

    def forward(self, sen, col, tab, sen_mask, col_mask, tab_mask):
        # Get len
        sen_len = sen.shape[1]
        col_len = col.shape[1]
        tab_len = tab.shape[1]

        # Change shape
        sen = sen.transpose(0, 1)
        col = col.transpose(0, 1)
        tab = tab.transpose(0, 1)

        # Encode positional embeddings
        sen = self.pos_encode(sen)

        # Combine
        src = torch.cat([sen, col, tab], dim=0)
        src_key_padding_mask = torch.cat([sen_mask, col_mask, tab_mask], dim=1).bool()

        encoded_src = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        ).transpose(0, 1)

        # Get split points
        sen_idx = sen_len
        col_idx = sen_idx + col_len
        tab_idx = col_idx + tab_len
        assert tab_idx == src.shape[0], "Size doesn't match {} {}".format(
            tab_idx, src.shape
        )

        # Split
        encoded_sen = encoded_src[:, :sen_idx, :]
        encoded_col = encoded_src[:, sen_idx:col_idx, :]
        encoded_tab = encoded_src[:, col_idx:tab_idx, :]

        return encoded_sen, encoded_col, encoded_tab
