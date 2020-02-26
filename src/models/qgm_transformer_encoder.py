import math
import torch
import torch.nn as nn
from src.transformer.transformer_encoder import TransformerEncoder, TransformerEncoderLayer


class QGM_Transformer_Encoder(nn.Module):
    def __init__(self, cfg):
        super(QGM_Transformer_Encoder, self).__init__()
        self.cfg = cfg

        encoder_layer = TransformerEncoderLayer(d_model=300, nhead=6)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=3)

        self._init_positional_embedding(300)


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


    def _pos_encode(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.pos_dropout(x)


    def forward(self, src, col, tab, src_mask, col_mask, tab_mask):

        # Add pos embeddings and type embeddings
        src = self._pos_encode(src)

        # Concatenate inputs
        input_emb = torch.cat([src, col, tab], dim=1).transpose(0, 1)
        input_mask = torch.cat([src_mask, col_mask, tab_mask], dim=1).bool()

        # Encode with transformer
        encoded_input = self.transformer_encoder(input_emb, src_key_padding_mask=input_mask).transpose(0, 1)

        # Split indices
        src_idx = src.shape[1]
        col_idx = src_idx+col.shape[1]
        tab_idx = col_idx+tab.shape[1]
        assert tab_idx == input_emb.shape[0]

        # Split output
        encoded_src = encoded_input[:, :src_idx, :]
        encoded_col = encoded_input[:, src_idx:col_idx, :]
        encoded_tab = encoded_input[:, col_idx:, :]

        return encoded_src, encoded_col, encoded_tab
