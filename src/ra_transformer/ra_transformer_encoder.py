import copy
import torch
import torch.nn as nn
from src.ra_transformer.ra_multi_head_attention import RAMultiheadAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class RATransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(RATransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, src, relation, mask=None, src_key_padding_mask=None, return_details=False
    ):
        output = src
        qk_weights_list = []
        qk_relation_weights_list = []

        for layer in self.layers:
            out = layer(
                output,
                relation,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                return_details=return_details,
            )
            if return_details:
                output, qk_weights, qk_relation_weights = out
                qk_weights_list += [qk_weights]
                qk_relation_weights_list += [qk_relation_weights]
            else:
                output = out

        if self.norm is not None:
            output = self.norm(output)

        if return_details:
            return output, qk_weights_list, qk_relation_weights_list
        else:
            return output


class RATransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nrelation=0, dim_feedforward=2048, dropout=0.1):
        super(RATransformerEncoderLayer, self).__init__()
        # Multi-head Attention
        self.self_attn = RAMultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Relation Embeddings
        self.relation_k_emb = nn.Embedding(nrelation, self.self_attn.head_dim)
        self.relation_v_emb = nn.Embedding(nrelation, self.self_attn.head_dim)

    def forward(
        self,
        src,
        relation=None,
        src_mask=None,
        src_key_padding_mask=None,
        return_details=False,
    ):
        # Relation Embedding
        relation_k = self.relation_k_emb(relation) if relation is not None else None
        relation_v = self.relation_v_emb(relation) if relation is not None else None

        # Zero out pad relations
        tmp = relation.unsqueeze(-1).expand(
            -1, -1, -1, self.relation_k_emb.embedding_dim
        )
        zeros = torch.zeros_like(relation_k)
        relation_k2 = torch.where(tmp == zeros, zeros, relation_k)
        relation_v2 = torch.where(tmp == zeros, zeros, relation_v)

        # self Multi-head Attention & Residual & Norm
        output = self.self_attn(
            src,
            src,
            src,
            relation_k=relation_k2,
            relation_v=relation_v2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_details=return_details,
        )
        if return_details:
            src2, qk_weights, qk_relation_weights = output
        else:
            src2 = output
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # FeedForward & Residual & Norm
        src2 = self.feed_forward(src)
        src = src + src2
        src = self.norm2(src)

        if return_details:
            return src, qk_weights, qk_relation_weights
        else:
            return src
