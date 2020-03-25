import copy
import torch
import torch.nn as nn
from src.transformer.multi_head_attention import MultiheadAttention
from src.ra_transformer.ra_multi_head_attention import RAMultiheadAttention

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class RATransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(RATransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, relation=None, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, relation, tgt_mask=tgt_mask, memory_pamsk=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class RATransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nrelation=0, dim_feedforward=2048, dropout=0.1):
        super(RATransformerDecoderLayer, self).__init__()
        # Multi-head Attention
        self.self_attn = RAMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Relation Embeddings
        self.relation_k_emb = nn.Embedding(nrelation, self.self_attn.head_dim)
        self.relation_v_emb = nn.Embedding(nrelation, self.self_attn.head_dim)

    def forward(self, tgt, memory, tgt_relation=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Relation Embedding
        relation_k = self.relation_k_emb(tgt_relation) if tgt_relation is not None else None
        relation_v = self.relation_v_emb(tgt_relation) if tgt_relation is not None else None

        # Zero out pad relations
        tmp = tgt_relation.unsqueeze(-1).expand(-1, -1, -1, self.relation_k_emb.embedding_dim)
        zeros = torch.zeros_like(relation_k)
        relation_k2 = torch.where(tmp == zeros, zeros, relation_k)
        relation_v2 = torch.where(tmp == zeros, zeros, relation_v)

        # Self Attention
        tgt2 = self.self_attn(tgt, tgt, tgt, relation_k=relation_k2, relation_v=relation_v2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Multihead Attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

