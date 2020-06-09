import torch
import torch.nn as nn


class RAMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        change_relation_contribution=False,
        explicit_relation_feature=False,
    ):
        super(RAMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.parameter.Parameter(
            torch.empty(3 * embed_dim, embed_dim)
        )

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.parameter.Parameter(
                torch.Tensor(embed_dim, embed_dim)
            )
            self.k_proj_weight = nn.parameter.Parameter(
                torch.Tensor(embed_dim, self.kdim)
            )
            self.v_proj_weight = nn.parameter.Parameter(
                torch.Tensor(embed_dim, self.vdim)
            )

        if bias:
            self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.parameter.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.parameter.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.explicit_relation_feature = explicit_relation_feature
        self.change_relation_contribution = change_relation_contribution
        if change_relation_contribution:
            self.relation_parameter = nn.parameter.Parameter(
                torch.Tensor(num_heads, self.head_dim, 1)
            )

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key,
        value,
        relation_k=None,
        relation_v=None,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        return_details=False,
    ):
        # relation_k : [batch_size, k_len, k_len, dim_of_head]
        # relation_v : [batch_size, k_len, k_len, dim_of_head]
        assert self._qkv_same_embed_dim, "query, key, value should all be same size"
        embed_dim_to_check = self.embed_dim
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        bias_k = self.bias_k
        bias_v = self.bias_v
        add_zero_attn = self.add_zero_attn
        dropout_p = self.dropout
        out_proj_weight = self.out_proj.weight
        out_proj_bias = self.out_proj.bias
        training = self.training

        # Modify code from nn.functional.multi_head_attention_forward
        kv_same = torch.equal(key, value)
        qkv_same = torch.equal(query, key) and kv_same

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim_to_check == embed_dim, "self:{} {}".format(
            embed_dim_to_check, embed_dim
        )
        assert key.size() == value.size()

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if qkv_same:
            q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(
                3, dim=-1
            )
        elif kv_same:
            _start, _end = 0, embed_dim
            _w = in_proj_weight[_start:_end, :]
            _b = in_proj_bias[_start:_end] if in_proj_bias is not None else None
            q = nn.functional.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = v = None
            else:
                _start, _end = embed_dim, None
                _w = in_proj_weight[_start:, :]
                _b = in_proj_bias[_start:] if in_proj_bias is not None else None
                k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            raise NotImplemented("Not implemented yet")  # Expected qkv same or kv same
            pass

        q = q * scaling

        assert bias_k is None and bias_v is None

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            raise RuntimeError(
                "add_zero_attn not implemented. check torch.nn.functional.multi_head_attention_forward"
            )
        # q k
        attn_output_weights_k = torch.bmm(q, k.transpose(1, 2))

        if self.change_relation_contribution:
            r_k = relation_k.unsqueeze(1).expand(-1, num_heads, -1, -1, -1)
            r_k = r_k.reshape(bsz * num_heads, tgt_len * tgt_len, head_dim)
            relation_parameter = self.relation_parameter.expand(bsz, -1, -1, -1)
            relation_parameter = relation_parameter.reshape(bsz * num_heads, -1, 1)
            attn_output_weights_relation_k = torch.bmm(r_k, relation_parameter).view(
                bsz * num_heads, tgt_len, tgt_len
            )

        elif qkv_same and relation_k is not None:
            # q r_k
            q = q.reshape(bsz * num_heads * tgt_len, 1, head_dim)
            r_k = relation_k.unsqueeze(1).expand(-1, num_heads, -1, -1, -1)
            r_k = r_k.transpose(-1, -2).reshape(
                bsz * num_heads * tgt_len, head_dim, tgt_len
            )
            attn_output_weights_relation_k = torch.bmm(q, r_k).view(
                bsz * num_heads, tgt_len, tgt_len
            )

        # Combine
        attn_output_weights = (
            attn_output_weights_k + attn_output_weights_relation_k
            if qkv_same and relation_k is not None
            else attn_output_weights_k
        )

        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            raise RuntimeError(
                "attn mask not implemented. check torch.nn.functional.multi_head_attention_forward"
            )

        if return_details:
            # attn_output_weights_k
            attn_output_weights_k = attn_output_weights_k.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights_k = attn_output_weights_k.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_output_weights_k = attn_output_weights_k.view(
                bsz * num_heads, tgt_len, src_len
            )
            attn_output_weights_k = nn.functional.softmax(attn_output_weights_k, dim=-1)

            # attn_output_weights_relation_k
            attn_output_weights_relation_k = attn_output_weights_relation_k.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights_relation_k = attn_output_weights_relation_k.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_output_weights_relation_k = attn_output_weights_relation_k.view(
                bsz * num_heads, tgt_len, src_len
            )
            attn_output_weights_relation_k = nn.functional.softmax(
                attn_output_weights_relation_k, dim=-1
            )

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        # qk v
        attn_output_v = torch.bmm(attn_output_weights, v)

        # qk v_k
        if qkv_same and relation_v is not None and self.explicit_relation_feature:
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads * tgt_len, 1, tgt_len
            )
            r_v = relation_v.unsqueeze(1).expand(-1, num_heads, -1, -1, -1)
            r_v = r_v.reshape(bsz * num_heads * tgt_len, tgt_len, head_dim)
            attn_output_r_v = torch.bmm(attn_output_weights, r_v)
            attn_output_r_v = attn_output_r_v.view(bsz * num_heads, tgt_len, head_dim)

        # Combine
        attn_output = (
            attn_output_v + attn_output_r_v
            if qkv_same and relation_v is not None and self.explicit_relation_feature
            else attn_output_v
        )

        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).reshape(tgt_len, bsz, embed_dim)
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        if return_details:
            return attn_output, attn_output_weights_k, attn_output_weights_relation_k
        else:
            return attn_output
