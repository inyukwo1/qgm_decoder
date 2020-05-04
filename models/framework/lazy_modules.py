from typing import Any, List
from abc import ABC, abstractmethod
import torch
from torch import nn
import math
from src.transformer.transformer_decoder import (
    TransformerDecoderLayer,
    TransformerDecoder,
)
from src.ra_transformer.ra_transformer_decoder import (
    RATransformerDecoderLayer,
    RATransformerDecoder,
)
from models.framework.sequential_monad import TensorPromise
from models.framework.utils import assert_dim, stack_sequential_tensor_with_mask
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LazyModule(ABC):
    waiting_lazy_modules: List["LazyModule"] = []

    def __init__(self):
        self.later_buffer = []
        self.promises = []
        self.done_buffer = []

    @classmethod
    def wait_all(cls):
        for lazy_module in LazyModule.waiting_lazy_modules:
            lazy_module.compute()
            for idx in range(len(lazy_module.later_buffer)):
                lazy_module.promises[idx].result = lazy_module.done_buffer[idx]
            lazy_module.later_buffer = []
            lazy_module.promises = []
            lazy_module.done_buffer = []
        LazyModule.waiting_lazy_modules = []

    def forward_later(self, *inputs) -> TensorPromise:
        if self not in LazyModule.waiting_lazy_modules:
            LazyModule.waiting_lazy_modules.append(self)
        self.assert_input(*inputs)
        self.later_buffer.append(inputs)
        appended_index = len(self.later_buffer) - 1
        promise = TensorPromise(self, appended_index)
        self.promises.append(promise)
        return promise

    def assert_input(self, any: Any):
        pass

    @abstractmethod
    def compute(self) -> None:
        pass


class LazyLinear(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyLinear, self).__init__()
        LazyModule.__init__(self)
        self.module = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def assert_input(self, *inputs):
        [tensor] = inputs
        assert isinstance(tensor, torch.Tensor)
        if len(tensor.size()) == 1:
            assert_dim([self.in_dim], tensor)
        elif len(tensor.size()) == 1:
            assert_dim([None, self.in_dim], tensor)

    def compute(self):
        tensor_list = [inputs[0] for inputs in self.later_buffer]
        tensor_length = [len(item) for item in tensor_list]

        stacked_tensors = torch.zeros(
            len(tensor_list), max(tensor_length), tensor_list[0].shape[-1]
        ).cuda()
        for idx, _tensor in enumerate(tensor_list):
            stacked_tensors[idx][: len(_tensor)] = _tensor

        computed_tensors = self.module(stacked_tensors)

        # Split
        self.done_buffer = [
            computed_tensor[:length]
            for length, computed_tensor in zip(tensor_length, computed_tensors)
        ]


class LazyLSTMCell(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyLSTMCell, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module = nn.LSTMCell(in_dim, out_dim)

    def assert_input(self, *inputs):
        pass

    def compute(self):
        input_list = []
        hidden_list = []
        cell_list = []
        for item in self.later_buffer:
            input_list.append(item[0])
            hidden_list.append(item[1][0])
            cell_list.append(item[1][1])

        # Stacked
        stacked_input = torch.stack(input_list)
        stacked_hidden = torch.stack(hidden_list)
        stacked_cell = torch.stack(cell_list)

        next_hid, next_cell = self.module(stacked_input, (stacked_hidden, stacked_cell))

        self.done_buffer = [(hid, cell) for hid, cell in zip(next_hid, next_cell)]


class LazyLSTM(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim, batch_first=True, bidirection=True):
        super(LazyLSTM, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module = nn.LSTM(
            in_dim, out_dim, batch_first=batch_first, bidirectional=bidirection
        )

    def assert_intput(self, *inputs):
        pass

    def compute(self):
        input_list = [item[0] for item in self.later_buffer]

        # Stack
        stacked_input, input_mask = stack_sequential_tensor_with_mask(input_list)

        # Sort
        input_len = [len(item) for item in input_list]
        sorted_len, sorted_indices = torch.tensor(input_len).sort(0, descending=True)
        sorted_data = stacked_input

        # Pass
        packed_data = pack_padded_sequence(sorted_data, sorted_len, batch_first=True)
        packed_output, (hn, cn) = self.module(packed_data)
        packed_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        last_state = torch.cat([hn[0], hn[1]], dim=-1)
        last_cell = torch.cat([cn[0], cn[1]], dim=-1)

        # Unsort
        new_indices = list(range(len(input_len)))
        new_indices.sort(key=lambda k: sorted_indices[k])
        encoded_data = packed_output[new_indices]
        last_state = last_state[new_indices]
        last_cell = last_cell[new_indices]

        # spread inputs
        self.done_buffer = [
            (item1[:length], (item2, item3))
            for length, item1, item2, item3 in zip(
                input_len, encoded_data, last_state, last_cell
            )
        ]


class LazyTransformerDecoder(nn.Module, LazyModule):
    def __init__(self, in_dim, nhead, layer_num):
        super(LazyTransformerDecoder, self).__init__()
        LazyModule.__init__(self)
        decoder_layer = TransformerDecoderLayer(d_model=in_dim, nhead=nhead)
        self.module = TransformerDecoder(decoder_layer, num_layers=layer_num)
        self.in_dim = in_dim
        self._init_positional_embedding()

    def _init_positional_embedding(self, dropout=0.1, max_len=100):
        d_model = self.in_dim

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

    def assert_input(self, *inputs):
        tgt, mem = inputs
        assert_dim([None, self.in_dim], tgt)
        assert_dim([None, self.in_dim], mem)

    def compute(self):
        tgt_list: List[torch.Tensor] = []
        mem_list: List[torch.Tensor] = []
        for tgt, mem in self.later_buffer:
            tgt_list.append(tgt)
            mem_list.append(mem)
        stacked_tgt, tgt_mask = stack_sequential_tensor_with_mask(tgt_list)
        stacked_mem, mem_mask = stack_sequential_tensor_with_mask(mem_list)
        stacked_tgt_batch_second = stacked_tgt.transpose(0, 1)
        stacked_mem_batch_second = stacked_mem.transpose(0, 1)
        pos_encoded_tgt_batch_second = self._pos_encode(stacked_tgt_batch_second)
        out_batch_first = self.module(
            pos_encoded_tgt_batch_second,
            stacked_mem_batch_second,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=mem_mask,
        ).transpose(0, 1)
        self.done_buffer = [
            out_batch_first[idx, : len(tgt_list[idx])]
            for idx in range(len(self.later_buffer))
        ]


class LazyRATransformerDecoder(nn.Module, LazyModule):
    def __init__(self, in_dim, nhead, layer_num, relation_num):
        super(LazyRATransformerDecoder, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        decoder_layer = RATransformerDecoderLayer(
            d_model=in_dim, nhead=nhead, nrelation=relation_num
        )
        self.module = RATransformerDecoder(decoder_layer, num_layers=layer_num)

    def assert_input(self, *inputs):
        pass

    def compute(self):
        # Parse input
        tgt_list: List[torch.Tensor] = []
        mem_list: List[torch.Tensor] = []
        relation_list: List[torch.Tensor] = []
        for tgt, mem, relation in self.later_buffer:
            tgt_list.append(tgt)
            mem_list.append(mem)
            relation_list.append(relation.unsqueeze(-1))

        # Stack
        stacked_tgt, tgt_mask = stack_sequential_tensor_with_mask(tgt_list)
        stacked_mem, mem_mask = stack_sequential_tensor_with_mask(mem_list)
        stacked_relation, relation_mask = stack_sequential_tensor_with_mask(
            relation_list
        )
        stacked_relation = stacked_relation.squeeze(-1)

        # Transpose
        stacked_tgt_batch_second = stacked_tgt.transpose(0, 1)
        stacked_mem_batch_second = stacked_mem.transpose(0, 1)

        # Forward
        out_batch_first = self.module(
            stacked_tgt_batch_second, stacked_mem_batch_second, stacked_relation,
        ).transpose(0, 1)

        # Spread
        tgt_len = [len(item) for item in tgt_list]
        self.done_buffer = [
            item[:length] for item, length in zip(out_batch_first, tgt_len)
        ]


class LazyCalculateSimilarity(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyCalculateSimilarity, self).__init__()
        LazyModule.__init__(self)
        self.affine_layer = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def assert_input(self, *inputs):
        [source, query, impossible_indices] = inputs
        assert isinstance(source, torch.Tensor)
        assert isinstance(query, torch.Tensor)
        assert_dim([self.in_dim], source)
        assert_dim([None, self.out_dim], query)

    def compute(self):
        source_list: List[torch.Tensor] = [inputs[0] for inputs in self.later_buffer]
        query_list: List[torch.Tensor] = [inputs[1] for inputs in self.later_buffer]
        impossible_indices_list: List[List[int]] = [
            inputs[2] for inputs in self.later_buffer
        ]
        stacked_source = torch.stack(source_list, dim=0)
        affined_source = self.affine_layer(stacked_source)
        stacked_query, query_mask = stack_sequential_tensor_with_mask(query_list)
        weight_scores = torch.bmm(
            affined_source.unsqueeze(1), stacked_query.transpose(1, 2)
        ).squeeze(1)
        for idx, impossible_indices in enumerate(impossible_indices_list):
            if impossible_indices is not None:
                query_mask[idx, impossible_indices] = 1
        weight_scores.data.masked_fill_(query_mask.bool(), -float("inf"))
        weight_probs = torch.log_softmax(weight_scores, dim=-1)
        self.done_buffer = [
            weight_probs[idx, : len(query_list[idx])]
            for idx in range(len(weight_probs))
        ]


class LazyAttention(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyAttention, self).__init__()
        LazyModule.__init__(self)
        self.layer = None
        self.in_dim = in_dim
        self.out_dim = out_dim

    def assert_input(self, *inputs):
        pass

    def compute(self):
        pass


class LazyLinearTanhDropout(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyLinearTanhDropout, self).__init__()
        LazyModule.__init__(self)
        self.layer = None
        self.in_dim = in_dim
        self.out_dim = out_dim

    def assert_input(self, *inputs):
        [tensor] = inputs
        assert isinstance(tensor, torch.Tensor)
        if len(tensor.size()) == 1:
            assert_dim([self.in_dim], tensor)
        elif len(tensor.size()) == 1:
            assert_dim([None, self.in_dim], tensor)

    def compute(self):
        tensor_list = [inputs[0] for inputs in self.later_buffer]
        tensor_length = [len(item) for item in tensor_list]

        stacked_tensors = torch.zeros(
            len(tensor_list), max(tensor_length), tensor_list[0].shape[-1]
        ).cuda()
        for idx, _tensor in enumerate(tensor_list):
            stacked_tensors[idx][: len(_tensor)] = _tensor

        computed_tensors = self.module(stacked_tensors)
        computed_tensors = torch.tanh(computed_tensors)
        computed_tensors = torch.dropout(computed_tensors)

        # Split
        self.done_buffer = [
            computed_tensor[:length]
            for length, computed_tensor in zip(tensor_length, computed_tensors)
        ]


class LazyActionProb(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyActionProb, self).__init__()
        LazyModule.__init__(self)
        self.layer = None
        self.in_dim = in_dim
        self.out_dim = out_dim

    def assert_input(self, *inputs):
        pass

    def compute(self):
        pass


class LazyDotProductAttention(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyDotProductAttention, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_layer = torch.nn.Linear(in_dim, out_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()

    def assert_input(self, *inputs):
        pass

    def compute(self):
        q_list: List[torch.Tensor] = []
        k_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        for src, key, vec in self.later_buffer:
            q_list.append(src)
            k_list.append(key)
            v_list.append(vec)
        stacked_q, q_mask = stack_sequential_tensor_with_mask(q_list)
        stacked_k = torch.stack(k_list)
        stacked_v, v_mask = stack_sequential_tensor_with_mask(v_list)

        # Combine
        att_scores = torch.bmm(stacked_q, stacked_k.unsqueeze(2)).squeeze(2)
        att_scores.data.masked_fill_(q_mask.bool(), -float("inf"))

        att_probs = torch.softmax(att_scores, dim=-1)
        att_vec = torch.bmm(att_probs.unsqueeze(1), stacked_v).squeeze(1)

        # LinearTanhDropout
        next_input = torch.cat([stacked_k, att_vec], dim=1)
        next_input = self.out_layer(next_input)
        next_input = self.tanh(next_input)
        att_ctx = self.dropout(next_input)

        self.done_buffer = [item for item in att_ctx]


class LazyLinearLinear(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyLinearLinear, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module_1 = nn.Linear(in_dim, in_dim, bias=False)
        self.module_2 = lambda q, k: torch.nn.functional.linear(q, k, self.bias)
        self.bias = nn.Parameter(torch.FloatTensor(out_dim).zero_())

    def assert_input(self, *inputs):
        pass

    def compute(self):
        input_tensors = [item[0] for item in self.later_buffer]
        input_mask = [item[1] for item in self.later_buffer]
        emb_weights = [item[2] for item in self.later_buffer]

        stacked_input = torch.stack(input_tensors)
        stacked_mask = torch.stack(input_mask)

        pre_scores = self.module_1(stacked_input)
        scores = self.module_2(pre_scores, emb_weights[0])
        scores.data.masked_fill_(stacked_mask.bool(), float("-inf"))
        probs = torch.softmax(scores, dim=-1)

        self.done_buffer = [item for item in probs]


class LazyPointerNet(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyPointerNet, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.pass_linear = nn.Linear(in_dim, out_dim, bias=False)

    def assert_input(self, *inputs):
        pass

    def compute(self):
        query_list = [item[0] for item in self.later_buffer]
        key_list = [item[1] for item in self.later_buffer]

        stacked_query, query_mask = stack_sequential_tensor_with_mask(query_list)
        stacked_key = torch.stack(key_list)

        encoded_query = self.pass_linear(stacked_query)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        weights = torch.bmm(encoded_query, stacked_key.unsqueeze(2)).squeeze(2)

        weights.data.masked_fill_(query_mask.bool(), float("-inf"))
        probs = torch.log_softmax(weights, dim=-1)

        self.done_buffer = [item for item in probs]


class LazyMemoryPointerNet(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyMemoryPointerNet, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pass_gate = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.col_linear = nn.Linear(in_dim, out_dim, bias=False)

    def assert_input(self, *inputs):
        pass

    def compute(self):
        encoded_col_list = [item[0] for item in self.later_buffer]
        att_vec_list = [item[1] for item in self.later_buffer]
        col_memory_mask = [item[2] for item in self.later_buffer]

        stacked_col, col_mask = stack_sequential_tensor_with_mask(encoded_col_list)
        stacked_att = torch.stack(att_vec_list)
        stacked_mem_mask, _ = stack_sequential_tensor_with_mask(col_memory_mask)

        # Create gate
        gate = self.pass_gate(stacked_att)

        encoded_col_stack = self.col_linear(stacked_col)

        weights = torch.bmm(encoded_col_stack, stacked_att.unsqueeze(2)).squeeze(-1)
        one = weights * stacked_mem_mask * gate
        two = weights * (1 - stacked_mem_mask) * (1 - gate)
        total = one + two

        total.data.masked_fill_(col_mask.bool(), float("-inf"))
        probs = torch.log_softmax(total, dim=-1)

        self.done_buffer = [item for item in probs]
