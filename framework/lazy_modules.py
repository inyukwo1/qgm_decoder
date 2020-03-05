from typing import Any, List
from abc import ABC, abstractmethod
import torch
from torch import nn
import numpy as np
import math
from src.transformer_decoder import (
    TransformerDecoderLayer,
    TransformerDecoder,
)
from framework.sequential_monad import TensorPromise
from framework.utils import assert_dim, stack_sequential_tensor_with_mask

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def assert_input(self, *inputs):
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
        elif len(tensor.size()) == 2:
            assert_dim([None, self.in_dim], tensor)

    def compute(self):
        tensor_list = [inputs[0] for inputs in self.later_buffer]
        one_tensor = tensor_list[0]
        b_size = len(tensor_list)
        if len(one_tensor.size()) == 1:
            stacked_tensors = torch.stack(tensor_list)
            computed_tensors = self.module(stacked_tensors)
            self.done_buffer = [computed_tensors[idx] for idx in range(b_size)]
        elif len(one_tensor.size()) == 2:
            stacked_tensors, mask = stack_sequential_tensor_with_mask(tensor_list)
            len_tensor_list = [len(tensor) for tensor in tensor_list]
            computed_tensors = self.module(stacked_tensors)
            self.done_buffer = [
                computed_tensors[idx, : len_tensor_list[idx]] for idx in range(b_size)
            ]


class LazyTransformerDecoder(nn.Module, LazyModule):
    def __init__(self, in_dim, nhead, layer_num):
        super(LazyTransformerDecoder, self).__init__()
        LazyModule.__init__(self)
        decoder_layer = TransformerDecoderLayer(d_model=in_dim, nhead=nhead)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=layer_num
        )
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
        out_batch_first = self.transformer_decoder(
            pos_encoded_tgt_batch_second,
            stacked_mem_batch_second,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=mem_mask,
        ).transpose(0, 1)
        self.done_buffer = [
            out_batch_first[idx, : len(tgt_list[idx])]
            for idx in range(len(self.later_buffer))
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
        assert impossible_indices is None or isinstance(impossible_indices, List)
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


class LazyBiLSTM(nn.Module, LazyModule):
    def __init__(self, in_dim, hidden_dim):
        super(LazyBiLSTM, self).__init__()
        LazyModule.__init__(self)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            self.in_dim, hidden_dim // 2, bidirectional=True, batch_first=True,
        )

    def assert_input(self, *inputs):
        [sequence] = inputs
        assert_dim([None, self.in_dim], sequence)

    def compute(self):
        sequence_list: List[torch.Tensor] = [inputs[0] for inputs in self.later_buffer]
        sequence_len_list = [len(sequence) for sequence in sequence_list]
        sequence_len_tensor = torch.LongTensor(sequence_len_list)
        sorted_seq_lengths, perm_idx = sequence_len_tensor.sort(0, descending=True)
        stacked_sequence, _ = stack_sequential_tensor_with_mask(sequence_list)
        sorted_stacked_sequence = stacked_sequence[perm_idx]
        sorted_packed_input = pack_padded_sequence(
            sorted_stacked_sequence, sorted_seq_lengths.cpu().numpy(), batch_first=True
        )
        sorted_packed_output, (sorted_last_state, sorted_last_cell) = self.lstm(
            sorted_packed_input
        )
        sorted_src_encodings, _ = pad_packed_sequence(
            sorted_packed_output, batch_first=True
        )
        sorted_last_state = torch.cat([sorted_last_state[0], sorted_last_state[1]], -1)
        sorted_last_cell = torch.cat([sorted_last_cell[0], sorted_last_cell[1]], -1)
        self.done_buffer = [
            (
                sorted_src_encodings[idx, : sorted_seq_lengths[idx]],
                (sorted_last_state[idx], sorted_last_cell[idx]),
            )
            for idx in perm_idx
        ]