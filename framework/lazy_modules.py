from typing import Any, List
from abc import ABC, abstractmethod
import torch
from torch import nn
import math
from src.transformer_decoder import (
    TransformerDecoderLayer,
    TransformerDecoder,
)
from framework.sequential_monad import TensorPromise
from framework.utils import assert_dim, stack_sequential_tensor_with_mask


class LazyModule(ABC):
    def __init__(self):
        self.later_buffer = []
        self.promises = []
        self.done_buffer = []

    def wait_if_not_done(self):
        if self.later_buffer:
            self.compute()
            for idx in range(len(self.later_buffer)):
                self.promises[idx].result = self.done_buffer[idx]
            self.later_buffer = []
            self.promises = []
            self.done_buffer = []

    def forward_later(self, *inputs) -> TensorPromise:
        self.assert_input(*inputs)
        self.later_buffer.append(inputs)
        appended_index = len(self.later_buffer) - 1
        promise = TensorPromise(self, appended_index)
        self.promises.append(promise)
        return promise

    def fetch(self, index: int) -> Any:
        return self.done_buffer[index]

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
        stacked_tensors = torch.stack(tensor_list)
        computed_tensors = self.module(stacked_tensors)
        self.done_buffer = [
            computed_tensors[idx] for idx in range(len(computed_tensors))
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
            out_batch_first[idx] for idx in range(len(self.later_buffer))
        ]
