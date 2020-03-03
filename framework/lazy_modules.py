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


class LazyModule(ABC):
    def __init__(self):
        self.computed = False
        self.later_buffer = []
        self.done_buffer = []

    def wait_if_not_done(self):
        if not self.computed:
            self.compute()
            self.later_buffer = []
            self.computed = True

    def forward_later(self, any: Any) -> TensorPromise:
        self.assert_input(any)
        self.later_buffer.append(any)
        appended_index = len(self.later_buffer) - 1
        return TensorPromise(self, appended_index)

    def fetch(self, index: int) -> Any:
        return self.done_buffer[index]

    def reset(self):
        self.computed = False
        self.later_buffer = []
        self.done_buffer = []

    def assert_input(self, any: Any):
        pass

    @abstractmethod
    def compute(self) -> None:
        pass


def assert_dim(dim, tensor: torch.Tensor):
    tensor_dim = list(tensor.size())
    for expected, real in zip(dim, tensor_dim):
        if expected is not None:
            assert expected == real, "expected: {} real: {}".format(dim, tensor_dim)


class LazyLinear(nn.Module, LazyModule):
    def __init__(self, in_dim, out_dim):
        super(LazyLinear, self).__init__()
        LazyModule.__init__(self)
        self.module = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def assert_input(self, any: Any):
        assert isinstance(any, torch.Tensor)
        assert_dim([self.in_dim], any)

    def compute(self):
        stacked_tensors = torch.stack(self.later_buffer)
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
        self._init_positional_embedding(in_dim)

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
