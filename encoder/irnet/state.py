import torch
from typing import List, NewType, Dict
from framework.sequential_monad import State
import numpy as np


class LSTMEncoderState(State):
    def __init__(
        self,
        sentence: List[List[str]],
        col_names: List[List[str]],
        col_hot_type: List[np.ndarray],
        tab_names: List[List[str]],
        tab_hot_type: List[np.ndarray],
    ):
        self.sentence: List[List[str]] = sentence
        self.col_names: List[List[str]] = col_names
        self.col_hot_type: List[np.ndarray] = col_hot_type
        self.tab_names: List[List[str]] = tab_names
        self.tab_hot_type: List[np.ndarray] = tab_hot_type
        self.result: Dict[str, torch.Tensor] = None

    def save_result(self, result: Dict[str, torch.Tensor]):
        self.result = result
