# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import numpy as np
from typing import Dict, List

from encoder.irnet.state import LSTMEncoderState
from src import utils
from framework.utils import assert_dim
from framework.lazy_modules import LazyLinear, LazyBiLSTM
from framework.sequential_monad import TensorPromiseOrTensor, SequentialMonad, LogicUnit


class IRNetLSTMEncoder(nn.Module):
    def __init__(self, cfg):
        super(IRNetLSTMEncoder, self).__init__()
        self.cfg = cfg
        self.is_cuda = cfg.cuda != -1
        self.decoder_name = cfg.model_name
        self.embed_size = 300
        self._import_word_emb()

        hidden_size = cfg.hidden_size

        self.col_type_encoder = LazyLinear(9, hidden_size)
        self.tab_type_encoder = LazyLinear(5, hidden_size)

        self.encoder_lstm = LazyBiLSTM(self.embed_size, hidden_size)

        self.dropout = nn.Dropout(cfg.dropout)

    def _import_word_emb(self):
        self.word_emb = utils.load_word_emb(self.cfg.glove_embed_path)

    def forward(self, examples):
        states = [
            LSTMEncoderState(
                example.src_sent,
                example.tab_cols,
                example.col_hot_type,
                example.table_names,
                example.tab_hot_type,
            )
            for example in examples
        ]

        def embed_sentence_col_tab(
            state: LSTMEncoderState, _
        ) -> Dict[str, torch.Tensor]:
            def emb_list_list_str(list_list_str: List[List[str]]):
                list_list_np: List[List[np.ndarray]] = [
                    [
                        self.word_emb.get(word, self.word_emb["unk"])
                        for word in word_list
                    ]
                    for word_list in list_list_str
                ]
                for list_np in list_list_np:
                    assert len(list_np) >= 1
                embedding_list: List[np.ndarray] = [
                    sum(list_np) / len(list_np) for list_np in list_list_np
                ]
                embedding_np: np.ndarray = np.stack(embedding_list, axis=0)
                embedding_tensor: torch.Tensor = torch.from_numpy(
                    embedding_np
                ).float().cuda()
                return embedding_tensor

            return {
                "src_embedding": emb_list_list_str(state.sentence),
                "col_embedding": emb_list_list_str(state.col_names),
                "tab_embedding": emb_list_list_str(state.tab_names),
            }

        def encode_sentence(
            state: LSTMEncoderState, prev_tensor_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, TensorPromiseOrTensor]:
            src_embedding = prev_tensor_dict["src_embedding"]
            lstm_promise = self.encoder_lstm.forward_later(src_embedding)
            return {
                "lstm_promise": lstm_promise,
                "src_embedding": prev_tensor_dict["src_embedding"],
                "col_embedding": prev_tensor_dict["col_embedding"],
                "tab_embedding": prev_tensor_dict["tab_embedding"],
            }

        def encode_col_tab(
            state: LSTMEncoderState, prev_tensor_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, TensorPromiseOrTensor]:
            def embed_context(src_encoding, src_embedding, compare_embedding):
                len_src = len(src_embedding)
                len_compare = len(compare_embedding)
                src_embedding_expanded = src_embedding.unsqueeze(0).repeat(
                    len_compare, 1, 1
                )
                compare_embedding_expanded = compare_embedding.unsqueeze(1).repeat(
                    1, len_src, 1
                )
                similarity = F.cosine_similarity(
                    compare_embedding_expanded, src_embedding_expanded, dim=-1
                )
                assert_dim([len_compare, len_src], similarity)
                context_embedding = (
                    src_encoding.unsqueeze(0) * similarity.unsqueeze(2)
                ).sum(1)
                assert_dim([len_compare, self.embed_size], context_embedding)
                return context_embedding

            src_encoding, (last_state, last_cell) = prev_tensor_dict[
                "lstm_promise"
            ].result

            src_encoding = self.dropout(src_encoding)
            src_embedding = prev_tensor_dict["src_embedding"]
            col_embedding = prev_tensor_dict["col_embedding"]
            col_encoding = col_embedding + embed_context(
                src_encoding, src_embedding, col_embedding
            )
            tab_embedidng = prev_tensor_dict["tab_embedding"]
            tab_encoding = tab_embedidng + embed_context(
                src_encoding, src_embedding, tab_embedidng
            )

            return {
                "src_encoding": src_encoding,
                "col_encoding": col_encoding,
                "tab_encoding": tab_encoding,
                "last_cell": last_cell,
            }

        def encode_col_tab_type(
            state: LSTMEncoderState, prev_tensor_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, TensorPromiseOrTensor]:
            col_hot_type: np.ndarray = state.col_hot_type
            col_hot_type_tensor: torch.Tensor = torch.from_numpy(
                col_hot_type
            ).float().cuda()
            col_type_embedding = self.col_type_encoder.forward_later(
                col_hot_type_tensor
            )
            tab_hot_type: np.ndarray = state.tab_hot_type
            tab_hot_type_tensor: torch.Tensor = torch.from_numpy(
                tab_hot_type
            ).float().cuda()
            tab_type_embedding = self.tab_type_encoder.forward_later(
                tab_hot_type_tensor
            )

            return {
                "src_encoding": prev_tensor_dict["src_encoding"],
                "col_encoding": prev_tensor_dict["col_encoding"],
                "col_type_embedding": col_type_embedding,
                "tab_encoding": prev_tensor_dict["tab_encoding"],
                "tab_type_embedding": tab_type_embedding,
                "last_cell": prev_tensor_dict["last_cell"],
            }

        def save_result_in_state(
            state: LSTMEncoderState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]
        ) -> None:
            state.save_result(
                {
                    "src_encoding": prev_tensor_dict["src_encoding"],
                    "col_encoding": prev_tensor_dict["col_encoding"]
                    + prev_tensor_dict["col_type_embedding"].result,
                    "tab_encoding": prev_tensor_dict["tab_encoding"]
                    + prev_tensor_dict["tab_type_embedding"].result,
                    "last_cell": prev_tensor_dict["last_cell"],
                }
            )

        states = SequentialMonad(states)(
            LogicUnit.If(lambda x: True)
            .Then(embed_sentence_col_tab)
            .Then(encode_sentence)
            .Then(encode_col_tab)
            .Then(encode_col_tab_type)
            .Then(save_result_in_state)
        ).states
        return [state.result for state in states]
