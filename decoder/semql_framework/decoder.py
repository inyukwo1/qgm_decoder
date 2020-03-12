import torch
import torch.nn as nn
from rule.semql.semql import SemQL
from encoder.irnet import nn_utils
from encoder.irnet.pointer_net import PointerNet

import logging
log = logging.getLogger(__name__)

SKETCH_LIST = ["Root1", "Root", "Sel", "N", "Filter", "Sup", "Order"]
DETAIL_LIST = ["A", "C", "T"]

from typing import List, Dict

from decoder.semql_framework.state import (
    LSTMStateGold,
    LSTMStatePred,
    LSTMState,
)
from framework.sequential_monad import (
    SequentialMonad,
    WhileLogic,
    LogicUnit,
    TensorPromiseOrTensor,
    TensorPromise,
)
from framework.lazy_modules import (
    LazyLinear,
    LazyDropout,
    LazyLSTMCellDecoder,
    LazyTransformerDecoder,
    LazyCalculateSimilarity,
)


class SemQLDecoderFramework(nn.Module):
    def __init__(self, cfg):
        super(SemQLDecoderFramework, self).__init__()
        self.cfg = cfg
        self.is_cuda = cfg.cuda != -1
        self.grammar = SemQL.Grammar()
        self.use_column_pointer = cfg.column_pointer

        self.new_tensor = torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor

        hidden_size = cfg.hidden_size
        att_vec_size = cfg.att_vec_size
        type_embed_size = cfg.type_embed_size
        action_embed_size = cfg.action_embed_size
        input_dim = action_embed_size + att_vec_size + type_embed_size

        self.decode_max_time_step = 40
        self.action_embed_size = action_embed_size
        self.type_embed_size = type_embed_size

        # Decoder Layers
        self.encoder_Lstm = nn.LSTM(self.embed_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.decoder_cell_init = LazyLinear(hidden_size, hidden_size)
        self.lf_decoder_lstm = nn.LSTMCell(input_dim, hidden_size)
        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, hidden_size)

        self.att_sketch_linear = LazyLinear(hidden_size, hidden_size, bias=False)
        self.att_lf_linear = LazyLinear(hidden_size, hidden_size, bias=False)
        self.sketch_att_vec_linear = LazyLinear(
            hidden_size + hidden_size, att_vec_size, bias=False
        )
        self.lf_att_vec_linear = LazyLinear(
            hidden_size + hidden_size, att_vec_size, bias=False
        )
        self.prob_att = LazyLinear(att_vec_size, 1)
        self.prob_len = LazyLinear(1, 1)
        self.q_att = LazyLinear(hidden_size, self.embed_size)
        self.column_rnn_input = LazyLinear(
            self.embed_size, action_embed_size, bias=False
        )
        self.table_rnn_input = LazyLinear(self.embed_size, action_embed_size, bias=False)
        self.dropout = LazyDropout(cfg.dropout)
        self.query_vec_to_action_embed = LazyLinear(
            att_vec_size, action_embed_size, bias=cfg.readout == "non_linear",
        )

        # Embeddings
        self.production_embed = nn.Embedding(
            len(self.grammar.prod2id), action_embed_size
        )
        self.type_embed = nn.Embedding(len(self.grammar.type2id), type_embed_size)
        self.production_readout_b = nn.Parameter(
            torch.FloatTensor(len(self.grammar.prod2id)).zero_()
        )
        self.N_embed = nn.Embedding(
            len(SemQL.N._init_grammar()), action_embed_size
        )
        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.N_embed.weight.data)

        self.read_out_act = (
            torch.tanh if cfg.readout == "non_linear" else nn_utils.identity
        )

        self.production_readout = lambda q: torch.nn.functional.linear(
            self.read_out_act(self.query_vec_to_action_embed(q)),
            self.production_embed.weight,
            self.production_readout_b,
        )

        self.column_pointer_net = PointerNet(
            hidden_size, self.embed_size, attention_type=cfg.column_att
        )

        self.table_pointer_net = PointerNet(
            hidden_size, self.embed_size, attention_type=cfg.column_att
        )

        # New
        self.dot_product_attention = LazyDotProductAttention(dim, dim)
        self.attention_linear = None # lazy and linear + tanh + dropout
        self.decoder_init_linear = None # lazy and linear + tanh
        self.get_action_prob = None # Lazy and linear + softmax

    def one_dim_zero_tensor(self):
        return torch.zeros(self.dim).cuda()

    def forward(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, golds):
        b_size = len(encoded_src)
        if golds:
            # Create sketch gold
            state_class = LSTMStateGold
            states = [LSTMStateGold(encoded_src[b_idx],
                                    encoded_col[b_idx],
                                    encoded_tab[b_idx],
                                    col_tab_dic[b_idx],
                                    golds[b_idx],
                                    )
                      for b_idx in range(b_size)
                      ]
        else:
            state_class = LSTMStatePred
            states = [
                LSTMStatePred(encoded_src[b_idx],
                                    encoded_col[b_idx],
                                    encoded_tab[b_idx],
                                    col_tab_dic[b_idx],
                                    self.grammar.start_symbol,
                              )
                for b_idx in range(b_size)
            ]

        def embed_action(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            action = state.get_prev_sketch()
            action_emb = self.grammar.action_emb[action]
            prev_tensor_dict = {"action_emb": action_emb}
            return prev_tensor_dict

        def embed_symbol(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            symbol = state.get_current_symbol()
            symbol_emb = self.grammar.symbol_emb[symbol]
            prev_tensor_dict.update({"symbol_emb": symbol_emb})
            return prev_tensor_dict

        def pass_sketch_lstm(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            action_emb = prev_tensor_dict["action_emb"].result
            symbol_emb = prev_tensor_dict["symbol_emb"].result
            attn_emb = state.get_attn_emb()
            combined_input = torch.stack([action_emb, symbol_emb, attn_emb])

            lstm_state = state.get_lstm_state()
            next_h_state, next_cell_state = self.sketch_lstm(combined_input, lstm_state)

            prev_tensor_dict.update({"new_lstm_state": (next_h_state, next_cell_state)})
            return prev_tensor_dict

        def update_sketch_lstm(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            new_lstm_state = prev_tensor_dict["new_lstm_state"].result
            state.update_lstm_state(new_lstm_state)
            return prev_tensor_dict

        def pass_sketch_att_src(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            h_state, _ = state.get_lstm_state()
            src_encodings = state.get_src_encodings()
            affined_src_encodings = state.get_affined_src_encodings()
            att_out, att_weights = self.dot_product_attention(affined_src_encodings, h_state,
                                                              src_encodings)  # dot product attention + linear + tanh + dropout
            prev_tensor_dict.update({"att_out": att_out, "att_weights": att_weights})
            return prev_tensor_dict

        def update_sketch_att_src(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            att_out = prev_tensor_dict["att_out"].result
            att_weights = prev_tensor_dict["att_weights"].result
            state.update_att_src(att_out, att_weights)
            return prev_tensor_dict

        def calc_action_prob(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            att_out = prev_tensor_dict["att_out"].result
            action_mask = state.get_mask()
            action_prob = self.action_linear_layer(att_out, action_mask)  # linear layer + linear layer
            prev_tensor_dict.update({"action_prob": action_prob})
            return prev_tensor_dict

        def apply_sketch(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            action_prob = prev_tensor_dict["action_prob"].result
            if isinstance(state, LSTMStateGold):
                gold_action_id = state.get_gold_action_id()
                state.apply_loss(action_prob[gold_action_id])
            else:
                pred_action_id = torch.argmax(action_prob).item()
                action = self.grammar.action_id_to_action(pred_action_id)
                state.save_pred(action)
            state.step_sketch()
            return {}

        def pass_detail_lstm(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            pass

        def update_detail_lstm(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            pass

        def pass_detail_att_src(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            pass

        def update_detail_att_src(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            pass

        def calc_prob(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            pass

        def apply_detail(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict:
            pass

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.detail_not_done).Do(
                LogicUnit.If(state_class.sketch_not_done)
                 .Then(embed_action)
                 .Then(embed_symbol)
                 .Then(pass_sketch_lstm)
                 .Then(update_sketch_lstm)
                 .Then(pass_sketch_att_src)
                 .Then(update_sketch_att_src)
                 .Then(calc_action_prob)
                 .Then(apply_sketch)
            ).Do(
                LogicUnit.If(state_class.sketch_done_detail_not_done)
                .Then(embed_action)
                .Then(embed_symbol)
                .Then(pass_detail_lstm)
                .Then(update_detail_lstm)
                .Then(pass_detail_att_src)
                .Then(update_detail_att_src)
                .Then(calc_prob)
                .Then(apply_detail)
            )
        ).states

        if golds:
            return LSTMStateGold.combine_loss(states)
        else:
            return LSTMStatePred.get_preds(states)
