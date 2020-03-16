import torch
import torch.nn as nn
from torch import Tensor
from rule.grammar import Symbol, Action
from rule.semql.semql import SemQL
from framework.lazy_modules import LazyDotProductAttention, LazyLinearTanhDropout, LazyLSTMCell, LazyLinearLinear, LazyPointerNet, LazyMemoryPointerNet

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


class SemQLDecoderFramework(nn.Module):
    def __init__(self, cfg):
        super(SemQLDecoderFramework, self).__init__()
        self.cfg = cfg
        self.is_cuda = cfg.cuda != -1
        self.use_column_pointer = cfg.column_pointer
        self.new_tensor = torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor

        dim = cfg.hidden_size
        self.dim = dim
        self.decode_max_time_step = 50

        self.grammar = SemQL(dim)
        self.col_symbol_id = self.grammar.symbol_to_sid["C"]
        self.tab_symbol_id = self.grammar.symbol_to_sid["T"]

        # Further encode
        self.src_lstm = nn.LSTM(dim, dim // 2, bidirectional=True, batch_first=True)
        self.dec_cell_init = nn.Linear(dim, dim)
        self.aff_sketch_linear = nn.Linear(dim, dim)
        self.aff_detail_linear = nn.Linear(dim, dim)
        self.gate_prob_linear = nn.Linear(dim, 1)

        self.sketch_lstm_cell = LazyLSTMCell(dim*3, dim)
        self.detail_lstm_cell = LazyLSTMCell(dim*3, dim)

        # New
        self.column_pointer_net= LazyMemoryPointerNet(dim, dim)
        self.table_pointer_net = LazyPointerNet(dim, dim)
        self.dot_product_attention = LazyDotProductAttention(dim*2, dim)
        self.action_linear_layer = LazyLinearLinear(dim, len(self.grammar.aid_to_action))

    def one_dim_zero_tensor(self):
        return torch.zeros(self.dim).cuda()

    def filter_sketch(self, actions):
        return [action for action in actions if action[0] not in ["T", "C", "A"]]

    def calc_init_state(self, enc_last_cell):
        h_0 = self.dec_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)
        return [(item1, item2) for item1, item2 in zip(h_0, torch.zeros_like(h_0))]

    def action_to_emb(self, state: LSTMState, action: Action) -> torch.Tensor:
        if action[0] == "C":
            col_idx = action[1]
            return state.encoded_col[col_idx]
        elif action[0] == "T":
            tab_idx = action[1]
            return state.encoded_tab[tab_idx]
        else:
            return self.grammar.action_to_emb(action)

    def further_encode(self, inputs, module):
        data_len = [len(item) for item in inputs]
        assert len(inputs[0].shape) == 2, "shape: {}".format(inputs[0].shape)
        stacked_input = torch.zeros(len(inputs), max(data_len), inputs[0].shape[1]).cuda()
        for idx, length in enumerate(data_len):
            stacked_input[idx][:length] = inputs[idx]

        encoded_out = module(stacked_input)

        return [item[:length] for length, item in zip(data_len, encoded_out)]


    def forward(self, enc_last_cell, encoded_src, encoded_col, encoded_tab, col_tab_dic, golds=None):
        b_size = len(encoded_src)

        # Further encode
        dec_init_state = self.calc_init_state(enc_last_cell)
        aff_sketch_src = self.further_encode(encoded_src, self.aff_sketch_linear)
        aff_detail_src = self.further_encode(encoded_src, self.aff_detail_linear)

        if golds:
            # Create sketch gold
            state_class = LSTMStateGold
            states = [LSTMStateGold(aff_sketch_src[b_idx],
                                    aff_detail_src[b_idx],
                                    encoded_src[b_idx],
                                    encoded_col[b_idx],
                                    encoded_tab[b_idx],
                                    dec_init_state[b_idx],
                                    col_tab_dic[b_idx],
                                    golds[b_idx],
                                    self.filter_sketch(golds[b_idx]),
                                    )
                      for b_idx in range(b_size)
                      ]
        else:
            state_class = LSTMStatePred
            states = [
                LSTMStatePred(aff_sketch_src[b_idx],
                                aff_detail_src[b_idx],
                                encoded_src[b_idx],
                                encoded_col[b_idx],
                                encoded_tab[b_idx],
                                dec_init_state[b_idx],
                                col_tab_dic[b_idx],
                                self.grammar.start_symbol,
                              )
                for b_idx in range(b_size)
            ]

        def embed_sketch_action(state: LSTMState, _) -> Dict:
            action = state.get_prev_sketch()
            action_emb = self.action_to_emb(state, action) if action else self.one_dim_zero_tensor()
            prev_tensor_dict = {"action_emb": action_emb}
            return prev_tensor_dict

        def embed_sketch_symbol(state: LSTMState, prev_tensor_dict: Dict[str, Tensor]) -> Dict:
            action = state.get_prev_sketch()
            symbol_emb = self.grammar.symbol_to_emb(action[0]) if action else self.one_dim_zero_tensor()
            prev_tensor_dict.update({"symbol_emb": symbol_emb})
            return prev_tensor_dict

        def pass_sketch_lstm_cell(state: LSTMState, prev_tensor_dict: Dict[str, Tensor]) -> Dict:
            action_emb = prev_tensor_dict["action_emb"]
            symbol_emb = prev_tensor_dict["symbol_emb"]
            attn_emb = state.get_att_emb()
            combined_input = torch.cat([action_emb, symbol_emb, attn_emb], dim=-1)

            lstm_state = state.get_sketch_state()
            next_lstm_state = self.sketch_lstm_cell.forward_later(combined_input, lstm_state)

            prev_tensor_dict.update({"new_lstm_state": next_lstm_state})
            return prev_tensor_dict

        def update_sketch_lstm_state(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            new_lstm_state = prev_tensor_dict["new_lstm_state"].result
            state.update_sketch_state(new_lstm_state)
            return prev_tensor_dict

        def pass_sketch_att_src(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            h_state, _ = state.get_sketch_state()
            src_encodings = state.get_src_encodings()
            aff_sketch_src = state.get_affine_src()
            att_out = self.dot_product_attention.forward_later(aff_sketch_src, h_state, src_encodings)  # dot product attention + linear + tanh + dropout
            prev_tensor_dict.update({"att_out": att_out})
            return prev_tensor_dict

        def update_sketch_att_src(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            att_out = prev_tensor_dict["att_out"].result
            state.update_att_emb(att_out)
            return prev_tensor_dict

        def calc_sketch_prob(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            # Get current symbol
            cur_symbol = state.get_current_symbol()
            action_ids = self.grammar.get_possible_aids(cur_symbol)

            # Calc mask
            action_mask = torch.ones(len(self.grammar.action_to_aid)).cuda()
            action_mask[action_ids] = 0

            att_emb = state.get_att_emb()
            action_prob = self.action_linear_layer.forward_later(att_emb, action_mask)  # linear layer + linear layer
            prev_tensor_dict.update({"action_prob": action_prob})
            return prev_tensor_dict

        def apply_sketch(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            action_prob = prev_tensor_dict["action_prob"].result
            if isinstance(state, LSTMStateGold):
                state.apply_sketch_loss(action_prob)
            else:
                pred_action_id = torch.argmax(action_prob).item()
                action = self.grammar.aid_to_action[pred_action_id]
                nonterminal_symbols = self.grammar.parse_nonterminal_symbol(action)
                state.apply_pred(action, nonterminal_symbols)
            state.step_sketch()
            # print("sketch_step_cnt:{} step_cnt:{}".format(state.sketch_step_cnt, state.step_cnt))
            # if isinstance(state, LSTMStateGold):
            #     print("\tgold: {}".format(state.gold))
            # if isinstance(state, LSTMStatePred):
            #     print("\tpred_sketch: {}".format(state.preds_sketch))
            #     print("\tpred: {}".format(state.preds))
            return {}

        def embed_detail_action(state: LSTMState, _) -> Dict:
            action = state.get_prev_action()
            action_emb = self.action_to_emb(state, action) if action else self.one_dim_zero_tensor()
            prev_tensor_dict = {"action_emb": action_emb}
            return prev_tensor_dict

        def embed_detail_symbol(state: LSTMState, prev_tensor_dict: Dict[str, Tensor]) -> Dict:
            action = state.get_prev_action()
            symbol_emb = self.grammar.symbol_to_emb(action[0]) if action else self.one_dim_zero_tensor()
            prev_tensor_dict.update({"symbol_emb": symbol_emb})
            return prev_tensor_dict

        def pass_detail_lstm_cell(state: LSTMState, prev_tensor_dict: Dict[str, Tensor]) -> Dict:
            action_emb = prev_tensor_dict["action_emb"]
            symbol_emb = prev_tensor_dict["symbol_emb"]
            attn_emb = state.get_att_emb()
            combined_input = torch.cat([action_emb, symbol_emb, attn_emb], dim=-1)

            lstm_state = state.get_detail_state()
            next_state = self.detail_lstm_cell.forward_later(combined_input, lstm_state)

            prev_tensor_dict.update({"new_lstm_state": next_state})
            return prev_tensor_dict

        def update_detail_lstm_state(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            new_lstm_state= prev_tensor_dict["new_lstm_state"].result
            state.update_detail_state(new_lstm_state)
            return prev_tensor_dict

        def pass_detail_att_src(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            h_state, _ = state.get_detail_state()
            src_encodings = state.get_src_encodings()
            aff_detail_src = state.get_affine_src()
            att_out = self.dot_product_attention.forward_later(aff_detail_src, h_state, src_encodings)

            prev_tensor_dict.update({"att_out": att_out})
            return prev_tensor_dict

        def update_detail_att_src(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            att_out = prev_tensor_dict["att_out"].result
            state.update_att_emb(att_out)
            return prev_tensor_dict

        def calc_detail_prob(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            cur_symbol = state.get_current_symbol()

            if cur_symbol == "C":
                att_emb = state.get_att_emb()
                encoded_col = state.get_col_encodings()

                if self.use_column_pointer:
                    # Create column appear mask
                    mem_col_ids = [action[1] for action in state.get_action_history() if action[0] == "C"]
                    col_appear_mask = torch.zeros(encoded_col.shape[0]).cuda()
                    col_appear_mask[mem_col_ids] = 1
                    col_probs = self.column_pointer_net.forward_later(encoded_col, att_emb, col_appear_mask)
                else:
                    col_probs = self.pointer_net(encoded_col, att_emb)

                prev_tensor_dict.update({"prob": col_probs})

            elif cur_symbol == "T":
                prev_action = state.get_prev_action()
                _, column_id = prev_action
                possible_table_ids = state.col_tab_dic[column_id]
                masked_encoded_tab = torch.zeros_like(state.encoded_tab)
                masked_encoded_tab[possible_table_ids] = state.encoded_tab[possible_table_ids]
                att_emb = state.get_att_emb()
                tab_probs = self.table_pointer_net.forward_later(masked_encoded_tab, att_emb)
                prev_tensor_dict.update({"prob": tab_probs})
            else:
                action_ids = self.grammar.get_possible_aids(cur_symbol)
                action_mask = torch.ones(len(self.grammar.action_to_aid)).cuda()
                action_mask[action_ids] = 0
                att_emb = state.get_att_emb()
                action_probs = self.action_linear_layer.forward_later(att_emb, action_mask)
                prev_tensor_dict.update({"prob": action_probs})

            return prev_tensor_dict

        def apply_detail(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromiseOrTensor]) -> Dict:
            prob = prev_tensor_dict["prob"].result
            if isinstance(state, LSTMStateGold):
                state.apply_detail_loss(prob)
            else:
                cur_symbol = state.get_current_symbol()
                if cur_symbol in ["A", "C", "T"]:
                    pred_action_id = torch.argmax(prob).item()
                    if cur_symbol == "A":
                        pred_action = self.grammar.aid_to_action[pred_action_id]
                        assert cur_symbol == pred_action[0], "{} {}".format(cur_symbol, pred_action)
                    else:
                        pred_action = (cur_symbol, pred_action_id)
                    state.edit_pred(pred_action)
            state.step()

            return {}

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done).Do(
                LogicUnit.If(state_class.sketch_not_done)
                 .Then(embed_sketch_action)
                 .Then(embed_sketch_symbol)
                 .Then(pass_sketch_lstm_cell)
                 .Then(update_sketch_lstm_state)
                 .Then(pass_sketch_att_src)
                 .Then(update_sketch_att_src)
                 .Then(calc_sketch_prob)
                 .Then(apply_sketch)
            ).Do(
                LogicUnit.If(state_class.detail_not_done)
                .Then(embed_detail_action)
                .Then(embed_detail_symbol)
                .Then(pass_detail_lstm_cell)
                .Then(update_detail_lstm_state)
                .Then(pass_detail_att_src)
                .Then(update_detail_att_src)
                .Then(calc_detail_prob)
                .Then(apply_detail)
            )
        ).states

        if golds:
            return LSTMStateGold.combine_loss(states)
        else:
            return LSTMStatePred.get_preds(states)
