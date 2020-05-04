import torch
from typing import List, Tuple
from models.framework.sequential_monad import State
from models.framework.utils import assert_dim
from rule.semql.semql import SemQL
from rule.grammar import Symbol, Action
from rule.semql.semql_loss import SemQL_Loss_New

MAX_STEP = 50


class LSTMState(State):
    def __init__(
        self,
        aff_sketch_src,
        aff_detail_src,
        encoded_src,
        encoded_col,
        encoded_tab,
        init_state,
        col_tab_dic,
    ):
        self.step_cnt = 0
        self.sketch_step_cnt = 0
        self.aff_sketch_src = aff_sketch_src
        self.aff_detail_src = aff_detail_src
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.col_tab_dic = col_tab_dic

        self.detail_att_emb = torch.zeros(encoded_src.shape[-1]).cuda()
        self.sketch_att_emb = torch.zeros(encoded_src.shape[-1]).cuda()

        self.sketch_state = init_state
        self.detail_state = init_state

    @classmethod
    def is_not_done(cls, state) -> bool:
        return state.detail_not_done(state) or state.sketch_not_done(state)

    @classmethod
    def detail_not_done(cls, state) -> bool:
        pass

    @classmethod
    def sketch_not_done(cls, state) -> bool:
        pass

    def is_gold(self):
        return isinstance(self, LSTMStateGold)

    def step(self):
        self.step_cnt += 1

    def step_sketch(self):
        self.sketch_step_cnt += 1

    def get_prev_sketch(self) -> Action:
        pass

    def get_prev_action(self):
        pass

    def get_current_symbol(self):
        pass

    def get_action_history(self):
        pass

    def get_att_emb(self):
        return (
            self.sketch_att_emb if self.sketch_not_done(self) else self.detail_att_emb
        )

    def get_sketch_state(self):
        return self.sketch_state

    def get_detail_state(self):
        return self.detail_state

    def get_src_encodings(self):
        return self.encoded_src

    def get_col_encodings(self):
        return self.encoded_col

    def get_affine_src(self):
        if self.sketch_not_done(self):
            return self.aff_sketch_src
        else:
            return self.aff_detail_src

    # Update
    def update_sketch_state(self, new_state: Tuple[torch.Tensor, torch.Tensor]):
        self.sketch_state = new_state

    def update_detail_state(self, new_state: Tuple[torch.Tensor, torch.Tensor]):
        self.detail_state = new_state

    def update_att_emb(self, src_attention_vector: torch.Tensor):
        if self.sketch_not_done(self):
            self.sketch_att_emb = src_attention_vector
        else:
            self.detail_att_emb = src_attention_vector

    def _apply_loss(self, golds, step_cnt, prod):
        gold_action = golds[step_cnt]
        gold_symbol = gold_action[0]
        if gold_symbol in ["C", "T"]:
            gold_action_idx = gold_action[1]
        else:
            assert_dim([SemQL.semql.get_action_len()], prod)
            gold_action_idx = SemQL.semql.action_to_aid[gold_action]
        prev_actions: List[Action] = golds[:step_cnt]
        self.loss.add(-prod[gold_action_idx], gold_symbol, prev_actions)


class LSTMStateGold(LSTMState):
    def __init__(
        self,
        aff_sketch_src,
        aff_detail_src,
        encoded_src,
        encoded_col,
        encoded_tab,
        init_state,
        col_tab_dic,
        gold,
        sketch_gold,
    ):
        LSTMState.__init__(
            self,
            aff_sketch_src,
            aff_detail_src,
            encoded_src,
            encoded_col,
            encoded_tab,
            init_state,
            col_tab_dic,
        )
        self.gold: List[Action] = gold
        self.sketch_gold: List[Action] = sketch_gold
        self.loss = SemQL_Loss_New()

    @classmethod
    def sketch_not_done(cls, state) -> bool:
        return state.sketch_step_cnt < len(state.sketch_gold)

    @classmethod
    def detail_not_done(cls, state) -> bool:
        return not state.sketch_not_done(state) and state.step_cnt < len(state.gold)

    @classmethod
    def combine_loss(cls, states: List["LSTMStateGold"]) -> SemQL_Loss_New:
        return sum([state.loss for state in states])

    def get_prev_sketch(self) -> Action:
        if self.sketch_step_cnt:
            return self.sketch_gold[self.sketch_step_cnt - 1]
        else:
            return []

    def get_prev_action(self) -> Action:
        if self.step_cnt:
            return self.gold[self.step_cnt - 1]
        else:
            return []

    def get_current_symbol(self) -> Symbol:
        return self.gold[self.step_cnt][0]

    def get_action_history(self):
        return self.gold[: self.step_cnt]

    def apply_sketch_loss(self, prod: torch.Tensor) -> None:
        self._apply_loss(self.sketch_gold, self.sketch_step_cnt, prod)

    def apply_detail_loss(self, prod: torch.Tensor) -> None:

        self._apply_loss(self.gold, self.step_cnt, prod)


class LSTMStatePred(LSTMState):
    def __init__(
        self,
        aff_sketch_src,
        aff_detail_src,
        encoded_src,
        encoded_col,
        encoded_tab,
        init_state,
        col_tab_dic,
        start_symbol: Symbol,
    ):
        LSTMState.__init__(
            self,
            aff_sketch_src,
            aff_detail_src,
            encoded_src,
            encoded_col,
            encoded_tab,
            init_state,
            col_tab_dic,
        )
        self.preds: List[Action] = []
        self.preds_sketch: List[Action] = []
        self.nonterminal_symbol_stack: List[Symbol] = [start_symbol]

    @classmethod
    def sketch_not_done(cls, state) -> bool:
        return state.nonterminal_symbol_stack != [] and state.sketch_step_cnt < MAX_STEP

    @classmethod
    def detail_not_done(cls, state) -> bool:
        return (
            not state.sketch_not_done(state)
            and state.step_cnt < len(state.preds)
            and state.step_cnt < MAX_STEP
        )

    @classmethod
    def get_preds(cls, states: List["LSTMStatePred"]) -> List[List[Action]]:
        return [state.preds for state in states]

    def get_prev_sketch(self) -> Action:
        return (
            self.preds_sketch[self.sketch_step_cnt - 1] if self.sketch_step_cnt else []
        )

    def get_prev_action(self):
        return self.preds[self.step_cnt - 1] if self.step_cnt else []

    def get_current_symbol(self):
        return (
            self.nonterminal_symbol_stack[0]
            if self.sketch_not_done(self)
            else self.preds[self.step_cnt][0]
        )

    def get_action_history(self):
        return self.preds[: self.step_cnt]

    def save_sketch_pred(self, action: Action):
        self.preds_sketch += [action]

    def apply_pred(self, action: Action, nonterminal_symbols: List[Symbol]):
        assert action[0] not in ["A", "C", "T"], "Weird value: {}".format(action)
        self.preds_sketch += [action]
        self.preds += [action]

        # Pop one and append nonterminal
        self.nonterminal_symbol_stack.pop(0)
        self.nonterminal_symbol_stack = (
            nonterminal_symbols + self.nonterminal_symbol_stack
        )

        # Filter out details
        while self.nonterminal_symbol_stack and self.nonterminal_symbol_stack[0] == "A":
            self.nonterminal_symbol_stack.pop(0)
            self.preds += [("A", 0), ("C", 0), ("T", 0)]

    def edit_pred(self, action: Action):
        assert self.preds[self.step_cnt][0] == action[0], "Different! {} {} ".format(
            self.preds[self.step_cnt], action
        )
        self.preds[self.step_cnt] = action
