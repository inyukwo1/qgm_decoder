import torch
from typing import List, NewType, Dict
from framework.sequential_monad import State
from framework.utils import assert_dim
from rule.semql.semql import SemQL
from rule.grammar import SymbolId, Symbol, Action
from rule.semql.semql_loss import SemQL_Loss_New


class LSTMState(State):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic):
        self.step_cnt = 0
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.col_tab_dic = col_tab_dic

    @classmethod
    def is_not_done(cls, state) -> bool:
        pass

    def is_gold(self):
        return isinstance(self, LSTMStateGold)

    def step(self):
        self.step_cnt += 1

    def get_history_actions(self) -> List[Action]:
        pass

    def get_history_symbols(self) -> List[Symbol]:
        pass

    def get_current_symbol(self) -> Symbol:
        pass

    def impossible_table_indices(self, idx) -> List[int]:
        pass


class LSTMStateGold(LSTMState):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, gold):
        LSTMState.__init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic)
        self.gold: List[Action] = gold
        self.loss = SemQL_Loss_New()
        self.max_step_cnt = 50

    @classmethod
    def is_not_done(cls, state) -> bool:
        pass

    @classmethod
    def combine_loss(cls, states: List["LSTMStateGold"]) -> SemQL_Loss_New:
        pass


class LSTMStatePred(LSTMState):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, start_symbol: Symbol):
        LSTMState.__init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic)
        self.preds: List[Action] = []
        self.nonterminal_symbol_stack: List[Symbol] = [start_symbol]

    @classmethod
    def is_not_done(cls, state) -> bool:
        if state.nonterminal_symbol_stack and state.step_cnt < 50:
            return True
        else:
            return False

    @classmethod
    def get_preds(cls, states: List["LSTMStatePred"]) -> List[List[Action]]:
        pass

