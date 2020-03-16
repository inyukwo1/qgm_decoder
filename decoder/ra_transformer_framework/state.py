import torch
from typing import List, Dict, Union
from framework.sequential_monad import State
from framework.utils import assert_dim
from rule.semql.semql import SemQL
from rule.grammar import SymbolId, Symbol, Action
from rule.semql.semql_loss import SemQL_Loss_New


class RATransformerState(State):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic):
        self.step_cnt = 0
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.col_tab_dic = col_tab_dic

    @classmethod
    def is_not_done(cls, state) -> bool:
        pass

    def is_gold(self) -> bool:
        pass

    def step(self):
        self.step_cnt += 1

    def get_history_actions(self) -> List[Action]:
        pass

    def get_history_symbols(self) -> List[Symbol]:
        return [item[1] for item in self.get_history_actions()]

    def get_current_symbol(self) -> Symbol:
        pass


class RATransformerStateGold(RATransformerState):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, gold: List[Action]):
        RATransformerState.__init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic)
        self.gold: List[Action] = gold
        self.loss = SemQL_Loss_New()

    @classmethod
    def is_not_done(cls, state):
        return state.step_cnt < len(state.gold)

    def is_gold(self):
        return True

    def get_history_actions(self) -> List[Action]:
        return self.gold[: self.step_cnt]

    def get_current_symbol(self) -> Symbol:
        return self.gold[self.step_cnt][0]

    def apply_loss(self, idx, prod: torch.Tensor) -> None:
        gold_action = self.gold[idx]
        gold_symbol = gold_action[0]
        if gold_symbol in {"C", "T"}:
            gold_action_idx = gold_action[1]
        else:
            assert_dim([SemQL.semql.get_action_len()], prod)
            gold_action_idx = SemQL.semql.action_to_aid[gold_action]
        prev_actions: List[Action] = self.gold[: self.step_cnt]
        self.loss.add(-prod[gold_action_idx], gold_symbol, prev_actions)



class RATransformerStatePred(RATransformerState):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, start_symbol: Symbol):
        RATransformerState.__init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic)
        self.preds: List[Action] = []
        self.nonterminal_symbol_stack: List[Symbol] = [start_symbol]

    @classmethod
    def is_not_done(cls, state):
        return len(state.nonterminal_smybol_stack) > 0 and state.step_cnt < 50

    def is_gold(self):
        return False

    def get_history_actions(self) -> List[Action]:
        assert len(self.preds) == self.step_cnt
        return self.preds

    def get_current_symbol(self) -> Symbol:
        return self.nonterminal_symbol_stack[0]

    def apply_pred(self, action: Action, new_nonterminal_symbols: List[Symbol]) -> None:
        self.nonterminal_symbol_stack = new_nonterminal_symbols + self.nonterminal_symbol_stack[1:]
        self.preds.append(action)