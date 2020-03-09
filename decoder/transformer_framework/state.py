import torch
from typing import List, NewType, Dict
from framework.sequential_monad import State
from framework.utils import assert_dim
from rule.semql.semql import SemQL
from rule.grammar import SymbolId, Symbol, Action
from rule.semql.semql_loss import SemQL_Loss_New


class TransformerState(State):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, tab_col_dic):
        self.step_cnt = 0
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.col_tab_dic = col_tab_dic
        self.tab_col_dic = tab_col_dic

    @classmethod
    def is_to_refine(cls, state) -> bool:
        pass

    @classmethod
    def is_not_done(cls, state) -> bool:
        pass

    def is_gold(self):
        return isinstance(self, TransformerStateGold)

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


class TransformerStateGold(TransformerState):
    def __init__(
        self,
        encoded_src,
        encoded_col,
        encoded_tab,
        col_tab_dic: Dict[int, List[int]],
        tab_col_dic: Dict[int, List[int]],
        gold: List[Action],
    ):
        TransformerState.__init__(
            self, encoded_src, encoded_col, encoded_tab, col_tab_dic, tab_col_dic
        )
        self.gold: List[Action] = gold
        self.loss = SemQL_Loss_New()

    @classmethod
    def is_to_refine(cls, state) -> bool:
        return False

    @classmethod
    def is_not_done(cls, state) -> bool:
        assert state.step_cnt <= len(state.gold)
        return state.step_cnt < len(state.gold)

    @classmethod
    def combine_loss(cls, states: List["TransformerStateGold"]) -> SemQL_Loss_New:
        return sum([state.loss for state in states])

    def get_history_actions(self) -> List[Action]:
        return self.gold[: self.step_cnt]

    def get_history_symbols(self) -> List[Symbol]:
        symbol_list = [action[0] for action in self.gold]
        return symbol_list[: self.step_cnt]

    def get_current_symbol(self) -> Symbol:
        return self.gold[self.step_cnt][0] if self.step_cnt < len(self.gold) else None

    def impossible_table_indices(self, idx) -> List[int]:
        prev_col_idx = self.gold[idx - 1][1]
        possible_indices = self.col_tab_dic[prev_col_idx]
        impossible_indices = [
            idx for idx in range(len(self.tab_col_dic)) if idx not in possible_indices
        ]
        return impossible_indices

    def apply_loss(self, idx, prod: torch.Tensor):
        gold_action = self.gold[idx]
        gold_symbol = gold_action[0]
        if gold_symbol in {"C", "T"}:
            gold_action_idx = gold_action[1]
        else:
            assert_dim([SemQL.semql.get_action_len()], prod)
            gold_action_idx = SemQL.semql.action_to_aid[gold_action]
        prev_actions: List[Action] = self.gold[: self.step_cnt]
        self.loss.add(-prod[gold_action_idx], gold_symbol, prev_actions)


class TransformerStatePred(TransformerState):
    def __init__(
        self,
        encoded_src,
        encoded_col,
        encoded_tab,
        col_tab_dic,
        tab_col_dic,
        start_symbol: Symbol,
    ):
        TransformerState.__init__(
            self, encoded_src, encoded_col, encoded_tab, col_tab_dic, tab_col_dic
        )
        self.preds: List[Action] = []
        self.nonterminal_symbol_stack: List[Symbol] = [start_symbol]
        self.refine_step_cnt = 0

    @classmethod
    def is_to_refine(cls, state) -> bool:
        return state.nonterminal_symbol_stack == [] and state.refine_step_cnt < state.step_cnt

    @classmethod
    def is_not_done(cls, state) -> bool:
        if state.nonterminal_symbol_stack and state.step_cnt < 50:
            return True
        else:
            return False

    @classmethod
    def get_preds(cls, states: List["TransformerStatePred"]) -> List[List[Action]]:
        return [state.preds for state in states]

    def get_history_actions(self) -> List[Action]:
        return self.preds[: self.step_cnt]

    def get_history_symbols(self) -> List[Symbol]:
        symbol_list = [action[0] for action in self.preds]
        return symbol_list[: self.step_cnt]

    def get_current_symbol(self) -> Symbol:
        return (
            self.nonterminal_symbol_stack[0] if self.nonterminal_symbol_stack else None
        )

    def impossible_table_indices(self, idx) -> List[int]:
        prev_col_idx = self.preds[idx - 1][1]
        possible_indices = self.col_tab_dic[prev_col_idx]
        impossible_indices = [
            idx for idx in range(len(self.tab_col_dic)) if idx not in possible_indices
        ]
        return impossible_indices

    def apply_pred(self, prod):
        pred_idx = torch.argmax(prod).item()
        current_symbol = self.nonterminal_symbol_stack.pop(0)
        if current_symbol == "C":
            assert_dim([len(self.col_tab_dic)], prod)
            action: Action = (current_symbol, pred_idx)
            new_nonterminal_symbols = ["T"]
        elif current_symbol == "T":
            assert_dim([len(self.tab_col_dic)], prod)
            action: Action = (current_symbol, pred_idx)
            new_nonterminal_symbols = []

        else:
            action: Action = SemQL.semql.aid_to_action[pred_idx]
            new_nonterminal_symbols = SemQL.semql.parse_nonterminal_symbol(action)
        self.nonterminal_symbol_stack = (
            new_nonterminal_symbols + self.nonterminal_symbol_stack
        )
        self.preds.append(action)
