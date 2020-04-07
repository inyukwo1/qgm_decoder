import torch
from torch import Tensor
from typing import List, NewType, Dict
from framework.sequential_monad import State
from framework.utils import assert_dim
from rule.semql.semql import SemQL
from rule.grammar import SymbolId, Symbol, Action
from rule.semql.semql_loss import SemQL_Loss_New


class TransformerState(State):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic):
        self.step_cnt = 0
        self.refine_step_cnt = 0
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.col_tab_dic = col_tab_dic

    @classmethod
    def is_not_done(cls, state) -> bool:
        return state.is_to_refine(state) or state.is_to_infer(state)

    @classmethod
    def is_to_refine(cls, state) -> bool:
        pass

    @classmethod
    def is_to_infer(cls, state) -> bool:
        pass

    def is_gold(self):
        pass

    def step(self):
        self.step_cnt += 1

    def get_encoded_src(self, idx=0):
        return self.encoded_src[idx]

    def get_encoded_col(self, idx=0):
        return self.encoded_col[idx]

    def get_encoded_tab(self, idx=0):
        return self.encoded_tab[idx]

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
        gold: List[Action],
    ):
        TransformerState.__init__(
            self, encoded_src, encoded_col, encoded_tab, col_tab_dic
        )
        self.gold: List[Action] = gold
        self.loss = SemQL_Loss_New()

    @classmethod
    def is_to_refine(cls, state) -> bool:
        return state.step_cnt == len(state.gold) and state.refine_step_cnt < len(
            state.gold
        )

    @classmethod
    def is_to_infer(cls, state) -> bool:
        assert state.step_cnt <= len(state.gold)
        return state.step_cnt < len(state.gold)

    @classmethod
    def combine_loss(cls, states: List["TransformerStateGold"]) -> SemQL_Loss_New:
        return sum([state.loss for state in states])

    def is_gold(self):
        return True

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
            idx for idx in self.col_tab_dic[0] if idx not in possible_indices
        ]
        return impossible_indices

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


class TransformerStatePred(TransformerState):
    def __init__(
        self,
        encoded_src,
        encoded_col,
        encoded_tab,
        col_tab_dic,
        start_symbol: Symbol,
        target_step=100,
    ):
        TransformerState.__init__(
            self, encoded_src, encoded_col, encoded_tab, col_tab_dic
        )
        self.target_step = target_step
        self.probs = []
        self.preds: List[Action] = []
        self.refined_preds: List[Action] = []
        self.arbitrated_preds: List[Action] = []
        self.nonterminal_symbol_stack: List[Symbol] = [start_symbol]

    @classmethod
    def is_to_refine(cls, state) -> bool:
        return (
            state.nonterminal_symbol_stack == []
            and state.refine_step_cnt < state.step_cnt
        )

    @classmethod
    def is_to_infer(cls, state) -> bool:
        return (
            state.nonterminal_symbol_stack
            and state.step_cnt < 50
            and state.step_cnt <= state.target_step
        )

    @classmethod
    def get_preds(
        cls, states: List["TransformerStatePred"]
    ) -> Dict[str, List[List[Action]]]:
        return {
            "preds": [state.preds for state in states],
            "refined_preds": [state.refined_preds for state in states],
            "arbitrated_preds": [state.arbitrated_preds for state in states],
        }

    def get_probs(self) -> List[List[Tensor]]:
        return self.probs[self.target_step]

    def is_gold(self):
        return False

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
            idx for idx in self.col_tab_dic[0] if idx not in possible_indices
        ]
        return impossible_indices

    def save_probs(self, probs):
        self.probs += [probs]

    def apply_pred(self, prod):
        pred_idx = torch.argmax(prod).item()

        current_symbol = self.nonterminal_symbol_stack.pop(0)
        if current_symbol == "C":
            assert_dim([len(self.col_tab_dic)], prod)
            action: Action = (current_symbol, pred_idx)
            new_nonterminal_symbols = ["T"]
        elif current_symbol == "T":
            assert_dim([len(self.col_tab_dic[0])], prod)
            action: Action = (current_symbol, pred_idx)
            new_nonterminal_symbols = []

        else:
            action: Action = SemQL.semql.aid_to_action[pred_idx]
            new_nonterminal_symbols = SemQL.semql.parse_nonterminal_symbol(action)
        self.nonterminal_symbol_stack = (
            new_nonterminal_symbols + self.nonterminal_symbol_stack
        )
        self.preds.append(action)

    def refine_pred(self, action: Action, idx: int = 0):
        # if len(self.refined_preds) < len(self.preds):
        #     self.refined_preds[len(self.refined_preds) : len(self.preds)] = self.preds[
        #         len(self.refined_preds) : len(self.preds)
        #     ]
        # self.refined_preds[idx] = action
        self.refined_preds += [action]

    def arbitrate_pred(self, action: Action, idx: int = 0):
        # if len(self.arbitrated_preds) < len(self.preds):
        #     self.arbitrated_preds[
        #         len(self.arbitrated_preds) : len(self.preds)
        #     ] = self.preds[len(self.arbitrated_preds) : len(self.preds)]
        # self.arbitrated_preds[idx] = action
        self.arbitrated_preds += [action]

    def infer_pred(self, action: Action, idx: int = 0):
        self.preds = self.preds[:idx] + [action]
