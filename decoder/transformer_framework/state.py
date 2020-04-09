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
    def is_not_done(cls, mode):
        def is_not_done(state):
            return state.is_to_infer(mode)(state)

        return is_not_done

    @classmethod
    def is_to_infer(cls, mode):
        pass

    @classmethod
    def is_initial_pred(cls, state) -> bool:
        pass

    @classmethod
    def is_to_apply(cls, state) -> bool:
        return state.is_to_refine(state) or state.is_to_arbitrate(state)

    def is_gold(self):
        pass

    def step(self):
        self.step_cnt += 1

    def get_encoded_src(self, mode):
        return self.encoded_src[mode]

    def get_encoded_col(self, mode):
        return self.encoded_col[mode]

    def get_encoded_tab(self, mode):
        return self.encoded_tab[mode]

    def get_history_actions(self, mode) -> List[Action]:
        pass

    def get_history_symbols(self, mode) -> List[Symbol]:
        return [action[0] for action in self.get_history_actions(mode)]

    def get_current_symbol(self, mode) -> Symbol:
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
        perturbed_gold: List[Action],
    ):
        TransformerState.__init__(
            self, encoded_src, encoded_col, encoded_tab, col_tab_dic
        )
        self.gold: List[Action] = gold
        self.perturbed_gold: List[Action] = perturbed_gold
        self.loss = SemQL_Loss_New()
        self.skip_infer = False
        self.skip_refinement = False
        self.skip_arbitrator = False
        self._set_state_to_skip_training()

    def _set_state_to_skip_training(self):
        # Skip inference model
        self.skip_infer = False
        self.skip_refinement = True
        self.skip_arbitrator = True

        if self.skip_infer:
            self.step_cnt = len(self.gold)

    @classmethod
    def is_to_infer(cls, mode):
        def is_to_infer(state):
            assert state.step_cnt <= len(state.gold)
            return state.step_cnt < len(state.gold)

        return is_to_infer

    @classmethod
    def combine_loss(cls, states: List["TransformerStateGold"]) -> SemQL_Loss_New:
        return sum([state.loss for state in states])

    @classmethod
    def is_initial_pred(cls, state) -> bool:
        return False

    def is_gold(self):
        return True

    def get_history_actions(self, mode) -> List[Action]:
        if mode == "infer":
            return self.gold[: self.step_cnt]
        else:
            return self.gold[: self.step_cnt] + self.perturbed_gold[self.step_cnt :]

    def get_current_symbol(self, mode) -> Symbol:
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
        preds=None,
    ):
        TransformerState.__init__(
            self, encoded_src, encoded_col, encoded_tab, col_tab_dic
        )
        self.target_step = target_step
        self.probs = []
        self.preds: List[Action] = preds if preds else []
        self.init_preds: List[Action] = []
        self.refined_preds: List[Action] = []
        self.arbitrated_preds: List[Action] = []
        self.nonterminal_symbol_stack: List[Symbol] = [start_symbol]

    @classmethod
    def is_to_infer(cls, mode):
        def is_to_infer(state):
            if mode == "infer":
                return (
                    state.nonterminal_symbol_stack != []
                    and state.step_cnt < 50
                    and state.step_cnt <= state.target_step
                )
            else:
                return state.step_cnt < len(state.preds)

        return is_to_infer

    @classmethod
    def get_preds(
        cls, states: List["TransformerStatePred"]
    ) -> Dict[str, List[List[Action]]]:
        return {
            "preds": [state.preds for state in states],
            "refined_preds": [state.refined_preds for state in states],
            "arbitrated_preds": [state.arbitrated_preds for state in states],
            "initial_preds": [state.init_preds for state in states],
        }

    @classmethod
    def is_initial_pred(cls, state) -> bool:
        return not state.is_to_infer(state) and state.init_preds == []

    def get_probs(self) -> List[List[Tensor]]:
        return self.probs[self.target_step]

    def is_gold(self):
        return False

    def get_history_actions(self, mode) -> List[Action]:
        if mode == "infer":
            return self.preds[: self.step_cnt]
        else:
            return self.preds

    def get_current_symbol(self, mode) -> Symbol:
        if mode == "infer":
            return (
                self.nonterminal_symbol_stack[0]
                if self.nonterminal_symbol_stack
                else None
            )
        else:
            return self.preds[self.step_cnt][0]

    def impossible_table_indices(self, idx) -> List[int]:
        prev_col_idx = self.preds[idx - 1][1]
        possible_indices = self.col_tab_dic[prev_col_idx]
        impossible_indices = [
            idx for idx in self.col_tab_dic[0] if idx not in possible_indices
        ]
        return impossible_indices

    def save_probs(self, probs):
        self.probs += [probs]

    def apply_pred(self, prod, mode):
        pred_idx = torch.argmax(prod).item()
        if mode == "infer":
            current_symbol = self.nonterminal_symbol_stack.pop(0)
        else:
            current_symbol = self.preds[self.step_cnt][0]
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
        if mode == "infer":
            self.preds.append(action)
        else:
            self.preds[self.step_cnt] = action

    def save_init_preds(self, actions):
        self.init_preds = actions
