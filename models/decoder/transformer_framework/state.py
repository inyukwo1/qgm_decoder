import torch
from typing import List, Dict
from models.framework.sequential_monad import State
from models.framework.utils import assert_dim
from rule.noqgm.noqgm_loss import NOQGM_Loss_New
from rule.grammar import Symbol, Action


class TransformerState(State):
    def __init__(self, grammar, encoded_src, encoded_col, encoded_tab, col_tab_dic):
        self.step_cnt = 0
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.col_tab_dic = col_tab_dic
        self.grammar = grammar

    @classmethod
    def is_not_done(cls, state) -> bool:
        return state.is_to_infer(state)

    @classmethod
    def is_to_infer(cls, state) -> bool:
        pass

    def is_gold(self):
        pass

    def step(self):
        self.step_cnt += 1

    def get_encoded_src(self):
        return self.encoded_src

    def get_encoded_col(self):
        return self.encoded_col

    def get_encoded_tab(self):
        return self.encoded_tab

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
        grammar,
        encoded_src,
        encoded_col,
        encoded_tab,
        col_tab_dic: Dict[int, List[int]],
        gold: List[Action],
    ):
        TransformerState.__init__(
            self, grammar, encoded_src, encoded_col, encoded_tab, col_tab_dic
        )
        self.gold: List[Action] = gold
        self.loss = NOQGM_Loss_New()

    @classmethod
    def is_to_infer(cls, state) -> bool:
        assert state.step_cnt <= len(state.gold)
        return state.step_cnt < len(state.gold)

    @classmethod
    def combine_loss(cls, states: List["TransformerStateGold"]):
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

    def invalid_table_indices(self, idx) -> List[int]:
        prev_col_idx = self.gold[idx - 1][1]
        valid_indices = self.col_tab_dic[prev_col_idx]
        invalid_indices = [
            idx for idx in self.col_tab_dic[0] if idx not in valid_indices
        ]
        return invalid_indices

    def apply_loss(self, idx, prod: torch.Tensor) -> None:
        gold_action = self.gold[idx]
        gold_symbol = gold_action[0]
        if gold_symbol in {"C", "T"}:
            gold_action_idx = gold_action[1]
        else:
            assert_dim([self.grammar.get_action_len()], prod)
            gold_action_idx = self.grammar.action_to_aid[gold_action]
        prev_actions: List[Action] = self.gold[: self.step_cnt]
        self.loss.add(-prod[gold_action_idx], gold_symbol, prev_actions)


class TransformerStatePred(TransformerState):
    def __init__(
        self, grammar, encoded_src, encoded_col, encoded_tab, col_tab_dic,
    ):
        TransformerState.__init__(
            self, grammar, encoded_src, encoded_col, encoded_tab, col_tab_dic
        )
        self.probs: List = []
        self.preds: List[Action] = []
        self.nonterminal_symbol_stack: List[Symbol] = [grammar.start_symbol]

    @classmethod
    def is_to_infer(cls, state) -> bool:
        return state.nonterminal_symbol_stack != [] and state.step_cnt < 12

    @classmethod
    def get_preds(
        cls, states: List["TransformerStatePred"]
    ) -> Dict[str, List[List[Action]]]:
        return [state.preds for state in states]

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

    def invalid_table_indices(self, idx) -> List[int]:
        prev_col_idx = self.preds[idx - 1][1]
        valid_indices = self.col_tab_dic[prev_col_idx]
        invalid_indices = [
            idx for idx in self.col_tab_dic[0] if idx not in valid_indices
        ]
        return invalid_indices

    def apply_pred(self, prod):
        # Save action prob at current step
        self.probs += [prod.cpu().numpy().tolist()]

        # Select highest prob action
        pred_idx = torch.argmax(prod).item()

        current_symbol = self.nonterminal_symbol_stack.pop(0)
        if current_symbol == "C":
            assert_dim([len(self.col_tab_dic)], prod)
            action: Action = (current_symbol, pred_idx)
            new_nonterminal_symbols = []
        else:
            action: Action = self.grammar.aid_to_action[pred_idx]
            new_nonterminal_symbols = self.grammar.parse_nonterminal_symbol(action)
        self.nonterminal_symbol_stack = (
            new_nonterminal_symbols + self.nonterminal_symbol_stack
        )
        self.preds.append(action)
