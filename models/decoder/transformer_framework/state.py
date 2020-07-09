import torch
from typing import List, Dict, Tuple
from models.framework.sequential_monad import State
from models.framework.utils import assert_dim
from rule.loss import Loss
from rule.acc import Acc
from qgm.qgm_action import Symbol, Action, QGM_ACTION
from qgm.qgm import QGM, SYMBOL_ACTIONS


class TransformerState(State):
    def __init__(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, nlq):
        self.step_cnt = 0
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.col_tab_dic = col_tab_dic
        self.nlq = nlq
        self.history: List[Tuple[Symbol, Action]] = []
        self.history_indices_by_tagging = {"predicate_col": [], "projection_col": []}

    @classmethod
    def is_not_done(cls, state) -> bool:
        return state.is_to_infer(state)

    @classmethod
    def is_to_infer(cls, state) -> bool:
        pass

    def is_gold(self):
        pass

    def ready(self):
        pass

    def get_encoded_src(self):
        return self.encoded_src

    def get_encoded_col(self):
        return self.encoded_col

    def get_encoded_tab(self):
        return self.encoded_tab

    def get_history_symbol_actions(self) -> List[Tuple[Symbol, Action]]:
        pass

    def get_current_symbol(self) -> Symbol:
        pass

    def invalid_table_indices(self, idx) -> List[int]:
        if self.history[idx - 1][0] == "col_previous_key_stack":
            return []
        prev_col_idx = self.history[idx - 1][1]
        valid_indices = self.col_tab_dic[prev_col_idx]
        invalid_indices = [
            idx for idx in self.col_tab_dic[0] if idx not in valid_indices
        ]
        return invalid_indices

    def get_prev_pointer(self):
        pass

    def get_history_indices_by_tagging(self, tagging):
        return self.history_indices_by_tagging[tagging]

    def get_last_symbol_in_history(self, symbol_list):
        for idx in range(len(self.history) - 1, -1, -1):
            symbol, action = self.history[idx]
            if symbol in symbol_list:
                return symbol
        return None

    def is_to_find_key_column(self):
        if len(self.history) < 2:
            return False
        last_symbol, last_action = self.history[-2]
        if last_action == "key":
            assert last_symbol == "col_previous_key_stack"
            return True
        return False

    def invalid_key_column_indices(self, org_idx):
        prev_tab_idx = self.history[org_idx - 1][1]
        valid_cols = [
            col_id
            for col_id in range(len(self.qgm.db["col_set"]))
            if prev_tab_idx in self.col_tab_dic[col_id]
        ]
        valid_col_keys = []
        for col_id in valid_cols:
            for p in self.qgm.db["primary_keys"]:
                if col_id == p:
                    valid_col_keys.append(col_id)
                    break
            for f, _ in self.qgm.db["foreign_keys"]:
                if col_id == f:
                    valid_col_keys.append(col_id)
                    break
        if len(valid_col_keys) == 0:
            valid_col_keys.append(valid_cols[1])
        invalid_indices = [
            col_id
            for col_id in range(len(self.qgm.db["col_set"]))
            if col_id not in valid_col_keys
        ]
        return invalid_indices

    def invalid_prev_column_indices(self):
        valid_cols = [action for symbol, action in self.history if symbol == "C"]
        assert len(valid_cols) >= 1
        invalid_indices = [
            col_id
            for col_id in range(len(self.qgm.db["col_set"]))
            if col_id not in valid_cols
        ]
        return invalid_indices

    def is_to_find_prev_column(self):
        if not self.history:
            return False
        last_symbol, last_action = self.history[-1]
        if last_action == "previous":
            assert last_symbol == "col_previous_key_stack"
            return True
        return False

    def is_to_find_prev_table(self):
        if len(self.history) < 2:
            return False
        last_symbol, last_action = self.history[-2]
        if last_action == "previous":
            assert last_symbol == "col_previous_key_stack"
            return True
        return False


class TransformerStateGold(TransformerState):
    def __init__(
        self,
        cfg,
        encoded_src,
        encoded_col,
        encoded_tab,
        col_tab_dic: Dict[int, List[int]],
        qgm: QGM,
        nlq,
    ):
        TransformerState.__init__(
            self, encoded_src, encoded_col, encoded_tab, col_tab_dic, nlq
        )
        self.soft_labeling = cfg.soft_labeling
        self.label_smoothing = cfg.label_smoothing
        self.qgm = qgm
        self.qgm_construct_generator = qgm.qgm_construct()
        self.current_symbol = None
        self.current_action = None
        self.prev_pointer = None
        self.is_done = False

        self.loss = Loss(
            [symbol for symbol, _ in SYMBOL_ACTIONS] + ["detail", "sketch", "total"]
        )

    @classmethod
    def is_to_infer(cls, state) -> bool:
        return not state.is_done

    @classmethod
    def combine_loss(cls, states: List["TransformerStateGold"]):
        return sum([state.loss for state in states])

    def ready(self):
        if self.current_symbol is not None:
            if self.current_symbol == "C":
                if (
                    self.get_last_symbol_in_history(
                        ["select_col_num", "predicate_col_done"]
                    )
                    == "select_col_num"
                ):
                    self.history_indices_by_tagging["projection_col"] += [
                        len(self.history)
                    ]
                else:
                    self.history_indices_by_tagging["predicate_col"] += [
                        len(self.history)
                    ]
            self.history += [(self.current_symbol, self.current_action)]
        symbol, action, pointer = next(self.qgm_construct_generator)
        self.current_symbol = symbol
        self.current_action = action
        self.prev_pointer = pointer
        if symbol is None:
            self.is_done = True

    def is_gold(self):
        return True

    def get_history_symbol_actions(self) -> List[Tuple[Symbol, Action]]:
        return self.history

    def get_current_symbol(self) -> Symbol:
        return self.current_symbol

    def apply_loss(self, prod: torch.Tensor) -> None:
        gold_symbol = self.current_symbol
        gold_action = self.current_action
        if gold_symbol in {"C", "T"}:
            gold_action_idx = gold_action
        else:
            assert_dim([QGM_ACTION.total_action_len()], prod)
            gold_action_idx = QGM_ACTION.symbol_action_to_action_id(
                gold_symbol, gold_action
            )
        prev_actions = self.history[:]
        self.loss.add(-prod[gold_action_idx], gold_symbol, prev_actions)

    def get_prev_pointer(self):
        return self.prev_pointer


class TransformerStatePred(TransformerState):
    def __init__(
        self,
        encoded_src,
        encoded_col,
        encoded_tab,
        col_tab_dic,
        gold,
        db,
        nlq,
        is_analyze=True,
    ):
        TransformerState.__init__(
            self, encoded_src, encoded_col, encoded_tab, col_tab_dic, nlq
        )
        self.is_analyze = is_analyze
        self.current_symbol = None
        if is_analyze:
            self.qgm = gold
            self.current_action = None
            self.qgm_acc = Acc()
            self.wrong = False
        else:
            self.qgm = QGM(db, False)
            self.prediction_setter = None
        self.prev_pointer = None
        self.qgm_constructor = self.qgm.qgm_construct()
        self.is_done = False

    @classmethod
    def is_to_infer(cls, state) -> bool:
        return state.is_done is not True and len(state.history) < 60

    @classmethod
    def get_preds(cls, states: List["TransformerStatePred"]) -> List[QGM]:
        return [state.qgm for state in states]

    @classmethod
    def get_accs(cls, states: List["TransformerStatePred"]):
        return sum([state.qgm_acc for state in states])

    def ready(self):
        if self.is_analyze:
            symbol, action, prev_pointer = next(self.qgm_constructor)
            self.current_symbol = symbol
            self.current_action = action
            self.prev_pointer = prev_pointer
        else:
            symbol, prev_pointer = next(self.qgm_constructor)
            self.current_symbol = symbol
            self.prev_pointer = prev_pointer

        if symbol is None:
            self.is_done = True
            if self.is_analyze:
                if self.wrong:
                    self.qgm_acc.wrong("total")
                else:
                    self.qgm_acc.correct("total")

    def is_gold(self):
        return False

    def get_history_symbol_actions(self) -> List[Tuple[Symbol, Action]]:
        return self.history

    def get_current_symbol(self) -> Symbol:
        return self.current_symbol

    def apply_pred(self, prod):
        # Select highest prob action
        pred_idx = torch.argmax(prod).item()
        current_symbol = self.current_symbol
        if self.is_analyze:
            action = self.current_action
            if current_symbol in {"C", "T"}:
                pred_action = pred_idx
            else:
                pred_action = QGM_ACTION.action_id_to_action(pred_idx)
            if action == pred_action:
                self.qgm_acc.correct(current_symbol)
            else:
                self.qgm_acc.wrong(current_symbol)
                self.wrong = True
        else:
            if current_symbol in {"C", "T"}:
                action = pred_idx
            else:
                action = QGM_ACTION.action_id_to_action(pred_idx)
            self.qgm.apply_action(current_symbol, action)
        if self.current_symbol == "C":
            if (
                self.get_last_symbol_in_history(
                    ["select_col_num", "predicate_col_done"]
                )
                == "select_col_num"
            ):
                self.history_indices_by_tagging["projection_col"] += [len(self.history)]
            else:
                self.history_indices_by_tagging["predicate_col"] += [len(self.history)]
        self.history.append((current_symbol, action))

    def get_prev_pointer(self):
        return self.prev_pointer
