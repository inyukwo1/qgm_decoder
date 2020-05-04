import copy
from models.framework.sequential_monad import State


class EnsembleState(State):
    def __init__(self, src_len, col_len, tab_len, col_tab_dic, gt):
        self.step_cnt = 0
        self.encoded_src = []
        self.encoded_col = []
        self.encoded_tab = []
        self.src_len = src_len
        self.col_len = col_len
        self.tab_len = tab_len
        self.col_tab_dic = col_tab_dic
        self.gt = gt
        self.preds = []
        self.nonterminals = ["Root"]
        self.child_states = []

    @classmethod
    def is_not_done(cls, state) -> bool:
        return state.nonterminals != [] and state.step_cnt < 50

    @classmethod
    def get_final_pred(cls, states):
        return [state.preds for state in states]

    def get_prev_action(self):
        return self.preds[-1]

    def get_encoded_src(self, idx):
        return self.encoded_src[idx].unsqueeze(0)

    def get_encoded_col(self, idx):
        return self.encoded_col[idx].unsqueeze(0)

    def get_encoded_tab(self, idx):
        return self.encoded_tab[idx].unsqueeze(0)

    def get_src_lens(self):
        return [self.src_len]

    def get_col_lens(self):
        return [self.col_len]

    def get_tab_lens(self):
        return [self.tab_len]

    def get_col_tab_dic(self):
        return [self.col_tab_dic]

    def get_preds(self):
        return [self.preds]

    def get_pred_len(self):
        return len(self.preds)

    def get_first_nonterminal(self):
        return self.nonterminals[0]

    def get_child_states(self):
        return self.child_states

    def append_encoded_values(self, src, col, tab):
        self.encoded_src.append(src)
        self.encoded_col.append(col)
        self.encoded_tab.append(tab)

    def save_pred(self, action, nonterminals):
        self.step_cnt += 1
        self.preds += [action]
        self.nonterminals = nonterminals + self.nonterminals[1:]

    def save_children_state(self, states):
        self.child_states = states

    def update_children_state(self):
        for idx, state in enumerate(self.child_states):
            state.preds[-1] = self.preds[-1]
            state.nonterminal_symbol_stack = copy.deepcopy(self.nonterminals)
