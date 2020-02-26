import torch


class LSTM_Batch_State:
    def __init__(
        self,
        b_indices,
        encoded_src,
        encoded_col,
        encoded_tab,
        src_mask,
        col_mask,
        tab_mask,
        col_tab_dic,
        tab_col_dic,
        golds,
        lstm_state,
        nonterminal_stack,
    ):

        self.step_cnt = 0
        self.b_indices = b_indices
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.src_mask = src_mask
        self.col_mask = col_mask
        self.tab_mask = tab_mask
        self.col_tab_dic = col_tab_dic
        self.tab_col_dic = tab_col_dic
        self.golds = golds
        self.lstm_state = lstm_state
        self.nonterminal_stack = nonterminal_stack  # Symbol ids

        # This two are not narrowing down.
        self.loss = None
        self.pred_history = None  # Action

    def set_state(self, step_cnt, loss, pred_history):
        self.step_cnt = step_cnt
        self.loss = loss
        self.pred_history = pred_history

    def get_b_size(self):
        return len(self.b_indices)

    def create_view(self, view_indices):
        if not view_indices:
            return None

        new_view = LSTM_Batch_State(
            self.b_indices[view_indices],
            self.encoded_src[view_indices],
            self.encoded_col[view_indices],
            self.encoded_tab[view_indices],
            self.src_mask[view_indices],
            self.col_mask[view_indices],
            self.tab_mask[view_indices],
            [self.col_tab_dic[idx] for idx in view_indices],
            [self.tab_col_dic[idx] for idx in view_indices],
            [self.golds[idx] for idx in view_indices],
            (self.lstm_state[0][view_indices], self.lstm_state[1][view_indices]),
            [self.nonterminal_stack[idx] for idx in view_indices],
        )

        new_view.set_state(self.step_cnt, self.loss, self.pred_history)
        return new_view

    def is_done(self):
        return self.get_b_size() == 0

    # Get methods
    def get_view_indices(self, column_sid, table_sid):
        action_view_indices = []
        column_view_indices = []
        table_view_indices = []
        for idx, nonterminal_stack in enumerate(self.nonterminal_stack):
            current_node = nonterminal_stack[0]
            if current_node == column_sid:
                column_view_indices += [idx]
            elif current_node == table_sid:
                table_view_indices += [idx]
            else:
                action_view_indices += [idx]
        return action_view_indices, column_view_indices, table_view_indices

    def get_prev_action(self):
        return [self.pred_history[b_idx][-1] for b_idx in self.b_indices]

    def get_current_action_node(self):
        return torch.tensor([item[0] for item in self.nonterminal_stack]).long().cuda()

    def get_encoded_src(self):
        return self.encoded_src

    def get_src_mask(self):
        return self.src_mask

    def get_hidden_state(self):
        return self.lstm_state[0]

    def get_lstm_state(self):
        return self.lstm_state

    def get_col_mask(self):
        return self.col_mask

    def get_encoded_col(self):
        return self.encoded_col

    def get_col_tab_dic(self):
        return self.col_tab_dic

    def get_encoded_tab(self):
        return self.encoded_tab

    def update_lstm_state(self, new_lstm_state):
        self.lstm_state = new_lstm_state

    def get_next_state(self):
        next_indices = []
        if self.step_cnt <= 50:
            for idx, nonterminal_stack in enumerate(self.nonterminal_stack):
                if nonterminal_stack:
                    next_indices += [idx]

        next_state = LSTM_Batch_State(
            self.b_indices[next_indices],
            self.encoded_src[next_indices],
            self.encoded_col[next_indices],
            self.encoded_tab[next_indices],
            self.src_mask[next_indices],
            self.col_mask[next_indices],
            self.tab_mask[next_indices],
            [self.col_tab_dic[idx] for idx in next_indices],
            [self.tab_col_dic[idx] for idx in next_indices],
            [self.golds[idx] for idx in next_indices],
            (self.lstm_state[0][next_indices], self.lstm_state[1][next_indices]),
            [self.nonterminal_stack[idx] for idx in next_indices],
        )

        next_state.set_state(self.step_cnt + 1, self.loss, self.pred_history)
        return next_state

    def insert_pred_history(self, actions):
        for idx, action in enumerate(actions):
            b_idx = self.b_indices[idx]
            self.pred_history[b_idx].insert(len(self.pred_history[b_idx]), action)

    def insert_nonterminals(self, nonterminals):
        for idx, symbols in enumerate(nonterminals):
            del self.nonterminal_stack[idx][0]
            for symbol in reversed(symbols):
                self.nonterminal_stack[idx].insert(0, symbol)

    def get_gold(self):
        return [golds[self.step_cnt] for golds in self.golds]
