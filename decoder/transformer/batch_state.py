import torch
import copy


class Transformer_Batch_State:
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
        self.nonterminal_stack = nonterminal_stack  # Symbol ids

        # This two are not narrowing down.
        self.loss = None
        self.pred_history = None  # Action

        # only for fine prediction
        self.second_pred_history = None

    def set_state(self, step_cnt, loss, pred_history, second_pred_history=None):
        self.step_cnt = step_cnt
        self.loss = loss
        self.pred_history = pred_history
        self.second_pred_history = second_pred_history

    def get_b_size(self):
        return len(self.b_indices)

    def create_view(self, view_indices):
        if not view_indices:
            return None

        new_view = Transformer_Batch_State(
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
            [self.nonterminal_stack[idx] for idx in view_indices],
        )

        new_view.set_state(
            self.step_cnt, self.loss, self.pred_history, self.second_pred_history
        )
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

    def get_second_view_indices_and_location(self, column_s, table_s, action_s):
        action_view_indices = []
        action_view_location = []
        column_view_indices = []
        column_view_location = []
        table_view_indices = []
        table_view_location = []
        for idx, b_idx in enumerate(self.b_indices):
            actions = self.pred_history[b_idx]
            second_actions = self.second_pred_history[b_idx]
            copied_second_actions = copy.copy(second_actions)
            for action_idx, action in enumerate(actions):
                if action[0] == column_s:
                    if copied_second_actions:
                        second_action = copied_second_actions.pop(0)
                        assert second_action[0] == column_s
                    else:
                        column_view_indices.append(idx)
                        column_view_location.append(action_idx)
                        break
                elif action[0] == table_s:
                    if copied_second_actions:
                        second_action = copied_second_actions.pop(0)
                        assert second_action[0] == table_s
                    else:
                        table_view_indices.append(idx)
                        table_view_location.append(action_idx)
                        break
                elif action[0] == action_s:
                    if copied_second_actions:
                        second_action = copied_second_actions.pop(0)
                        assert second_action[0] == action_s
                    else:
                        action_view_indices.append(idx)
                        action_view_location.append(action_idx)
                        break
        return (
            column_view_indices,
            column_view_location,
            table_view_indices,
            table_view_location,
            action_view_indices,
            action_view_location,
        )

    def get_prev_actions(self):
        return [self.pred_history[b_idx] for b_idx in self.b_indices]

    def get_second_prev_actions(self):
        return [self.second_pred_history[b_idx] for b_idx in self.b_indices]

    def get_prev_action(self):
        return [self.pred_history[b_idx][-1] for b_idx in self.b_indices]

    def get_second_prev_action(self):
        return [self.second_pred_history[b_idx][-1] for b_idx in self.b_indices]

    def get_current_action_node(self):
        return torch.tensor([item[0] for item in self.nonterminal_stack]).long().cuda()

    def get_encoded_src(self):
        return self.encoded_src

    def get_src_mask(self):
        return self.src_mask.bool()

    def get_col_mask(self):
        return self.col_mask

    def get_encoded_col(self):
        return self.encoded_col

    def get_col_tab_dic(self):
        return self.col_tab_dic

    def get_encoded_tab(self):
        return self.encoded_tab

    def get_next_state(self, second=False, second_state_symbols=None):
        next_indices = []
        if second:
            self.nonterminal_stack = [[] for _ in range(self.get_b_size())]
            for idx, b_idx in enumerate(self.b_indices):
                actions = self.pred_history[b_idx]
                second_actions = self.second_pred_history[b_idx]
                copied_second_actions = copy.copy(second_actions)
                for action_idx, action in enumerate(actions):
                    if action[0] in second_state_symbols:
                        if copied_second_actions:
                            second_action = copied_second_actions.pop(0)
                            assert second_action[0] in second_state_symbols
                        else:
                            next_indices.append(idx)
                            self.nonterminal_stack[idx].append(action)
                            break

        else:
            if self.step_cnt <= 50:
                for idx, nonterminal_stack in enumerate(self.nonterminal_stack):
                    if nonterminal_stack:
                        next_indices += [idx]

        next_state = Transformer_Batch_State(
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
            [self.nonterminal_stack[idx] for idx in next_indices],
        )
        if not second:
            next_state.set_state(self.step_cnt + 1, self.loss, self.pred_history)
        else:
            next_state.set_state(
                self.step_cnt + 1,
                self.loss,
                self.pred_history,
                self.second_pred_history,
            )
        return next_state

    def insert_pred_history(self, actions):
        for idx, action in enumerate(actions):
            b_idx = self.b_indices[idx]
            self.pred_history[b_idx].insert(len(self.pred_history[b_idx]), action)

    def insert_second_pred_history(self, actions):
        for idx, action in enumerate(actions):
            b_idx = self.b_indices[idx]
            self.second_pred_history[b_idx].insert(
                len(self.second_pred_history[b_idx]), action
            )

    def insert_nonterminals(self, nonterminals):
        for idx, symbols in enumerate(nonterminals):
            del self.nonterminal_stack[idx][0]
            for symbol in reversed(symbols):
                self.nonterminal_stack[idx].insert(0, symbol)

    def get_gold(self):
        return [golds[self.step_cnt] for golds in self.golds]

    def combine_history(self, second_state_symbols):
        for idx, b_idx in enumerate(self.b_indices):
            actions = self.pred_history[b_idx]
            second_actions = self.second_pred_history[b_idx]
            copied_second_actions = copy.copy(second_actions)
            for action_idx, action in enumerate(actions):
                if action[0] in second_state_symbols:
                    assert copied_second_actions
                    second_action = copied_second_actions.pop(0)
                    assert second_action[0] in second_state_symbols
                    actions[action_idx] = second_action
