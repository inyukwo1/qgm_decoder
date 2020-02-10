import copy
import torch


class TransformerBatchState:
    def __init__(self,
        encoded_src,
        encoded_col,
        encoded_tab,
        src_mask,
        col_mask,
        tab_mask,
        tab_col_dic,
        batch_indices,
        golds,
    ):
        # Embeddings
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.src_mask = src_mask
        self.col_mask = col_mask
        self.tab_mask = tab_mask
        self.tab_col_dic = tab_col_dic
        self.b_indices = batch_indices
        self.golds = golds

        # Not removing these variables when update (always stay in total batch view)
        self.step_cnt = 0
        self.losses = [torch.tensor(0.0).cuda() for _ in range(self.get_b_size())]
        self.pred_history = [[] for _ in range(self.get_b_size())]
        self.nonterminal_stack = [[] for _ in range(self.get_b_size())]


    def _make_view(self, view_indices):
        new_view = copy.deepcopy(self)
        # Narrowing the view
        new_view.encoded_src = new_view.encoded_src(view_indices)
        new_view.encoded_col = new_view.encoded_col(view_indices)
        new_view.encoded_tab = new_view.encoded_tab(view_indices)
        new_view.src_mask = new_view.src_mask(view_indices)
        new_view.col_mask = new_view.col_mask(view_indices)
        new_view.tab_mask = new_view.tab_mask(view_indices)
        new_view.tab_col_dic = [new_view.tab_col_dic[idx] for idx in view_indices]
        new_view.b_indices = [new_view.b_indices[idx] for idx in view_indices]
        new_view.golds = [new_view.golds[idx] for idx in view_indices]
        return new_view


    def _combine_view(self, view):
        if view:
            assert self.step_cnt == view.step_cnt
            for b_idx in view.b_indices:
                assert len(self.pred_history[b_idx]) + 1 == view.pred_history[b_idx]
                self.pred_history[b_idx] = view.pred_history[b_idx]
                self.nonterminal_stack[b_idx] = view.nonterminal_stack[b_idx]
                self.losses[b_idx] = view.losses[b_idx]


    def get_views(self):
        # Parse view indices
        action_view_indices = []
        column_view_indices = []
        table_view_indices = []
        for idx, b_idx in enumerate(self.b_indices):
            action_type = None
            column_type = None
            table_type = None
            pred_action = self.pred_history[b_idx]
            if pred_action == action_type:
                action_view_indices[idx]
            elif pred_action == column_type:
                column_view_indices += [idx]
            else:
                assert pred_action == table_type
                table_view_indices += [idx]

        # Create views
        action_view = self._make_view(action_view_indices)
        column_view = self._make_view(column_view_indices)
        table_view = self._make_view(table_view_indices)

        return action_view, column_view, table_view


    def update_state(self):
        # Increase step count
        self.step_cnt += 1

        # Update Variables
        next_indices = [idx for idx in range(len(self.b_indices)) if self.nonterminal_stack[idx]]
        self.encoded_src = self.encoded_src[next_indices]
        self.encoded_col = self.encoded_col[next_indices]
        self.encoded_tab = self.encoded_tab[next_indices]
        self.src_mask = self.src_mask[next_indices]
        self.col_mask = self.col_mask[next_indices]
        self.tab_mask = self.tab_mask[next_indices]
        self.tab_col_dic = [self.tab_col_dic[idx] for idx in next_indices]
        self.b_indices = [self.b_indices[idx] for idx in next_indices]
        if self.golds:
            self.golds = [self.golds[idx][1:] for idx in next_indices]


    def combine_views(self, action_view, column_view, table_view):
        self._combine_view(action_view)
        self._combine_view(column_view)
        self._combine_view(table_view)


    def is_done(self):
        return bool(self.b_indices)


    def get_b_size(self):
        return len(self.b_indices)


    def get_b_indices(self):
        return self.b_indices


    def get_memory(self):
        return self.encoded_src.transpose(0, 1)


    def get_pred_history(self):
        view_indices = self.b_indices
        return self.pred_history[view_indices]


    def get_gold(self):
        return self.golds


    def save_losses(self, losses):
        for idx, b_idx in enumerate(self.b_indices):
            self.losses[b_idx] += losses[idx]


    def save_preds(self, preds):
        for idx, b_idx in enumerate(self.b_indices):
            self.pred_history[b_idx] += [preds[idx]]

