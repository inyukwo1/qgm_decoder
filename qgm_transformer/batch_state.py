import copy
import torch
from qgm_transformer.utils import array_to_tensor


class TransformerBatchState:
    def __init__(
        self,
        encoded_src,
        encoded_col,
        encoded_tab,
        src_mask,
        col_mask,
        tab_mask,
        tab_col_dic,
        b_indices,
        grammar,
        golds,
        state_type=None,
    ):
        self.grammar = grammar
        self.state_type = state_type

        # Embeddings
        self.encoded_src = encoded_src
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.src_mask = src_mask
        self.col_mask = col_mask
        self.tab_mask = tab_mask
        self.tab_col_dic = tab_col_dic
        self.b_indices = b_indices
        self.golds = copy.deepcopy(golds)

        # Not removing these variables when update (always stay in total batch view)
        self.step_cnt = 0
        # Global
        self.nonterminal_stack = [
            [grammar.get_start_symbol_id()] for _ in range(self.get_b_size())
        ]
        self.sketch_loss = [torch.tensor(0.0).cuda() for _ in range(self.get_b_size())]
        self.detail_loss = [torch.tensor(0.0).cuda() for _ in range(self.get_b_size())]
        self.pred_history = [[] for _ in range(self.get_b_size())]

    def _make_view(self, view_indices, state_type):
        if not view_indices:
            return None
        # Narrowing the view
        new_view = TransformerBatchState(
            self.encoded_src[view_indices],
            self.encoded_col[view_indices],
            self.encoded_tab[view_indices],
            self.src_mask[view_indices],
            self.col_mask[view_indices],
            self.tab_mask[view_indices],
            copy.deepcopy([self.tab_col_dic[idx] for idx in view_indices]),
            copy.deepcopy([self.b_indices[idx] for idx in view_indices]),
            self.grammar,
            copy.deepcopy([self.golds[idx] for idx in view_indices]),
            state_type=state_type,
        )
        new_view.step_cnt = self.step_cnt
        new_view.sketch_loss = self.sketch_loss
        new_view.detail_loss = self.detail_loss
        new_view.pred_history = copy.deepcopy(self.pred_history)
        new_view.nonterminal_stack = copy.deepcopy(self.nonterminal_stack)
        return new_view

    def _combine_view(self, view):
        if view:
            assert self.step_cnt == view.step_cnt
            for idx, b_idx in enumerate(view.b_indices):
                assert len(self.pred_history[b_idx]) + 1 == len(
                    view.pred_history[b_idx]
                )
                self.nonterminal_stack[b_idx] = view.nonterminal_stack[b_idx]
                self.pred_history[b_idx] = view.pred_history[b_idx]
                if view.state_type == "action":
                    self.sketch_loss[b_idx] = view.sketch_loss[b_idx]
                else:
                    self.detail_loss[b_idx] = view.detail_loss[b_idx]

    def _get_view_indices(self):
        # Parse view indices
        action_view_indices = []
        column_view_indices = []
        table_view_indices = []
        for idx, b_idx in enumerate(self.b_indices):
            pred_action = self.nonterminal_stack[b_idx][0]
            if pred_action == self.grammar.symbol_to_symbol_id["T"]:
                table_view_indices += [idx]
            elif pred_action == self.grammar.symbol_to_symbol_id["C"]:
                column_view_indices += [idx]
            else:
                action_view_indices += [idx]
        return action_view_indices, column_view_indices, table_view_indices

    def split_by_view(self, out):
        # Get view indices
        (
            action_view_indices,
            column_view_indices,
            table_view_indices,
        ) = self._get_view_indices()

        return (
            out[action_view_indices],
            out[column_view_indices],
            out[table_view_indices],
        )

    def get_views(self):
        # Get view indices
        (
            action_view_indices,
            column_view_indices,
            table_view_indices,
        ) = self._get_view_indices()

        # Create views
        action_view = self._make_view(action_view_indices, "action")
        column_view = self._make_view(column_view_indices, "column")
        table_view = self._make_view(table_view_indices, "table")

        return action_view, column_view, table_view

    def update_state(self):
        # Increase step count
        self.step_cnt += 1

        # Update Variables
        if self.step_cnt == 30:
            next_indices = []
        else:
            next_indices = [
                idx
                for idx, b_idx in enumerate(self.b_indices)
                if self.nonterminal_stack[b_idx]
            ]
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

    def get_current_action_node(self):
        symbols = [self.nonterminal_stack[b_idx][0] for b_idx in self.b_indices]
        return symbols

    def is_done(self):
        return not bool(self.b_indices)

    def get_b_size(self):
        return len(self.b_indices)

    def get_b_indices(self):
        return self.b_indices

    def get_memory(self):
        return self.encoded_src.transpose(0, 1)

    def get_memory_mask(self, nhead, tgt_size):
        # Expand as nhead
        mask = self.src_mask.repeat(nhead, 1)

        # Expand as length of tgt
        mask = mask.unsqueeze(1).repeat(1, tgt_size, 1)

        return mask

    def get_tgt(self):
        view_indices = self.b_indices

        # Get prev symbols
        prev_symbols = [
            self.pred_history[idx] if self.pred_history[idx] else []
            for idx in view_indices
        ]

        # To symbols
        symbols = self.grammar.action_to_symbol(prev_symbols)
        for idx, b_idx in enumerate(view_indices):
            symbols[idx] += [self.nonterminal_stack[b_idx][0]]

        # To tgt
        symbols = array_to_tensor(symbols, torch.long)
        tgt = self.grammar.symbol_emb(symbols)

        return tgt.transpose(0, 1)

    def get_gold(self):
        action_ids = []
        for item in self.golds:
            symbol = item[0].split("(")[0]
            symbol_idx = int(item[0].split("(")[1].split(")")[0])
            if symbol in ["C", "T"]:
                action_ids += [symbol_idx]
            else:
                action_ids += [self.grammar.action_to_action_id[(symbol, symbol_idx)]]
        return action_ids

    def save_loss(self, loss):
        for idx, b_idx in enumerate(self.b_indices):
            if self.state_type == "action":
                self.sketch_loss[b_idx] -= loss[idx]
            else:
                self.detail_loss[b_idx] -= loss[idx]

    def save_pred(self, preds):
        for idx, b_idx in enumerate(self.b_indices):
            if self.state_type == "action":
                action = self.grammar.action_id_to_action[preds[idx]]
                self.pred_history[b_idx] += [action]

                # If non-terminal add to nonterminals
                next_symbols = [
                    symbol
                    for symbol in self.grammar.get_next_action(*action).split(" ")
                    if symbol not in self.grammar.terminals
                ]
                next_symbol_ids = [
                    self.grammar.symbol_to_symbol_id[symbol] for symbol in next_symbols
                ]
                self.nonterminal_stack[b_idx] = (
                    next_symbol_ids + self.nonterminal_stack[b_idx][1:]
                )

            elif self.state_type == "column":
                self.pred_history[b_idx] += [("C", preds[idx])]
                self.nonterminal_stack[b_idx] = self.nonterminal_stack[b_idx][1:]
            elif self.state_type == "table":
                self.pred_history[b_idx] += [("T", preds[idx])]
                self.nonterminal_stack[b_idx] = self.nonterminal_stack[b_idx][1:]
            else:
                raise RuntimeError("Should not be here")
