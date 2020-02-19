import copy
import torch
import qgm_transformer.utils as utils
from qgm_transformer.QGMLoss import QGMLoss
import random


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
        self.loss = [QGMLoss(self.grammar) for _ in range(self.get_b_size())]
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
        new_view.loss = self.loss
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
                self.loss[b_idx] = self.loss[b_idx]

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

    def history_action_indices(self):
        b_history_action_emb_indices = []
        b_history_action_nodes = []
        b_history_action_gold_indices = []
        for b_idx in self.b_indices:
            emb_indices = []
            action_nodes = []
            gold_indices = []
            for idx, action in enumerate(self.pred_history[b_idx]):
                if action[0] not in {"T", "C"}:
                    emb_indices.append(idx)
                    action_nodes.append(action[0])
                    gold_indices.append(self.grammar.action_to_action_id[action])
            emb_indices = torch.tensor(emb_indices).cuda()
            gold_indices = torch.tensor(gold_indices).cuda()
            b_history_action_emb_indices.append(emb_indices)
            b_history_action_nodes.append(action_nodes)
            b_history_action_gold_indices.append(gold_indices)
        return (
            b_history_action_emb_indices,
            b_history_action_nodes,
            b_history_action_gold_indices,
        )

    def history_col_indices(self):
        b_history_col_emb_indices = []
        b_history_tab_indices = []
        b_history_col_gold_indices = []
        for b_idx in self.b_indices:
            col_indices = []
            tab_indices = []
            gold_indices = []
            for idx, action in enumerate(self.pred_history[b_idx]):
                if action[0] == "C":
                    col_indices.append(idx)
                    gold_indices.append(action[1])
                if action[0] == "T":
                    tab_indices.append(action[1])
            col_indices = torch.tensor(col_indices).cuda()
            b_history_col_emb_indices.append(col_indices)
            b_history_tab_indices.append(tab_indices)
            b_history_col_gold_indices.append(gold_indices)
        return (
            b_history_col_emb_indices,
            b_history_tab_indices,
            b_history_col_gold_indices,
        )

    def history_tab_indices(self):
        b_history_tab_emb_indices = []
        b_history_tab_gold_indices = []
        for b_idx in self.b_indices:
            tab_indices = []
            gold_indices = []
            for idx, action in enumerate(self.pred_history[b_idx]):
                if action[0] == "T":
                    tab_indices.append(idx)
                    gold_indices.append(action[1])
            tab_indices = torch.tensor(tab_indices).cuda()
            b_history_tab_emb_indices.append(tab_indices)
            b_history_tab_gold_indices.append(gold_indices)
        return (
            b_history_tab_emb_indices,
            b_history_tab_gold_indices,
        )

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

    def get_memory_key_padding_mask(self):
        # Expand as nhead
        mask = self.src_mask
        return mask.bool()

    def get_tgt(self, affine_layer, linear_layer, training):
        # Get prev symbols
        prev_actions = [
            self.pred_history[b_idx] if self.pred_history[b_idx] else []
            for b_idx in self.b_indices
        ]

        # To symbols
        symbols = self.grammar.actions_to_symbols(prev_actions)
        for idx, b_idx in enumerate(self.b_indices):
            symbols[idx] += [self.nonterminal_stack[b_idx][0]]

        # Get symbol embedding
        symbols = utils.to_long_tensor(symbols)
        symbol_embs = self.grammar.symbol_emb(symbols)

        # Get action embedding
        action_embs = torch.zeros(
            self.get_b_size(), 1, self.grammar.action_emb.embedding_dim
        ).cuda()
        if self.step_cnt > 0:
            stacked_action_emb_list = []
            for b_idx, actions in enumerate(prev_actions):
                action_emb_list = []
                for s_idx, action in enumerate(actions):
                    if training and random.randint(0, 100) < 10:
                        action_emb_list += [
                            torch.zeros(self.grammar.action_emb.embedding_dim).cuda()
                        ]
                    else:
                        # Table
                        if action[0] == "T":
                            action_emb_list += [self.encoded_tab[b_idx][action[1]]]
                        # Column
                        elif action[0] == "C":
                            action_emb_list += [self.encoded_col[b_idx][action[1]]]
                        # Action
                        else:
                            action_emb_list += [
                                self.grammar.action_emb(
                                    utils.to_long_tensor(
                                        self.grammar.action_to_action_id[action]
                                    )
                                )
                            ]
                stacked_action_emb_list += [torch.stack(action_emb_list)]
            action_embs = torch.cat(
                (torch.stack(stacked_action_emb_list), action_embs), dim=1
            )

        # Linear Layer
        action_embs = affine_layer(action_embs)

        # Concatenate
        tgt = torch.cat((symbol_embs, action_embs), dim=-1)
        tgt = linear_layer(tgt)

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
            self.loss[b_idx].add(
                -loss[idx], self.nonterminal_stack[b_idx][0], self.pred_history[b_idx]
            )

    def save_aux_loss(self, b_idx, loss):
        self.loss[b_idx].add_aux(-loss)

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
