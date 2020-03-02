import torch
import torch.nn as nn
import decoder.utils as utils
from rule.semql.semql import SemQL
from decoder.lstm.batch_state import LSTM_Batch_State


class LSTM_Decoder(nn.Module):
    def __init__(self, cfg):
        super(LSTM_Decoder, self).__init__()
        is_bert = cfg.is_bert
        hidden_size = cfg.hidden_size

        # Decode Layers
        dim = 1024 if is_bert else hidden_size
        self.dim = dim
        self.action_affine_layer = nn.Linear(dim, dim)
        self.col_affine_layer = nn.Linear(dim, dim)
        self.tab_affine_layer = nn.Linear(dim, dim)
        self.tgt_affine_layer = nn.Linear(dim, dim)
        self.out_linear_layer = nn.Linear(dim, dim)

        self.grammar = SemQL(dim)
        self.col_symbol_id = self.grammar.symbol_to_sid["C"]
        self.tab_symbol_id = self.grammar.symbol_to_sid["T"]
        self.start_symbol_id = self.grammar.symbol_to_sid[self.grammar.start_symbol]

        # LSTM Decoder
        self.lstm_decoder = nn.LSTMCell(dim * 3, dim)

    def forward(
        self,
        init_lstm_state,
        encoded_src,
        encoded_col,
        encoded_tab,
        src_mask,
        col_mask,
        tab_mask,
        col_tab_dic,
        tab_col_dic,
        golds,
    ):
        init_b_size = encoded_src.shape[0]
        start_symbol_ids = [[self.start_symbol_id] for _ in range(init_b_size)]

        # Mask Batch State
        state = LSTM_Batch_State(
            torch.arange(init_b_size).long(),
            encoded_src,
            encoded_col,
            encoded_tab,
            src_mask,
            col_mask,
            tab_mask,
            col_tab_dic,
            tab_col_dic,
            golds,
            init_lstm_state,
            start_symbol_ids,
        )

        # Set Starting conditions
        losses = [self.grammar.create_loss_object() for _ in range(init_b_size)]
        pred_histories = [[] for _ in range(init_b_size)]
        state.set_state(0, losses, pred_histories)

        # Decode
        while not state.is_done():
            # Get prev action embedding
            if state.step_cnt:
                prev_action_emb = []
                for idx, action in enumerate(state.get_prev_action()):
                    if action[0] in ["C"]:
                        col_idx = action[1]
                        prev_action_emb += [state.get_encoded_col()[idx][col_idx]]
                    elif action[0] in ["T"]:
                        tab_idx = action[1]
                        prev_action_emb += [state.get_encoded_tab()[idx][tab_idx]]
                    else:
                        prev_action_id = (
                            torch.tensor(self.grammar.action_to_aid[action])
                            .long()
                            .cuda()
                        )
                        prev_action_emb += [self.grammar.action_emb(prev_action_id)]
                prev_action_emb = torch.stack(prev_action_emb, dim=0)
            else:
                prev_action_emb = torch.zeros(state.get_b_size(), 300).cuda()

            # Get current node embedding
            cur_node = state.get_current_action_node()
            cur_node_emb = self.grammar.symbol_emb(cur_node)

            # Get attention
            encoded_src = state.get_encoded_src()
            src_mask = state.get_src_mask()
            h_s = state.get_hidden_state()
            src_att = self.attention(encoded_src, src_mask, h_s)

            # Create input
            lstm_input = torch.cat([prev_action_emb, cur_node_emb, src_att], dim=-1)

            # Get lstm state
            lstm_state = state.get_lstm_state()

            new_lstm_state = self.lstm_decoder(lstm_input, lstm_state)

            # Save lstm state
            state.update_lstm_state(new_lstm_state)

            # linear and compute out
            out = self.out_linear_layer(new_lstm_state[0]).unsqueeze(1)

            # Get views
            (
                action_view_indices,
                column_view_indices,
                table_view_indices,
            ) = state.get_view_indices(self.col_symbol_id, self.tab_symbol_id)

            if action_view_indices:
                action_view = state.create_view(action_view_indices)
                action_out = out[action_view_indices]

                # Get last action
                cur_nodes = action_view.get_current_action_node()
                next_aids = [
                    self.grammar.get_possible_aids(int(symbol)) for symbol in cur_nodes
                ]

                action_mask = (
                    torch.ones(action_view.get_b_size(), self.grammar.get_action_len())
                    .long()
                    .cuda()
                )

                for idx, item in enumerate(next_aids):
                    action_mask[idx][item] = 0

                action_emb = self.grammar.action_emb.weight.unsqueeze(0)
                action_emb = action_emb.repeat(action_view.get_b_size(), 1, 1)
                pred_aid = self.predict(
                    action_view,
                    action_out,
                    action_emb,
                    action_mask,
                    self.action_affine_layer,
                )

                # Append pred_history
                actions = [self.grammar.aid_to_action[aid] for aid in pred_aid]
                action_view.insert_pred_history(actions)

                # Get next nonterminals
                nonterminal_symbols = self.grammar.parse_nonterminal_symbols(actions)
                nonterminal_symbol_ids = [
                    self.grammar.symbols_to_sids(symbols)
                    for symbols in nonterminal_symbols
                ]
                action_view.insert_nonterminals(nonterminal_symbol_ids)

            if column_view_indices:
                column_view = state.create_view(column_view_indices)
                column_out = out[column_view_indices]

                encoded_col = column_view.get_encoded_col()
                col_mask = column_view.get_col_mask()

                pred_col_id = self.predict(
                    column_view,
                    column_out,
                    encoded_col,
                    col_mask,
                    self.col_affine_layer,
                )
                actions = [("C", col_id) for col_id in pred_col_id]
                column_view.insert_pred_history(actions)

                # Get next nonterminals
                nonterminal_symbol_ids = [
                    [self.tab_symbol_id] for _ in range(column_view.get_b_size())
                ]
                column_view.insert_nonterminals(nonterminal_symbol_ids)

            if table_view_indices:
                table_view = state.create_view(table_view_indices)
                table_out = out[table_view_indices]

                encoded_tab = table_view.get_encoded_tab()

                # Get Mask from prev col
                prev_col_id = [item[1] for item in table_view.get_prev_action()]
                tab_mask = torch.ones(
                    table_view.get_b_size(), len(table_view.tab_mask[0])
                ).cuda()
                col_tab_dic = table_view.get_col_tab_dic()

                for idx, col_id in enumerate(prev_col_id):
                    tab_ids = col_tab_dic[idx][col_id]
                    tab_mask[idx][tab_ids] = 0

                pred_tab_id = self.predict(
                    table_view, table_out, encoded_tab, tab_mask, self.tab_affine_layer
                )
                actions = [("T", table_id) for table_id in pred_tab_id]
                table_view.insert_pred_history(actions)

                # Get next nonterminals
                table_view.insert_nonterminals(
                    [[] for _ in range(table_view.get_b_size())]
                )

            # State Transition
            state = state.get_next_state()

        # get losses, preds
        return (
            state.loss,
            state.pred_history,
        )

    def attention(self, src, src_mask, hidden_state):
        hidden_state = hidden_state.unsqueeze(1)
        weights = utils.calculate_similarity(
            src,
            hidden_state,
            source_mask=src_mask,
            affine_layer=self.tgt_affine_layer,
            log_softmax=False,
        )

        src = src * weights.unsqueeze(-1)
        src = torch.sum(src, dim=1)

        return src

    def predict(self, view, out, src, src_mask, affine_layer=None):
        # Calculate similarity
        probs = utils.calculate_similarity(
            src, out, source_mask=src_mask, affine_layer=affine_layer
        )

        if self.training:
            golds = view.get_gold()
            if view.nonterminal_stack[0][0] in [self.col_symbol_id, self.tab_symbol_id]:
                pred_indices = [item[1] for item in golds]
            else:
                pred_indices = [self.grammar.action_to_aid[action] for action in golds]
            pred_probs = [probs[idx][item] for idx, item in enumerate(pred_indices)]
        else:
            pred_probs, pred_indices = torch.topk(probs, 1)
            pred_probs = pred_probs.squeeze(1)
            pred_indices = [int(item) for item in pred_indices.squeeze(1)]

        # Append loss
        for idx, value in enumerate(pred_probs):
            cur_symbol = view.nonterminal_stack[idx][0]
            b_idx = view.b_indices[idx]
            prev_actions = view.pred_history[b_idx]
            view.loss[idx].add(-value, cur_symbol, prev_actions)

        return pred_indices
