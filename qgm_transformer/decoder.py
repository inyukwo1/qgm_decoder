import copy
import math
import torch
import torch.nn as nn
from qgm_transformer.rule import Grammar
from qgm_transformer.batch_state import TransformerBatchState


class qgm_transformer_decoder(nn.Module):
    def __init__(self):
        d_model = 512
        # Grammar
        mani_path = "/home/hkkang/debugging/irnet_qgm_transformer/qgm_transformer/qgm.manifesto"
        self.grammar = Grammar(mani_path)

        # Decode Layers
        dim = 300
        self.action_affine_layer = nn.Linear(dim, dim)
        self.col_affine_layer = nn.Linear(dim, dim)
        self.tab_affine_layer = nn.Linear(dim, dim)

        self.linear_layer = nn.Linear(d_model, d_model)

        # Transformer Layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self._init_poistional_emb(d_model)

    def _init_positional_embedding(self, d_model, dropout=0.1, max_len=5000):
        self.pos_dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def pos_encode(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.pos_dropout(x)


    def decode(self,
            encoded_src,
            encoded_col,
            encoded_tab,
            src_mask,
            col_mask,
            tab_mask,
            col_tab_dic,
            gold_qgms=None):
        # Create table to column matching
        tab_col_dic = []
        for b_idx in range(len(col_tab_dic)):
            b_tmp = []
            tab_len = len(col_tab_dic[b_idx][0])
            for t_idx in range(tab_len):
                tab_tmp = [
                    idx
                    for idx in range(len(col_tab_dic[b_idx]))
                    if t_idx in col_tab_dic[b_idx][idx]
                ]
                b_tmp += [tab_tmp]
            tab_col_dic += [b_tmp]

        # Mask Batch State
        state = TransformerBatchState(
            encoded_src,
            encoded_col,
            encoded_tab,
            src_mask,
            col_mask,
            tab_mask,
            tab_col_dic,
            list(range(len(encoded_src))),
            copy.copy(gold_qgms),
        )

        # Decode
        while not state.is_done():
            # Get sub-mini-batch
            memory = state.get_memory()
            pred_history = state.get_pred_history()
            tgt = self.grammar.get_action_emb(pred_history)

            # Mask
            src_masks = state.get_src_mask()

            # Decode
            tgt = self.pos_encode(tgt)
            out = self.transformer_decoder(tgt, memory, src_masks=src_masks)
            outs = self.linear_layer(out[-1, :, :])

            # Get views
            action_view, column_view, table_view = state.get_views()

            if action_view:
                # Out
                action_outs = out[action_view.get_b_indices()]

                # Get action masks
                current_nodes = action_view.get_current_node()
                next_action_ids = [self.grammar.get_next_action_ids(cur_node) for cur_node in current_nodes]
                action_masks = torch.zeros((action_view.get_b_size(), self.grammar.action_len()), dtype=torch.long)
                action_masks[next_action_ids] = 1

                # Forward
                action_probs = self.calculate_attention_weights(action_outs, self.grammar.action_embs,
                                                                self.action_affine_layer, action_masks)

                if self.training:
                    # Get gold
                    action_pred_indices = action_view.get_gold()
                    action_pred_probs = action_probs[action_pred_indices]
                else:
                    action_pred_probs, action_pred_indices = torch.topk(action_probs, 1)

                # Save preds
                action_view.save_losses(action_pred_probs)
                action_view.save_preds(action_pred_indices)

            if column_view:
                # Out
                col_outs = outs[column_view.get_b_indices()]

                # Get input
                encoded_cols = state.encoded_col

                # Get column mask
                col_masks = column_view.col_mask()

                # Forward
                col_probs = self.calculate_attention_weights(col_outs, encoded_cols,
                                                             self.col_affine_layer, col_masks)

                # Get
                if self.training:
                    # Get gold
                    col_pred_indices = column_view.get_gold()
                    col_pred_probs = col_probs[col_pred_indices]
                else:
                    col_pred_probs, col_pred_indices = torch.topk(col_probs, 1)

                # Save
                column_view.save_losses(col_pred_probs)
                column_view.save_preds(col_pred_indices)

            if table_view:
                # Out
                tab_outs = outs[table_view.get_b_indices()]

                # Get input
                encoded_tabs = state.encoded_tab

                # Get table mask
                tab_masks = table_view.tab_mask

                # Forward
                tab_probs = self.calculate_attention_weights(tab_outs, encoded_tabs,
                                                             self.tab_affine_layer, tab_masks)

                if self.training:
                    # Get gold
                    tab_pred_indices = table_view.get_gold()
                    tab_pred_probs = tab_probs[tab_pred_indices]
                else:
                    tab_pred_probs, tab_pred_indices = torch.topk(tab_probs, 1)

                # Save
                table_view.save_losses(tab_pred_probs)
                table_view.save_preds(tab_pred_indices)

            # Gather views to original batch state
            state.combine_views(action_view, column_view, table_view)

            # State Transition
            state.update_state()

        # get losses, preds
        losses = state.losses
        preds = state.pred_history
        return losses, preds


    def calculate_attention_weights(self, query, source, affine_layer, source_mask=None):
        source = affine_layer(source)
        query = query.unsqueeze(-1)
        weight_scores = torch.bmm(source, query).unsqueeze(-1)

        # Masking
        if source_mask is not None:
            weight_scores.data.masked_fill_(source_mask.bool(), -float("inf"))

        weight_probs = torch.log_softmax(weight_scores, dim=-1)

        return weight_probs

