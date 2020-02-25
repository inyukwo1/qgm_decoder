import math
import torch
import torch.nn as nn
import decoder.transformer.utils as utils
from rule.grammar import Grammar
from decoder.transformer.batch_state import TransformerBatchState
from src.transformer_decoder import (
    TransformerDecoderLayer,
    TransformerDecoder,
)


class Transformer_Decoder(nn.Module):
    def __init__(self, cfg):
        super(Transformer_Decoder, self).__init__()
        # Grammar
        is_bert = cfg.is_bert
        grammar_path = cfg.grammar_path
        hidden_size = cfg.hidden_size

        # Decode Layers
        dim = 1024 if is_bert else hidden_size
        d_model = 1024 if is_bert else hidden_size
        self.dim = dim
        self.nhead = cfg.nhead
        self.layer_num = cfg.layer_num
        self.att_affine_layer = nn.Linear(dim, dim)
        self.tgt_affine_layer = nn.Linear(dim, dim)
        self.tgt_linear_layer = nn.Linear(dim * 2, d_model)
        self.out_linear_layer = nn.Linear(d_model, dim)

        self.grammar = Grammar(grammar_path, dim)

        # Transformer Layers
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=self.nhead)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=self.layer_num
        )
        self._init_positional_embedding(d_model)

    def _init_positional_embedding(self, d_model, dropout=0.1, max_len=100):
        self.pos_dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def pos_encode(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.pos_dropout(x)

    def forward(
        self,
        batch,
        step=None,  # for captum
        col_names=None,  # for captum
        tab_names=None): # for captum

        encoded_src = batch.sen_encoding
        encoded_col = batch.col_emb
        encoded_tab = batch.tab_emb
        src_mask = batch.sen_mask
        col_mask = batch.col_mask
        tab_mask = batch.tab_mask
        tab_col_dic = batch.tab_col_dic
        gold_qgms = [item.split(" ") if self.training else [] for item in batch.qgm_action]

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
            self.grammar,
            gold_qgms,
        )

        # Decode
        decode_step_num = 0
        while not state.is_done():
            # Get sub-mini-batch
            memory = state.get_memory()
            tgt = state.get_tgt(self.tgt_affine_layer, self.tgt_linear_layer)
            memory_key_padding_mask = state.get_memory_key_padding_mask()

            # Decode
            tgt = self.pos_encode(tgt)
            out = self.transformer_decoder(
                tgt, memory, memory_key_padding_mask=memory_key_padding_mask
            ).transpose(0, 1)
            out = self.out_linear_layer(out[:, -1:, :])

            # Get views
            action_view, column_view, table_view = state.get_views()
            action_out, column_out, table_out = state.split_by_view(out)

            if action_view:
                # Get input: last action
                current_nodes = action_view.get_current_action_node()
                next_action_ids = [ self.grammar.get_next_possible_action_ids(cur_node) for cur_node in current_nodes ]

                # Get input mask
                action_masks = torch.ones(
                    (action_view.get_b_size(), self.grammar.get_action_len()),
                    dtype=torch.long,
                ).cuda()
                for idx, item in enumerate(next_action_ids):
                    action_masks[idx][item] = 0

                # action to action embedding
                action_emb = self.grammar.action_emb.weight.contiguous()
                action_emb = action_emb.unsqueeze(0).repeat(
                    action_view.get_b_size(), 1, 1
                )
                if step is not None and decode_step_num == step:
                    (
                        gold_index,
                        selected_idx,
                        gold_prob,
                        selected_prob,
                    ) = self.predict_for_captum(
                        action_view, action_out, action_emb, action_masks,
                    )
                    gold_action = self.grammar.action_id_to_action[gold_index]
                    action = self.grammar.action_id_to_action[selected_idx]
                    return (
                        "{}({})".format(gold_action[0], gold_action[1]),
                        "{}({})".format(action[0], action[1]),
                        gold_prob,
                        selected_prob,
                    )

                self.predict(
                    action_view,
                    action_out,
                    action_emb,
                    action_masks,
                    self.training or step is not None,
                )
                decode_step_num += 1

            if column_view:
                # Get input: column encoding
                encoded_cols = column_view.encoded_col

                # Get input mask
                # col_masks = column_view.col_mask
                col_masks = torch.ones(
                    (column_view.get_b_size(), column_view.encoded_col.shape[1]),
                    dtype=torch.long,
                ).cuda()
                for idx, b_idx in enumerate(column_view.b_indices):
                    table_ids = [
                        action[1]
                        for action in column_view.pred_history[b_idx]
                        if action[0] == "T"
                    ]
                    col_ids = []
                    for table_id in table_ids:
                        col_ids += column_view.tab_col_dic[idx][table_id]
                    col_ids = utils.to_long_tensor(list(set(col_ids)))
                    col_masks[idx][col_ids] = 0
                if step is not None and decode_step_num == step:
                    (
                        gold_index,
                        selected_idx,
                        gold_prob,
                        selected_prob,
                    ) = self.predict_for_captum(
                        column_view, column_out, encoded_cols, col_masks,
                    )
                    return (
                        " ".join(col_names[0][gold_index]),
                        " ".join(col_names[0][selected_idx]),
                        gold_prob,
                        selected_prob,
                    )
                self.predict(
                    column_view,
                    column_out,
                    encoded_cols,
                    col_masks,
                    self.training or step is not None,
                )
                decode_step_num += 1

            if table_view:
                # Get input: table encoding
                encoded_tabs = table_view.encoded_tab

                # Get input mask
                tab_masks = table_view.tab_mask
                if step is not None and decode_step_num == step:
                    (
                        gold_index,
                        selected_idx,
                        gold_prob,
                        selected_prob,
                    ) = self.predict_for_captum(
                        table_view, table_out, encoded_tabs, tab_masks,
                    )

                    return (
                        " ".join(tab_names[0][gold_index]),
                        " ".join(tab_names[0][selected_idx]),
                        gold_prob,
                        selected_prob,
                    )
                self.predict(
                    table_view,
                    table_out,
                    encoded_tabs,
                    tab_masks,
                    self.training or step is not None,
                )
                decode_step_num += 1

            # Gather views to original batch state
            state.combine_views(action_view, column_view, table_view)

            # State Transition
            state.update_state()
        assert step is None
        # get losses, preds

        return (
            [item.get_loss_dic() for item in state.loss],
            state.pred_history,
        )

    def predict_for_captum(self, view, out, src, src_mask):
        probs = torch.nn.functional.softmax(
            utils.calculate_attention_weights(
                src, out, source_mask=src_mask, affine_layer=self.att_affine_layer,
            ),
            dim=1,
        )
        gold_indices = view.get_gold()
        gold_index = gold_indices[0]
        selected_idx = torch.argmax(probs, 1)[-1].item()
        gold_prob = probs[:, gold_index]
        selected_prob = probs[:, selected_idx]
        return gold_index, selected_idx, gold_prob, selected_prob

    def predict(self, view, out, src, src_mask, pred_with_gold):
        # Calculate similarity
        probs = utils.calculate_attention_weights(
            src, out, source_mask=src_mask, affine_layer=self.att_affine_layer
        )

        if pred_with_gold:
            pred_indices = view.get_gold()
            pred_probs = []
            for idx, item in enumerate(pred_indices):
                pred_probs += [probs[idx][item]]
        else:
            pred_probs, pred_indices = torch.topk(probs, 1)
            pred_probs = pred_probs.squeeze(1)
            pred_indices = [int(item) for item in pred_indices.squeeze(1)]

        view.save_loss(pred_probs)
        view.save_pred(pred_indices)
