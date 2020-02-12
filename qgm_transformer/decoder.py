import math
import torch
import torch.nn as nn
from qgm_transformer.grammar import Grammar
from qgm_transformer.batch_state import TransformerBatchState
from qgm_transformer.utils import array_to_tensor
from qgm_transformer.transformer import TransformerDecoderLayer, TransformerDecoder


class QGM_Transformer_Decoder(nn.Module):
    def __init__(self):
        super(QGM_Transformer_Decoder, self).__init__()
        # Grammar
        mani_path = (
            "/home/hkkang/debugging/irnet_qgm_transformer/qgm_transformer/qgm.manifesto"
        )
        self.grammar = Grammar(mani_path)

        # Decode Layers
        dim = 300
        d_model = 300
        self.nhead = 6
        self.affine_layer = nn.Linear(dim, dim)
        self.linear_layer = nn.Linear(d_model, d_model)

        # Transformer Layers
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=self.nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
        self._init_positional_embedding(d_model)

    def _init_positional_embedding(self, d_model, dropout=0.1, max_len=5000):
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
        encoded_src,
        encoded_col,
        encoded_tab,
        src_mask,
        col_mask,
        tab_mask,
        tab_col_dic,
        gold_qgms=None,
    ):
        if gold_qgms:
            gold_qgms = [item.split(" ") for item in gold_qgms]
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
        while not state.is_done():
            # Get sub-mini-batch
            memory = state.get_memory()
            tgt = state.get_tgt()
            memory_mask = state.get_memory_mask(self.nhead, tgt_size=tgt.shape[0])

            # Decode
            tgt = self.pos_encode(tgt)
            out = self.transformer_decoder(
                tgt, memory, memory_mask=memory_mask
            ).transpose(0, 1)
            out = self.linear_layer(out[:, -1:, :])

            # Get views
            action_view, column_view, table_view = state.get_views()
            action_out, column_out, table_out = state.split_by_view(out)

            if action_view:
                # Get input: last action
                current_nodes = action_view.get_current_action_node()
                next_action_ids = array_to_tensor(
                    [
                        self.grammar.get_next_possible_action_ids(cur_node)
                        for cur_node in current_nodes
                    ],
                    dtype=torch.long,
                )

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

                self.predict(action_view, action_out, action_emb, action_masks)

            if column_view:
                # Get input: column encoding
                encoded_cols = column_view.encoded_col

                # Get input mask
                col_masks = column_view.col_mask

                self.predict(column_view, column_out, encoded_cols, col_masks)

            if table_view:
                # Get input: table encoding
                encoded_tabs = table_view.encoded_tab

                # Get input mask
                tab_masks = table_view.tab_mask

                self.predict(table_view, table_out, encoded_tabs, tab_masks)

            # Gather views to original batch state
            state.combine_views(action_view, column_view, table_view)

            # State Transition
            state.update_state()

        # get losses, preds
        return (state.sketch_loss, state.detail_loss), state.pred_history

    def predict(self, view, out, src, src_mask):
        # Calculate similarity
        probs = self.calculate_attention_weights(
            src, out, source_mask=src_mask, affine_layer=self.affine_layer
        )

        if self.training:
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

    def calculate_attention_weights(
        self, source, query, source_mask=None, affine_layer=None
    ):
        if affine_layer:
            source = affine_layer(source)
        weight_scores = torch.bmm(source, query.transpose(1, 2)).squeeze(-1)

        # Masking
        if source_mask is not None:
            weight_scores.data.masked_fill_(source_mask.bool(), -float("inf"))

        weight_probs = torch.log_softmax(weight_scores, dim=-1)

        return weight_probs

    def get_accuracy(self, pred_qgm_actions, gold_qgm_actions):
        # process gold_qgm_actions
        parsed_gold_qgm_actions = []
        for qgm_action in gold_qgm_actions:
            tmp = []
            for item in qgm_action.split(" "):
                symbol = item.split("(")[0]
                idx = item.split("(")[1].split(")")[0]
                tmp += [(symbol, idx)]
            parsed_gold_qgm_actions += [tmp]

        total_acc = {"total": 0.0, "detail": 0.0, "sketch": 0.0}
        is_correct_list = []

        assert len(pred_qgm_actions) == len(parsed_gold_qgm_actions)
        for pred_qgm_action, gold_qgm_action in zip(
            pred_qgm_actions, parsed_gold_qgm_actions
        ):
            # Sketch
            pred_sketch = [
                item for item in pred_qgm_action if item[0] not in ["A", "C", "T"]
            ]
            gold_sketch = [
                item for item in gold_qgm_action if item[0] not in ["A", "C", "T"]
            ]
            sketch_is_correct = pred_sketch == gold_sketch

            # Detail
            pred_detail = [
                item for item in pred_qgm_action if item[0] in ["A", "C", "T"]
            ]
            gold_detail = [
                item for item in gold_qgm_action if item[0] in ["A", "C", "T"]
            ]
            detail_is_correct = pred_detail == gold_detail

            # Total
            total_is_correct = sketch_is_correct and detail_is_correct

            total_acc["detail"] += detail_is_correct
            total_acc["sketch"] += sketch_is_correct
            total_acc["total"] += total_is_correct

            is_correct_list += [total_is_correct]

        for key in total_acc.keys():
            total_acc[key] = total_acc[key] / len(gold_qgm_actions)

        return total_acc, is_correct_list
