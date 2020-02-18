import math
import torch
import torch.nn as nn
import qgm_transformer.utils as utils
from qgm_transformer.grammar import Grammar
from qgm_transformer.batch_state import TransformerBatchState
from qgm_transformer.transformer_decoder import (
    TransformerDecoderLayer,
    TransformerDecoder,
)


class QGM_Transformer_Decoder(nn.Module):
    def __init__(self, cfg):
        super(QGM_Transformer_Decoder, self).__init__()
        # Grammar
        is_bert = cfg.is_bert
        grammar_path = cfg.grammar_path
        hidden_size = cfg.hidden_size
        self.grammar = Grammar(is_bert, grammar_path, hidden_size)

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

        # Transformer Layers
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=self.nhead)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=self.layer_num
        )
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
        step=None,  # for captum
        col_names=None,  # for captum
        tab_names=None,  # for captum
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
        decode_step_num = 0
        while not state.is_done():
            # Get sub-mini-batch
            memory = state.get_memory()
            tgt = state.get_tgt(self.tgt_affine_layer, self.tgt_linear_layer)
            memory_mask = state.get_memory_mask(self.nhead, tgt_size=tgt.shape[0])

            # Decode
            tgt = self.pos_encode(tgt)
            if True:
                out = self.transformer_decoder(
                    tgt, memory, memory_mask=memory_mask
                ).transpose(0, 1)
            else:
                memory_key_padding_mask = state.get_memory_key_padding_mask()
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
                next_action_ids = utils.array_to_tensor(
                    [
                        self.grammar.get_next_possible_action_ids(cur_node)
                        for cur_node in current_nodes
                    ]
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
            state.loss,
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

    def get_accuracy(self, pred_qgm_actions, gold_qgm_actions):
        # process gold_qgm_actions
        parsed_gold_qgm_actions = []
        for qgm_action in gold_qgm_actions:
            tmp = []
            for item in qgm_action.split(" "):
                symbol = item.split("(")[0]
                idx = item.split("(")[1].split(")")[0]
                tmp += [(symbol, int(idx))]
            parsed_gold_qgm_actions += [tmp]

        keys = [
            "total",
            "detail",
            "sketch",
            "head_num",
            "head_agg",
            "head_col",
            "quantifier_num",
            "quantifier_tab",
            "local_predicate_num",
            "local_predicate_op",
            "local_predicate_agg",
            "local_predicate_col",
        ]
        total_acc = {key: 0.0 for key in keys}

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

            # More detailed Accs
            # Head Num: Count number of B
            pred_head_num = utils.count_symbol(pred_qgm_action, "H")
            gold_head_num = utils.count_symbol(gold_qgm_action, "H")
            head_num_is_correct = pred_head_num == gold_head_num

            # Head others
            head_agg_is_correct = head_col_is_correct = head_num_is_correct

            if head_num_is_correct:
                # Head Agg: Check A after H
                pred_head_agg = utils.filter_action(pred_qgm_action, "A", ["H"])
                gold_head_agg = utils.filter_action(gold_qgm_action, "A", ["H"])
                head_agg_is_correct = pred_head_agg == gold_head_agg
                # Head col: Check C After H A
                pred_head_col = utils.filter_action(pred_qgm_action, "C", ["H", "A"])
                gold_head_col = utils.filter_action(gold_qgm_action, "C", ["H", "A"])
                head_col_is_correct = pred_head_col == gold_head_col

            total_acc["head_num"] += head_num_is_correct
            total_acc["head_agg"] += head_agg_is_correct
            total_acc["head_col"] += head_col_is_correct

            # Quantifier Num: Count number of Q
            pred_quan_num = utils.count_symbol(pred_qgm_action, "Q")
            gold_quan_num = utils.count_symbol(gold_qgm_action, "Q")
            quan_num_is_correct = pred_quan_num == gold_quan_num

            # Quantifier others
            quan_tab_is_correct = quan_num_is_correct
            if quan_num_is_correct:
                # Quantifier Tab: Check T After Q
                pred_quan_tab = utils.filter_action(pred_qgm_action, "T", ["Q"])
                gold_quan_tab = utils.filter_action(gold_qgm_action, "T", ["Q"])
                quan_tab_is_correct = pred_quan_tab == gold_quan_tab

            total_acc["quantifier_num"] += quan_num_is_correct
            total_acc["quantifier_tab"] += quan_tab_is_correct

            # Predicate Num: Count number of P
            pred_predicate_num = utils.count_symbol(pred_qgm_action, "P")
            gold_predicate_num = utils.count_symbol(gold_qgm_action, "P")
            predicate_num_is_correct = pred_predicate_num == gold_predicate_num

            # Others
            predicate_op_is_correct = (
                predicate_agg_is_correct
            ) = predicate_col_is_correct = predicate_num_is_correct

            # Predicate Num: Count number of P
            if predicate_num_is_correct:
                # Predicate op: Check O After P
                pred_predicate_op = utils.filter_action(pred_qgm_action, "O", ["P"])
                gold_predicate_op = utils.filter_action(gold_qgm_action, "O", ["P"])
                predicate_op_is_correct = pred_predicate_op == gold_predicate_op

                # Predicate Agg: Check A After O
                pred_predicate_agg = utils.filter_action(pred_qgm_action, "A", ["O"])
                gold_predicate_agg = utils.filter_action(gold_qgm_action, "A", ["O"])
                predicate_agg_is_correct = pred_predicate_agg == gold_predicate_agg

                # Predicate Col: Check C After O A
                pred_predicate_col = utils.filter_action(
                    pred_qgm_action, "C", ["O", "A"]
                )
                gold_predicate_col = utils.filter_action(
                    gold_qgm_action, "C", ["O", "A"]
                )
                predicate_col_is_correct = pred_predicate_col == gold_predicate_col

            total_acc["local_predicate_num"] += predicate_num_is_correct
            total_acc["local_predicate_agg"] += predicate_agg_is_correct
            total_acc["local_predicate_col"] += predicate_col_is_correct
            total_acc["local_predicate_op"] += predicate_op_is_correct

        for key in total_acc.keys():
            total_acc[key] = total_acc[key] / len(gold_qgm_actions)

        return total_acc, is_correct_list
