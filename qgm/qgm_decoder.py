import torch
import torch.nn as nn
from qgm.ops import WHERE_OPS, BOX_OPS, AGG_OPS, IUEN
from qgm.utils import (
    get_iue_box_op,
    split_boxes,
    get_is_box_info,
    get_quantifier_num_with_type,
    get_quantifier_with_type,
    get_local_predicate_num,
    get_local_predicate_agg_with_idx,
    get_local_predicate_col_with_idx,
    get_local_predicate_op_with_idx,
    to_tensor,
    get_head_num,
    get_head_col,
    get_head_agg_with_idx,
    get_limit_num,
    get_is_asc,
    get_box_with_op_type,
    compare_boxes,
)


class QGM_Decoder(nn.Module):
    def __init__(self, emb_dim):
        super(QGM_Decoder, self).__init__()
        att_vec_dim = emb_dim
        input_dim = emb_dim + att_vec_dim
        # Prev
        self.att_linear = nn.Linear(emb_dim, emb_dim, bias=False)
        # Embeddings
        self.emb_dim = emb_dim
        self.p_op_emb = nn.Embedding(len(WHERE_OPS), emb_dim)
        self.box_op_emb = nn.Embedding(len(BOX_OPS), emb_dim)
        self.agg_emb = nn.Embedding(len(AGG_OPS), emb_dim)
        # LSTM for state transition
        self.lstm = nn.LSTMCell(input_dim, emb_dim)
        self.att_layer = nn.Sequential(
            torch.nn.Linear(emb_dim * 2, att_vec_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        # Box Encode
        self.box_encode_layer = nn.Sequential(
            torch.nn.Linear(emb_dim * 3, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        # Sketch
        self.iuen_linear = nn.Linear(att_vec_dim, 4)
        self.is_group_by = nn.Linear(att_vec_dim, 2)
        self.is_order_by = nn.Linear(att_vec_dim, 2)
        # Select
        self.select_qf_num = nn.Linear(att_vec_dim, 10)
        self.select_qs_num = nn.Linear(att_vec_dim, 4)
        self.select_p_num = nn.Linear(att_vec_dim, 5)
        self.select_p_layer = nn.Linear(att_vec_dim + emb_dim * 3, emb_dim)
        self.select_p_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        self.select_p_op = nn.Linear(att_vec_dim, len(WHERE_OPS))
        self.select_head_layer = nn.Linear(att_vec_dim + emb_dim, att_vec_dim)
        self.select_head_num = nn.Linear(att_vec_dim, 6)
        self.select_head_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        # Group by
        self.group_qf_num = nn.Linear(att_vec_dim, 4)
        self.group_qs_num = nn.Linear(att_vec_dim, 3)
        self.group_p_num = nn.Linear(att_vec_dim, 5)
        self.group_p_layer = nn.Linear(att_vec_dim + emb_dim * 3, att_vec_dim)
        self.group_p_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        self.group_p_op = nn.Linear(att_vec_dim, len(WHERE_OPS))
        self.group_head_layer = nn.Linear(att_vec_dim + emb_dim, att_vec_dim)
        self.group_head_num = nn.Linear(att_vec_dim, 3)
        self.group_head_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        # Order by
        self.order_qf_num = nn.Linear(att_vec_dim, 4)
        self.order_p_num = nn.Linear(att_vec_dim, 5)
        self.order_p_layer = nn.Linear(att_vec_dim + emb_dim * 3, att_vec_dim)
        self.order_p_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        self.order_p_op = nn.Linear(att_vec_dim, len(WHERE_OPS))
        self.order_head_layer = nn.Linear(att_vec_dim + emb_dim, att_vec_dim)
        self.order_head_num = nn.Linear(att_vec_dim, 3)
        self.order_head_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        self.order_is_asc = nn.Linear(att_vec_dim, 2)
        self.order_limit_num = nn.Linear(att_vec_dim, 3)

        self.select_layers = {
            "qf_num": self.select_qf_num,
            "qs_num": self.select_qs_num,
            "p_num": self.select_p_num,
            "p_layer": self.select_p_layer,
            "p_agg": self.select_p_agg,
            "p_op": self.select_p_op,
            "head_layer": self.select_head_layer,
            "head_num": self.select_head_num,
            "head_agg": self.select_head_agg,
        }

        self.group_layers = {
            "qf_num": self.group_qf_num,
            "qs_num": self.group_qs_num,
            "p_num": self.group_p_num,
            "p_layer": self.group_p_layer,
            "p_agg": self.group_p_agg,
            "p_op": self.group_p_op,
            "head_layer": self.group_head_layer,
            "head_num": self.group_head_num,
            "head_agg": self.group_head_agg,
        }

        self.order_layers = {
            "qf_num": self.order_qf_num,
            "p_num": self.order_p_num,
            "p_layer": self.order_p_layer,
            "p_agg": self.order_p_agg,
            "p_op": self.order_p_op,
            "head_layer": self.order_head_layer,
            "head_num": self.order_head_num,
            "head_agg": self.order_head_agg,
            "is_asc": self.order_is_asc,
            "limit_num": self.order_limit_num,
        }

        # Store variables
        self.encoded_src = None
        self.encoded_src_att = None
        self.encoded_col = None
        self.encoded_tab = None

    def set_variables(
        self,
        encoded_src,
        encoded_col,
        encoded_tab,
        src_mask,
        col_mask,
        tab_mask,
        col_tab_dic,
    ):
        self.encoded_src = encoded_src
        self.encoded_src_att = self.att_linear(encoded_src)
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.src_mask = src_mask
        self.col_mask = col_mask
        self.tab_mask = tab_mask
        self.col_tab_dic = col_tab_dic
        self.tab_col_dic = []
        for b_idx in range(len(col_tab_dic)):
            tmp = []
            tab_len = len(col_tab_dic[b_idx][0])
            for t_idx in range(tab_len):
                tmp += [
                    idx
                    for idx in range(len(col_tab_dic[b_idx]))
                    if t_idx in col_tab_dic[b_idx][idx]
                ]
            self.tab_col_dic += [tmp]

    def decode(self, b_indices, att, state_vector, prev_box=None, gold_boxes=None):
        b_size = len(b_indices)

        # att_linear
        x = (
            self.compute_context_vector(b_indices, None, state_vector)
            if att
            else torch.zeros(b_size, self.lstm.input_size).cuda()
        )
        state_vector, att = self.compute_context_vector(b_indices, x, state_vector)

        iuen_score = self.iuen_linear(att)
        iuen_prob = torch.log_softmax(iuen_score, dim=-1)

        # Choose box_operator
        if self.training:
            op_keys = get_iue_box_op(gold_boxes)
            op_selected_idx = to_tensor(
                [
                    IUEN.index(op_key) if op_key else IUEN.index("select")
                    for op_key in op_keys
                ]
            )
            op_selected_prob = torch.gather(iuen_prob, 1, op_selected_idx)
        else:
            op_selected_prob, op_selected_idx = torch.topk(iuen_prob, 1)

        # Split boxes
        if self.training:
            gold_front_boxes, gold_rear_boxes = split_boxes(gold_boxes)
        else:
            gold_front_boxes = gold_rear_boxes = None

        # Calculate state vector
        state_vector, losses, pred_front_boxes = self.decode_boxes(
            b_indices, att, state_vector, prev_box=prev_box, gold_boxes=gold_front_boxes
        )
        pred_boxes = pred_front_boxes

        next_operators = [item for item in op_selected_idx if item != 0]
        next_b_indices = [
            b_indices[idx] for idx in range(b_size) if op_selected_idx[idx] != 0
        ]
        next_b_indices = torch.tensor(next_b_indices).cuda() if next_b_indices else None

        if next_b_indices is not None:
            # get sub att and state_vector
            att = att[next_b_indices]
            next_state_vector = [
                state_vector[0][next_b_indices],
                state_vector[1][next_b_indices],
            ]
            prev_boxes = [pred_boxes[idx][-1] for idx in next_b_indices]

            next_state_vector, rear_losses, pred_rear_boxes = self.decode_boxes(
                next_b_indices,
                att,
                next_state_vector,
                prev_box=prev_boxes,
                gold_boxes=gold_rear_boxes,
            )

            # Change operator
            assert len(next_operators) == len(pred_rear_boxes)
            for idx in range(len(pred_rear_boxes)):
                operator = IUEN[next_operators[idx]]
                pred_rear_boxes[idx][0]["operator"] = BOX_OPS.index(operator)

            state_vector = [
                list(torch.unbind(state_vector[0])),
                list(torch.unbind(state_vector[1])),
            ]

            for idx, b_idx in enumerate(next_b_indices):
                # Append boxes
                list_b_indices = list(b_indices)
                ori_idx = list_b_indices.index(b_idx)
                pred_boxes[ori_idx] += pred_rear_boxes[idx]

                # Add loss
                for key in losses[ori_idx].keys():
                    losses[ori_idx][key] += rear_losses[idx][key]

            # Change State vector
            for idx, b_idx in enumerate(next_b_indices):
                state_vector[0][b_idx] = next_state_vector[0][idx]
                state_vector[1][b_idx] = next_state_vector[1][idx]

            state_vector = (torch.stack(state_vector[0]), torch.stack(state_vector[1]))

        # Add loss for op
        for idx in range(b_size):
            losses[idx]["operator"] = -op_selected_prob[idx]

        return state_vector, losses, pred_boxes

    def decode_boxes(self, b_indices, att, state_vector, prev_box, gold_boxes):
        b_size = len(b_indices)

        x = self.compute_lstm_input(b_indices, prev_box, att)
        state_vector, att = self.compute_context_vector(b_indices, x, state_vector)

        # Get scores for groupBy, orderBy
        groupBy_scores = self.is_group_by(att)
        orderBy_scores = self.is_order_by(att)

        # Change as probs
        groupBy_probs = torch.log_softmax(groupBy_scores, dim=-1)
        orderBy_probs = torch.log_softmax(orderBy_scores, dim=-1)

        # Get answer
        if self.training:
            group_selected_idx, order_selected_idx = get_is_box_info(gold_boxes)
            group_selected_idx = to_tensor(group_selected_idx)
            order_selected_idx = to_tensor(order_selected_idx)
            group_selected_prob = torch.gather(groupBy_probs, 1, group_selected_idx)
            order_selected_prob = torch.gather(orderBy_probs, 1, order_selected_idx)
        else:
            group_selected_prob, group_selected_idx = torch.topk(groupBy_probs, 1)
            order_selected_prob, order_selected_idx = torch.topk(orderBy_probs, 1)

        # Get b_indices for group by
        is_groupBy = group_selected_idx == 1
        is_orderBy = order_selected_idx == 1
        groupBy_indices = [idx for idx in range(b_size) if is_groupBy[idx]]
        groupBy_indices = (
            torch.tensor(groupBy_indices).cuda() if groupBy_indices else None
        )
        orderBy_indices = [idx for idx in range(b_size) if is_orderBy[idx]]
        orderBy_indices = (
            torch.tensor(orderBy_indices).cuda() if orderBy_indices else None
        )

        # Get gold boxes
        groupBy_gold_box = (
            get_box_with_op_type("groupBy", gold_boxes) if self.training else None
        )
        orderBy_gold_box = (
            get_box_with_op_type("orderBy", gold_boxes) if self.training else None
        )

        # Predict SELECT BOX
        select_gold_box = [box[0] for box in gold_boxes] if self.training else None
        select_loss, select_box = self.decode_box_type(
            "select", b_indices, att, select_gold_box
        )
        pred_boxes = [[box] for box in select_box]

        losses = select_loss

        # Predict GroupBy BOX
        if groupBy_indices is not None:
            groupBy_att = att[groupBy_indices]
            groupBy_prev_box = [
                select_box[idx] for idx in range(b_size) if is_groupBy[idx]
            ]
            groupBy_state_vector = [
                state_vector[0][groupBy_indices],
                state_vector[1][groupBy_indices],
            ]

            groupBy_b_indices = torch.tensor(
                [b_indices[idx] for idx in groupBy_indices]
            ).cuda()
            x = self.compute_lstm_input(
                groupBy_b_indices, groupBy_prev_box, groupBy_att
            )
            groupBy_state_vector, groupBy_att = self.compute_context_vector(
                groupBy_b_indices, x, groupBy_state_vector
            )

            groupBy_loss, groupBy_pred_box = self.decode_box_type(
                "groupBy", groupBy_b_indices, groupBy_att, groupBy_gold_box
            )

            # Unbind
            state_vector = [
                list(torch.unbind(state_vector[0])),
                list(torch.unbind(state_vector[1])),
            ]
            att = list(torch.unbind(att))

            for idx, ori_idx in enumerate(groupBy_indices):
                # Append box
                pred_boxes[ori_idx] += [groupBy_pred_box[idx]]

                # Add loss
                for key in losses[ori_idx].keys():
                    losses[ori_idx][key] += groupBy_loss[idx][key]

                # Save state vector
                state_vector[0][ori_idx] = groupBy_state_vector[0][idx]
                state_vector[1][ori_idx] = groupBy_state_vector[1][idx]

                att[ori_idx] = groupBy_att[idx]

            # Bind back
            state_vector = (torch.stack(state_vector[0]), torch.stack(state_vector[1]))
            att = torch.stack(att)

        # Predict OrderBy BOX
        if orderBy_indices is not None:
            orderBy_att = att[orderBy_indices]
            orderBy_prev_box = [
                pred_boxes[idx][-1] for idx in range(b_size) if is_orderBy[idx]
            ]
            orderBy_state_vector = [
                state_vector[0][orderBy_indices],
                state_vector[1][orderBy_indices],
            ]

            orderBy_b_indices = torch.tensor(
                [b_indices[idx] for idx in orderBy_indices]
            ).cuda()
            x = self.compute_lstm_input(
                orderBy_b_indices, orderBy_prev_box, orderBy_att
            )
            orderBy_state_vector, orderBy_att = self.compute_context_vector(
                orderBy_b_indices, x, orderBy_state_vector
            )

            orderBy_loss, orderBy_pred_box = self.decode_box_type(
                "orderBy", orderBy_b_indices, orderBy_att, orderBy_gold_box
            )

            # Unbind
            state_vector = [
                list(torch.unbind(state_vector[0])),
                list(torch.unbind(state_vector[1])),
            ]
            att = list(torch.unbind(att))

            for idx, ori_idx in enumerate(orderBy_indices):
                # Append box
                pred_boxes[ori_idx] += [orderBy_pred_box[idx]]
                # Add loss
                for key in losses[ori_idx].keys():
                    losses[ori_idx][key] += orderBy_loss[idx][key]

                # Save state vector
                state_vector[0][ori_idx] = orderBy_state_vector[0][idx]
                state_vector[1][ori_idx] = orderBy_state_vector[1][idx]

                att[ori_idx] = orderBy_att[idx]

            # Bind back
            state_vector = (torch.stack(state_vector[0]), torch.stack(state_vector[1]))
            att = torch.stack(att)

        # Add is_group_by, is_order_by
        for idx in range(b_size):
            losses[idx]["is_group_by"] = -group_selected_prob[idx]
            losses[idx]["is_order_by"] = -order_selected_prob[idx]

        return state_vector, losses, pred_boxes

    def decode_box_type(self, box_type, b_indices, att, gold_box):
        b_size = len(att)
        # Create column mask and table mask
        col_masks = self.col_mask[b_indices]
        tab_masks = self.tab_mask[b_indices]

        if box_type == "select":
            layers = self.select_layers
        elif box_type == "groupBy":
            layers = self.group_layers
        elif box_type == "orderBy":
            layers = self.order_layers
        else:
            raise RuntimeError("Should not be here")

        # Create Boxes
        pred_boxes = [
            {
                "head": [],
                "body": {
                    "local_predicates": [],
                    "quantifiers": [],
                    "quantifier_types": [],
                },
                "operator": BOX_OPS.index(box_type),
            }
            for _ in range(b_size)
        ]

        # Predict quantifier type f num
        qf_num_score = layers["qf_num"](att)

        # Apply mask to qf_num_scores
        tab_lens = [list(mask).count(0) for mask in tab_masks]
        max_len = qf_num_score.shape[1]
        qf_num_mask = torch.arange(max_len).expand(b_size, max_len) >= torch.tensor(
            tab_lens
        ).unsqueeze(1)
        qf_num_score = qf_num_score.masked_fill(qf_num_mask.cuda(), float("-inf"))

        # To prob
        qf_num_prob = torch.log_softmax(qf_num_score, dim=-1)

        if self.training:
            selected_qf_num_idx = to_tensor(get_quantifier_num_with_type("f", gold_box))
            selected_qf_num_prob = torch.gather(qf_num_prob, 1, selected_qf_num_idx)
        else:
            selected_qf_num_prob, selected_qf_num_idx = torch.topk(qf_num_prob, 1)

        # Predict from table
        qf_prob = self.get_attention_weights(
            self.encoded_tab[b_indices], att, tab_masks, is_log_softmax=True
        )
        if self.training:
            selected_qf_idx = get_quantifier_with_type("f", gold_box)
            selected_qf_prob = []
            for b_idx in range(b_size):
                tmp_qf_probs = torch.gather(
                    qf_prob[b_idx], 0, torch.tensor(selected_qf_idx[b_idx]).cuda()
                )
                selected_qf_prob += [sum(tmp_qf_probs)]
            selected_qf_prob = torch.stack(selected_qf_prob)
        else:
            selected_qf_prob = []
            selected_qf_idx = []
            for b_idx in range(b_size):
                tmp_qf_prob, tmp_qf_idx = torch.topk(
                    qf_prob[b_idx], int(selected_qf_num_idx[b_idx]) + 1
                )
                selected_qf_prob += [sum(tmp_qf_prob)]
                selected_qf_idx += [[int(item) for item in tmp_qf_idx]]
            selected_qf_prob = torch.stack(selected_qf_prob)

        # Add quantifiers to boxes
        for idx in range(b_size):
            pred_boxes[idx]["body"]["quantifiers"] = selected_qf_idx[idx]
            pred_boxes[idx]["body"]["quantifier_types"] = ["f"] * len(
                selected_qf_idx[idx]
            )

        # Create new column mask from prediction based on quantifiers
        stop = 1
        # use tab_col_dic

        if box_type != "order":
            # Predict local predicate Num
            p_num_score = layers["p_num"](att)
            p_num_prob = torch.log_softmax(p_num_score, dim=-1)
            if self.training:
                selected_p_num_idx = to_tensor(get_local_predicate_num(gold_box))
                selected_p_num_prob = torch.gather(p_num_prob, 1, selected_p_num_idx)
            else:
                selected_p_num_prob, selected_p_num_idx = torch.topk(p_num_prob, 1)

            selected_p_agg_indices = [[] for _ in range(b_size)]
            selected_p_col_indices = [[] for _ in range(b_size)]
            selected_p_op_indices = [[] for _ in range(b_size)]

            selected_p_agg_probs = [torch.tensor([0.0]).cuda() for _ in range(b_size)]
            selected_p_col_probs = [torch.tensor([0.0]).cuda() for _ in range(b_size)]
            selected_p_op_probs = [torch.tensor([0.0]).cuda() for _ in range(b_size)]

            prev_agg = prev_col = prev_op = torch.zeros(b_size, self.emb_dim).cuda()
            for n_idx in range(max(selected_p_num_idx)):
                next_b_indices = torch.tensor(
                    [
                        b_indices[idx]
                        for idx in range(b_size)
                        if selected_p_num_idx[idx] > n_idx
                    ]
                ).cuda()
                list_b_indices = list(b_indices)
                ori_indices = [
                    list_b_indices.index(idx) for idx in list(next_b_indices)
                ]

                # Create context
                new_context_vector = torch.cat(
                    [
                        att[ori_indices],
                        prev_agg[ori_indices],
                        prev_col[ori_indices],
                        prev_op[ori_indices],
                    ],
                    1,
                )
                p_ctx_vector = layers["p_layer"](new_context_vector)

                # Predict Agg
                p_agg_score = layers["p_agg"](p_ctx_vector)
                p_agg_prob = torch.log_softmax(p_agg_score, dim=-1)

                if self.training:
                    selected_p_agg_idx = to_tensor(
                        get_local_predicate_agg_with_idx(
                            n_idx, [gold_box[idx] for idx in ori_indices]
                        )
                    )
                    selected_p_agg_prob = torch.gather(
                        p_agg_prob, 1, selected_p_agg_idx
                    )
                else:
                    selected_p_agg_prob, selected_p_agg_idx = torch.topk(p_agg_prob, 1)

                # Predict Col
                p_col_prob = self.get_attention_weights(
                    self.encoded_col[next_b_indices],
                    p_ctx_vector,
                    col_masks[ori_indices],
                    is_log_softmax=True,
                )

                if self.training:
                    selected_p_col_idx = to_tensor(
                        get_local_predicate_col_with_idx(
                            n_idx, [gold_box[idx] for idx in ori_indices]
                        )
                    )
                    selected_p_col_prob = torch.gather(
                        p_col_prob, 1, selected_p_col_idx
                    )
                else:
                    selected_p_col_prob, selected_p_col_idx = torch.topk(p_col_prob, 1)

                # Predict operator
                p_op_score = layers["p_op"](p_ctx_vector)
                p_op_prob = torch.log_softmax(p_op_score, dim=-1)

                if self.training:
                    selected_p_op_idx = to_tensor(
                        get_local_predicate_op_with_idx(
                            n_idx, [gold_box[idx] for idx in ori_indices]
                        )
                    )
                    selected_p_op_prob = torch.gather(p_op_prob, 1, selected_p_op_idx)
                else:
                    selected_p_op_prob, selected_p_op_idx = torch.topk(p_op_prob, 1)

                # Save and prev
                prev_agg[ori_indices] = self.agg_emb(selected_p_agg_idx).squeeze(1)
                prev_op[ori_indices] = self.p_op_emb(selected_p_op_idx).squeeze(1)
                prev_col[ori_indices] = torch.stack(
                    [
                        self.encoded_col[b_idx][selected_p_col_idx][idx]
                        for idx, b_idx in enumerate(next_b_indices)
                    ]
                ).squeeze(1)

                # Add
                for idx, ori_idx in enumerate(ori_indices):
                    selected_p_agg_probs[ori_idx] += selected_p_agg_prob[idx]
                    selected_p_col_probs[ori_idx] += selected_p_col_prob[idx]
                    selected_p_op_probs[ori_idx] += selected_p_op_prob[idx]

                    # Append
                    selected_p_agg_indices[ori_idx] += [selected_p_agg_idx[idx]]
                    selected_p_col_indices[ori_idx] += [selected_p_col_idx[idx]]
                    selected_p_op_indices[ori_idx] += [selected_p_op_idx[idx]]

            # Add local predicates to boxes
            for idx in range(b_size):
                for sub_idx in range(selected_p_num_idx[idx]):
                    p = (
                        int(selected_p_agg_indices[idx][sub_idx]),
                        int(selected_p_col_indices[idx][sub_idx]),
                        int(selected_p_op_indices[idx][sub_idx]),
                        "",
                    )
                    pred_boxes[idx]["body"]["local_predicates"] += [p]

        # Predict Head num
        head_num_score = layers["head_num"](att)
        head_num_prob = torch.log_softmax(head_num_score, dim=-1)
        if self.training:
            selected_head_num_idx = to_tensor(get_head_num(gold_box))
            selected_head_num_prob = torch.gather(
                head_num_prob, 1, selected_head_num_idx
            )
        else:
            selected_head_num_prob, selected_head_num_idx = torch.topk(head_num_prob, 1)

        # Predict Head column
        column_prob = self.get_attention_weights(
            self.encoded_col[b_indices], att, col_masks, is_log_softmax=True
        )
        if self.training:
            selected_head_col_idx = get_head_col(gold_box)
            selected_head_col_prob = []
            for b_idx in range(b_size):
                tmp_col_prob = torch.gather(
                    column_prob[b_idx],
                    0,
                    torch.tensor(selected_head_col_idx[b_idx]).cuda(),
                )
                selected_head_col_prob += [sum(tmp_col_prob)]
            selected_head_col_prob = torch.stack(selected_head_col_prob)
        else:
            selected_head_col_prob = []
            selected_head_col_idx = []
            for b_idx in range(b_size):
                tmp_col_prob, tmp_col_idx = torch.topk(
                    column_prob[b_idx], int(selected_head_num_idx[b_idx]) + 1
                )
                selected_head_col_idx += [tmp_col_idx]
                selected_head_col_prob += [sum(tmp_col_prob)]
            selected_head_col_prob = torch.stack(selected_head_col_prob)

        selected_head_agg_indices = [[] for _ in range(b_size)]
        selected_head_agg_probs = [torch.tensor([0.0]).cuda() for _ in range(b_size)]

        # Predict Head Agg
        for n_idx in range(max(selected_head_num_idx) + 1):
            next_b_indices = torch.tensor(
                [
                    b_indices[idx]
                    for idx in range(b_size)
                    if selected_head_num_idx[idx] + 1 > n_idx
                ]
            ).cuda()
            list_b_indices = list(b_indices)
            ori_indices = [list_b_indices.index(idx) for idx in list(next_b_indices)]

            # Create context
            tmp_col_idx = torch.tensor(
                [
                    selected_head_col_idx[ori_indices[idx]][n_idx]
                    for idx in range(len(ori_indices))
                ]
            ).cuda()
            col_emb = torch.stack(
                [
                    self.encoded_col[ori_idx][tmp_col_idx[idx]]
                    for idx, ori_idx in enumerate(ori_indices)
                ]
            )
            new_context_vector = torch.cat([att[ori_indices], col_emb], 1)
            h_ctx_vector = layers["head_layer"](new_context_vector)

            # Predict Agg
            head_agg_score = layers["head_agg"](h_ctx_vector)
            head_agg_prob = torch.log_softmax(head_agg_score, dim=-1)

            if self.training:
                selected_head_agg_idx = to_tensor(
                    get_head_agg_with_idx(n_idx, [gold_box[idx] for idx in ori_indices])
                )
                selected_head_agg_prob = torch.gather(
                    head_agg_prob, 1, selected_head_agg_idx
                )
            else:
                selected_head_agg_prob, selected_head_agg_idx = torch.topk(
                    head_agg_prob, 1
                )

            for idx, ori_idx in enumerate(ori_indices):
                # Add loss
                selected_head_agg_probs[ori_idx] += selected_head_agg_prob[idx]

                # Append
                selected_head_agg_indices[ori_idx] += [selected_head_agg_idx[idx]]

        # Add Head to boxes
        for idx in range(b_size):
            for sub_idx in range(selected_head_num_idx[idx] + 1):
                pred_boxes[idx]["head"] += [
                    (
                        int(selected_head_agg_indices[idx][sub_idx]),
                        int(selected_head_col_idx[idx][sub_idx]),
                    )
                ]

        if box_type == "order":
            # predict is_asc
            is_asc_score = layers["is_asc"](att)
            is_asc_prob = torch.log_softmax(is_asc_score, dim=-1)
            if self.training:
                selected_is_asc_idx = to_tensor(get_is_asc(gold_box))
                selected_is_asc_prob = torch.gather(is_asc_prob, 1, selected_is_asc_idx)
            else:
                selected_is_asc_prob, selected_is_asc_idx = torch.topk(is_asc_prob, 1)

            # predict limit_num
            limit_num_score = layers["limit_num"](att)
            limit_num_prob = torch.log_softmax(limit_num_score, dim=-1)
            if self.training:
                selected_limit_num_idx = to_tensor(get_limit_num(gold_box))
                selected_limit_num_prob = torch.gather(
                    limit_num_prob, 1, selected_is_asc_idx
                )
            else:
                selected_limit_num_prob, selected_limit_num_idx = torch.topk(
                    limit_num_prob, 1
                )

            # Add auxiliary info to boxes
            for idx in range(b_size):
                pred_boxes[idx]["is_auxiliary"] = int(selected_is_asc_idx[idx]) == 1
                pred_boxes[idx]["limit_num"] = int(selected_limit_num_idx[idx])

        # Add loss
        losses = []
        for idx in range(b_size):
            loss = {
                "head_num": -selected_head_num_prob[idx],
                "head_agg": -selected_head_agg_probs[idx],
                "head_col": -selected_head_col_prob[idx],
                "quantifier_num": -selected_qf_num_prob[idx],
                "quantifier": -selected_qf_prob[idx],
                "predicate_num": -selected_p_num_prob[idx],
                "predicate_agg": -selected_p_agg_probs[idx],
                "predicate_col": -selected_p_col_probs[idx],
                "predicate_op": -selected_p_op_probs[idx],
                "is_auxiliary": -selected_is_asc_prob[idx]
                if box_type == "order"
                else torch.tensor(0),
                "limit_num": -selected_limit_num_prob[idx]
                if box_type == "order"
                else torch.tensor(0),
            }
            losses += [loss]

        return losses, pred_boxes

    def encode_box(self, b_indices, boxes):
        encoded_col = self.encoded_col[b_indices]
        encoded_tab = self.encoded_tab[b_indices]
        # Encode head
        b_embs = []
        for b_idx, box in enumerate(boxes):
            tmp = []
            for head in box["head"]:
                agg_id, col_id = head
                agg_emb = self.agg_emb(torch.tensor(agg_id).cuda()).squeeze(0)
                col_emb = encoded_col[b_idx][torch.tensor(col_id).cuda()]
                emb = agg_emb + col_emb
                tmp += [emb]
            head_emb = sum(tmp) / len(tmp)

            # Encode body (only quantifiers)
            tmp = []
            for table_id in box["body"]["quantifiers"]:
                tab_emb = encoded_tab[b_idx][table_id]
                tmp += [tab_emb]
            body_emb = sum(tmp) / len(tmp) if tmp else torch.zeros(self.emb_dim)

            # Encode operator
            box_op_emb = self.box_op_emb(torch.tensor(box["operator"]).cuda())

            # Combine and encode
            embs = torch.cat([head_emb, body_emb, box_op_emb], 0)

            b_embs += [embs]
        ctx_vector = self.box_encode_layer(torch.stack(b_embs))
        return ctx_vector

    def get_attention_weights(
        self, source, query, source_mask=None, is_log_softmax=False
    ):
        # Compute weight
        weight_scores = torch.bmm(source, query.unsqueeze(-1)).squeeze(-1)

        # Masking
        if source_mask is not None:
            weight_scores.data.masked_fill_(source_mask.bool(), -float("inf"))

        if is_log_softmax:
            weight_probs = torch.log_softmax(weight_scores, dim=-1)
        else:
            weight_probs = torch.softmax(weight_scores, dim=-1)

        return weight_probs

    def attention(self, encoded_src, encoded_src_attn, state_vector, src_mask=None):
        # Compute attention weights
        att_weight = self.get_attention_weights(
            encoded_src_attn, state_vector, src_mask
        )

        # Compute new context_vector
        ctx_vector = torch.bmm(att_weight.unsqueeze(1), encoded_src).squeeze(1)

        return ctx_vector

    def compute_context_vector(self, b_indices, x, state_vector):
        # Compute LSTM
        state_vector = self.lstm(x, state_vector)

        # Select
        encoded_src = self.encoded_src[b_indices]
        encoded_src_att = self.encoded_src_att[b_indices]
        src_mask = self.src_mask[b_indices]

        # Compute attention
        ctx_vector = self.attention(
            encoded_src, encoded_src_att, state_vector[0], src_mask
        )
        att = self.att_layer(torch.cat([state_vector[0], ctx_vector], 1))

        return state_vector, att

    def compute_lstm_input(self, b_indices, box, att):
        # Encode Box
        encoded_box = (
            self.encode_box(b_indices, box)
            if box
            else torch.zeros(len(att), self.emb_dim).cuda()
        )
        return torch.cat([encoded_box, att], 1)

    def get_accuracy(self, pred_qgms, gold_qgms):
        return compare_boxes(pred_qgms, gold_qgms)
