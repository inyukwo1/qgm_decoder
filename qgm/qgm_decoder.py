import torch
import torch.nn as nn
from qgm.utils import WHERE_OPS, BOX_OPS, AGG_OPS, IUEN, IUE, IUE_INDICES, GROUP_IDX, ORDER_IDX
from qgm.utils import get_iue_box_op, split_boxes, check_box_ops, get_quantifier_num_with_type, \
                      get_quantifier_with_type, get_local_predicate_num, get_local_predicate_agg_with_idx, \
                      get_local_predicate_col_with_idx, get_local_predicate_op_with_idx, to_tensor, \
                      get_head_num, get_head_col, get_head_agg_with_idx, get_limit_num, get_is_asc


class QGM_Decoder(nn.Module):
    def __init__(self):
        super(QGM_Decoder, self).__init__()
        emb_dim = 300
        att_vec_dim = emb_dim
        input_dim = emb_dim + att_vec_dim + emb_dim #?

        # LSTM
        self.lstm = nn.LSTMCell(input_dim, emb_dim)
        self.att_layer = nn.Sequential(
            torch.nn.Linear(emb_dim*2, att_vec_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        self.nested_layer = nn.Linear(emb_dim, emb_dim)

        # Pointer Network
        self.src_encoding_layer = nn.Linear(emb_dim, emb_dim, bias=False)

        # Box Encode
        self.box_encode_layer = nn.Sequential(
            torch.nn.Linear(emb_dim * 3,emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        # Necessary
        self.iuen_linear = nn.Linear(att_vec_dim, 4)
        self.op_emb = nn.Embedding(len(BOX_OPS), emb_dim)
        self.agg_emb = nn.Embedding(len(AGG_OPS), emb_dim)
        #
        self.is_group_by = nn.Linear(att_vec_dim, 2)
        self.is_order_by = nn.Linear(att_vec_dim, 2)

        # Select
        self.select_qf_num = nn.Linear(att_vec_dim, 5)
        self.select_qs_num = nn.Linear(att_vec_dim, 4)
        self.select_p_num = nn.Linear(att_vec_dim, 5)
        self.select_p_layer = nn.Linear(att_vec_dim + emb_dim*3, emb_dim)
        self.select_p_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        self.select_p_op = nn.Linear(att_vec_dim, len(WHERE_OPS))
        self.select_head_layer = nn.Linear(att_vec_dim + emb_dim, att_vec_dim)
        self.select_head_num = nn.Linear(att_vec_dim, 6)
        self.select_head_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        # Group by
        self.group_by_qf_num = nn.Linear(att_vec_dim, 4)
        self.group_by_qs_num = nn.Linear(att_vec_dim, 3)
        self.group_by_p_num = nn.Linear(att_vec_dim, 5)
        self.group_by_p_layer = nn.Linear(emb_dim + emb_dim, att_vec_dim)
        self.group_by_p_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        self.group_by_p_op = nn.Linear(att_vec_dim, len(WHERE_OPS))
        self.group_by_head_layer = nn.Linear(att_vec_dim + emb_dim, att_vec_dim)
        self.group_by_head_num = nn.Linear(att_vec_dim, 3)
        self.group_by_head_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        # Order by
        self.order_by_q_num = nn.Linear(att_vec_dim, 3)
        self.order_by_head_layer = nn.Linear(att_vec_dim + emb_dim, att_vec_dim)
        self.order_by_head_num = nn.Linear(att_vec_dim, 3)
        self.order_by_head_agg = nn.Linear(att_vec_dim, len(AGG_OPS))
        self.order_by_is_asc = nn.Linear(att_vec_dim, 2)
        self.order_by_limit_num = nn.Linear(att_vec_dim, 3)
        self.order_by_agg = nn.Linear(att_vec_dim, len(AGG_OPS))

        # Store variables
        self.encoded_src = None
        self.encoded_src_att = None
        self.encoded_col = None
        self.encoded_tab = None


    def set_variables(self, encoded_src, encoded_src_att, encoded_col, encoded_tab, src_mask):
        self.encoded_src = encoded_src
        self.encoded_src_att = encoded_src_att
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab
        self.src_mask = src_mask


    def decode(self, b_indices, att, state_vector, qgm=None):
        '''
        Assumption:
            1. no nested query (yet)
            2. only local predicates for the where clause and having clause (yet)
        '''
        b_size = len(qgm)

        x = self.compute_context_vector(b_indices, None, att) if att else torch.zeros(b_size, 900).cuda()
        state_vector, att = self.compute_context_vector(b_indices, x, state_vector)

        iuen_score = self.iuen_linear(att)
        iuen_prob = torch.log_softmax(iuen_score, dim=-1)

        # Choose box_operator
        if self.training:
            op_keys = get_iue_box_op(qgm)
            selected_idx = to_tensor([IUEN.index(op_key) if op_key
                                         else IUEN.index('select') for op_key in op_keys])
            selected_prob = torch.gather(iuen_prob, 1, selected_idx)
        else:
            selected_prob, selected_idx = torch.topk(iuen_prob, 1)

        # Split boxes
        if self.training:
            gold_front_boxes, gold_rear_boxes = split_boxes(qgm)
        else:
            gold_front_boxes = gold_rear_boxes = None

        # Calculate state vector
        pred_front_boxes, loss, state_vector = self.decode_box(b_indices, att, state_vector, prev_box=None, gold_boxes=gold_front_boxes)
        pred_boxes = [pred_front_boxes]

        b_indices = [idx for idx in range(b_size) if selected_idx[idx] != 0]
        if b_indices:
            # Calculate state vector
            pred_rear_boxes, loss, state_vector = self.decode_box(b_indices, att, state_vector, prev_box=pred_boxes[b_indices][-1], gold_boxes=gold_rear_boxes)
            pred_boxes += [pred_rear_boxes]


        return pred_boxes #loss, pred


    def decode_box(self, b_indices, att, state_vector, prev_box, gold_boxes):
        '''To-Do: Need to split self.is_group_by for front and rear??'''
        b_size = len(b_indices)

        x = self.compute_lstm_input(prev_box, att)
        state_vector, att = self.compute_context_vector(b_indices, x, state_vector)

        # Get scores for groupBy, orderBy
        groupBy_scores = self.is_group_by(att)
        orderBy_scores = self.is_order_by(att)

        # Change as probs
        groupBy_probs = torch.log_softmax(groupBy_scores, dim=-1)
        orderBy_probs = torch.log_softmax(orderBy_scores, dim=-1)

        # Get answer
        if self.training:
            group_selected_idx, order_selected_idx = check_box_ops(gold_boxes)
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
        groupBy_b_indices = [idx for idx in range(b_size) if is_groupBy[idx]]
        orderBy_b_indices = [idx for idx in range(b_size) if is_orderBy[idx]]

        # Predict SELECT BOX
        select_gold_box = [box[0] for box in gold_boxes] if self.training else None
        select_loss, select_box = self.decode_select_box(att, state_vector, select_gold_box)
        boxes = [select_box]

        if groupBy_b_indices:
            #Predict GROUPBY BOX
            groupBy_prev_box  = prev_box[groupBy_b_indices]
            groupBy_att = att[groupBy_b_indices]
            groupBy_state_vector = state_vector[groupBy_b_indices]

            x = self.compute_lstm_input(groupBy_prev_box, groupBy_att)
            groupBy_state_vector, att = self.compute_context_vector(x, groupBy_state_vector)

            groupBy_loss, group_by_box = self.decode_groupBy_box(groupBy_state_vector)
            boxes += [group_by_box]

            # Save state vector
            state_vector[groupBy_b_indices] = groupBy_state_vector

        if orderBy_b_indices:
            #Predict ORDERBY BOX
            orderBy_prev_box  = prev_box[orderBy_b_indices]
            orderBy_att = att[orderBy_b_indices]
            orderBy_state_vector = state_vector[orderBy_b_indices]

            x = self.compute_lstm_input(orderBy_prev_box, orderBy_att)
            state_vector, att = self.compute_context_vector(x, orderBy_state_vector)

            orderBy_loss, order_by_box = self.decode_orderBy_box(state_vector)
            boxes += [order_by_box]

            # Save state vector
            state_vector[orderBy_b_indices] = orderBy_state_vector

        # Combine loss
        total_loss = None

        return boxes, state_vector, total_loss


    def decode_select_box(self, att, state_vector, gold_box):
        b_size = len(att)

        # Predict quantifier type f num
        qf_num_score = self.select_qf_num(att)
        qf_num_prob = torch.log_softmax(qf_num_score, dim=-1)
        if self.training:
            selected_qf_num_idx = to_tensor(get_quantifier_num_with_type('f', gold_box))
            selected_qf_num_prob = torch.gather(qf_num_prob, 1, selected_qf_num_idx)
        else:
            selected_qf_num_prob, selected_qf_num_idx = torch.topk(qf_num_prob, 1)
        '''
        # Predict quantifier type s num
        qs_num_score = self.select_qs_num(att)
        qs_num_prob = torch.log_softmax(qs_num_score, dim=-1)
        if self.training:
            selected_qs_num_idx = to_tensor(get_quantifier_num_with_type('s', gold_box))
            selected_qs_num_prob = torch.gather(qs_num_prob, 1, selected_qs_num_idx)
        else:
            selected_qs_num_prob, selected_qs_num_idx = torch.topk(qs_num_prob, 1)
        '''

        # Predict from table
        table_score = self.get_attention_weights(self.encoded_tab, att)
        table_prob = torch.log_softmax(table_score, dim=-1)
        if self.training:
            selected_table_idx = get_quantifier_with_type('f', gold_box)
            selected_table_prob = []
            for b_idx in range(b_size):
                probs = torch.gather(table_prob[b_idx], 0, torch.tensor(selected_table_idx[b_idx]).cuda())
                selected_table_prob += [sum(probs)]
            selected_table_prob = torch.stack(selected_table_prob)
        else:
            selected_table_prob = []
            selected_table_idx = []
            for b_idx in range(b_size):
                tmp_table_prob, tmp_table_idx = torch.topk(table_prob[b_idx], selected_qf_num_idx[b_idx])
                selected_table_prob += [sum(tmp_table_prob)]
                selected_table_idx += [selected_table_idx]
            selected_table_prob = torch.stack(selected_table_prob)

        '''
        # Predict nested quantifier
        if selected_qs_num_idx:
            if self.training:
                gold_sub_box = [item for idx, item in enumerate(gold_box['body']['quantifiers']) if
                                gold_box['body']['quantifier_types'][idx] == 's']
            else:
                gold_sub_box = None
            pred_box, context_vector = self.decode(att, state_vector, gold_sub_box)

        '''
        # Predict local predicate Num
        p_num_score = self.select_p_num(att)
        p_num_prob = torch.log_softmax(p_num_score, dim=-1)
        if self.training:
            selected_p_num_idx = torch.tensor(get_local_predicate_num(gold_box)).unsqueeze(-1).cuda()
            selected_p_num_prob = torch.gather(p_num_prob, 1, selected_p_num_idx)
        else:
            selected_p_num_prob, selected_p_num_idx = torch.topk(p_num_prob, 1)

        prev_agg = prev_col = prev_op = torch.zeros(b_size, 300).cuda()
        for n_idx in range(max(selected_p_num_idx)):
            b_indices = torch.tensor([b_idx for b_idx in range(b_size) if selected_p_num_idx[b_idx] > n_idx]).cuda()

            # Create context
            new_context_vector = torch.cat([att[b_indices], prev_agg[b_indices], prev_col[b_indices], prev_op[b_indices]], 1)
            p_ctx_vector = self.select_p_layer(new_context_vector)

            # Predict Agg
            p_agg_score = self.select_p_agg(p_ctx_vector)
            p_agg_prob = torch.log_softmax(p_agg_score, dim=-1)

            if self.training:
                selected_p_agg_idx = to_tensor(get_local_predicate_agg_with_idx(n_idx, [gold_box[idx] for idx in b_indices]))
                selected_p_agg_prob = torch.gather(p_agg_prob, 1, selected_p_agg_idx)
            else:
                selected_p_agg_prob, selected_p_agg_idx = torch.topk(p_agg_prob, 1)

            # Predict Col
            p_col_score = self.get_attention_weights(self.encoded_col[b_indices], p_ctx_vector)
            p_col_prob = torch.log_softmax(p_col_score, dim=-1)

            if self.training:
                selected_p_col_idx = to_tensor(get_local_predicate_col_with_idx(n_idx, [gold_box[idx] for idx in b_indices]))
                selected_p_col_prob = torch.gather(p_col_prob, 1, selected_p_col_idx)
            else:
                selected_p_col_prob, selected_p_col_idx = torch.topk(p_col_prob, 1)

            # Predict operator
            p_op_score = self.select_p_op(p_ctx_vector)
            p_op_prob = torch.log_softmax(p_op_score, dim=-1)

            if self.training:
                selected_p_op_idx = to_tensor(get_local_predicate_op_with_idx(n_idx, [gold_box[idx] for idx in b_indices]))
                selected_p_op_prob = torch.gather(p_op_prob, 1, selected_p_op_idx)
            else:
                selected_p_op_prob, selected_p_op_idx = torch.topk(p_op_prob, 1)

            # Save and prev
            prev_agg[b_indices] = self.agg_emb(selected_p_agg_idx).squeeze(1)
            prev_op[b_indices] = self.op_emb(selected_p_op_idx).squeeze(1)
            prev_col[b_indices] = torch.stack([self.encoded_col[b_idx][selected_p_col_idx[idx]] for idx, b_idx in enumerate(b_indices)]).squeeze(1)

        # Predict Head num
        head_num_score = self.select_head_num(att)
        head_num_prob = torch.log_softmax(head_num_score, dim=-1)
        if self.training:
            selected_head_num_idx = to_tensor(get_head_num(gold_box))
            selected_head_num_prob = torch.gather(head_num_prob, 1, selected_head_num_idx)
        else:
            selected_head_num_prob, selected_head_num_idx = torch.topk(head_num_prob, 1)

        # Predict Head column
        column_score = self.get_attention_weights(self.encoded_col, att)
        column_prob = torch.log_softmax(column_score, dim=-1)
        # Mask out those not in quantifiers
        if self.training:
            selected_head_col_idx = get_head_col(gold_box)
            selected_head_col_prob = []
            for b_idx in range(b_size):
                tmp_col_prob = torch.gather(column_prob[b_idx], 0, torch.tensor(selected_head_col_idx[b_idx]).cuda())
                selected_head_col_prob += [sum(tmp_col_prob)]
            selected_head_col_prob = torch.stack(selected_head_col_prob)
        else:
            selected_head_col_prob = []
            selected_head_col_idx = []
            for b_idx in range(b_size):
                tmp_col_prob, tmp_col_idx = torch.topk(column_prob[b_idx], selected_head_num_idx[b_idx]+1)
                selected_head_col_idx += [tmp_col_idx]
                selected_head_col_prob += [sum(tmp_col_prob)]
            selected_head_col_prob = torch.stack(selected_head_col_prob)

        # Predict Head Agg
        for n_idx in range(max(selected_head_num_idx)):
            b_indices = torch.tensor([b_idx for b_idx in range(b_size) if selected_head_num_idx[b_idx]+1 > n_idx]).cuda()

            # Create context
            tmp_col_idx = torch.tensor([selected_head_col_idx[b_indices[idx]][n_idx] for idx in range(len(b_indices))]).cuda()
            col_emb = torch.stack([self.encoded_col[b_idx][tmp_col_idx[idx]] for idx, b_idx in enumerate(b_indices)])
            new_context_vector = torch.cat([att[b_indices], col_emb], 1)
            h_ctx_vector = self.select_head_layer(new_context_vector)

            # Predict Agg
            head_agg_score = self.select_head_agg(h_ctx_vector)
            head_agg_prob = torch.log_softmax(head_agg_score, dim=-1)

            if self.training:
                selected_head_agg_idx = to_tensor(get_head_agg_with_idx(n_idx, gold_box))
                selected_head_agg_prob = torch.gather(head_agg_prob, 1, selected_head_agg_idx)
            else:
                selected_head_agg_prob, selected_head_agg_idx = torch.topk(head_agg_prob, 1)

        return loss, pred_boxes


    def decode_groupBy_box(self, att, state_vector, gold_box):
        b_size= len(att)

        # Predict quantifier type f num
        qf_num_score = self.group_by_qf_num(att)
        qf_num_prob = torch.log_softmax(qf_num_score, dim=-1)
        if self.training:
            selected_qf_num_idx = to_tensor(get_quantifier_num_with_type('f', gold_box))
            selected_qf_num_prob = torch.gather(qf_num_prob, 1, selected_qf_num_idx)
        else:
            selected_qf_num_prob, selected_qf_num_idx = torch.topk(qf_num_prob, 1)
        '''
        # Predict quantifier type s num
        qs_num_score = self.group_by_qs_num(att)
        qs_num_prob = torch.log_softmax(qs_num_score, dim=-1)
        if self.training:
            selected_qs_num_idx = to_tensor(get_quantifier_num_with_type('s', gold_box))
            selected_qs_num_prob = torch.gather(qs_num_prob, 1 ,selected_qs_num_idx)
        else:
            selected_qs_num_prob, selected_qs_num_idx = torch.topk(qs_num_prob, 1)
        '''
        # Predict from table
        table_score = self.get_attention_weights(self.encoded_tab, att)
        table_prob = torch.log_softmax(table_score, dim=-1)
        if self.training:
            selected_table_idx = get_quantifier_with_type('f', gold_box)
            selected_table_prob = []
            for b_idx in range(b_size):
                probs = torch.gather(table_prob[b_idx], 0, torch.tensor(selected_table_idx[b_idx]).cuda())
                selected_table_prob += [sum(probs)]
            selected_table_prob = torch.stack(selected_table_prob)
        else:
            selected_table_prob = []
            selected_table_idx = []
            for b_idx in range(b_size):
                tmp_table_prob, tmp_table_idx = torch.topk(table_prob[b_idx], selected_qf_num_idx[b_idx])
                selected_table_prob += [sum(tmp_table_prob)]
                selected_table_idx += [selected_table_idx]
            selected_table_prob = torch.stack(selected_table_prob)

        '''
        # Predict nested quantifier
        if selected_qs_num_idx:
            if self.training:
                gold_sub_box = [item for idx, item in enumerate(gold_box['body']['quantifiers']) if
                                gold_box['body']['quantifier_types'][idx] == 's']
            else:
                gold_sub_box = None
            pred_box, context_vector = self.decode(att, state_vector, gold_sub_box)
        '''

        # Predict local predicates num
        p_num_score = self.group_by_p_num(att)
        p_num_prob = torch.log_softmax(p_num_score, dim=-1)
        if self.training:
            selected_p_num_idx = torch.tensor(get_local_predicate_num(gold_box)).unsqueeze(-1).cuda()
            selected_p_num_prob = torch.gather(p_num_prob, 1, selected_p_num_idx)
        else:
            selected_p_num_prob, selected_p_num_idx = torch.topk(p_num_prob, 1)

        prev_agg = prev_col = prev_op = torch.zeros(b_size, 300).cuda()
        for n_idx in range(max(selected_p_num_idx)):
            b_indices = torch.tensor([b_idx for b_idx in range(b_size) if selected_p_num_idx[b_idx] > n_idx]).cuda()

            # Create context
            new_context_vector = torch.cat([att[b_indices], prev_agg[b_indices], prev_col[b_indices], prev_op[b_indices]], 1)
            p_ctx_vector = self.group_by_p_layer(new_context_vector)

            # Predict Agg
            p_agg_score = self.group_by_p_agg(p_ctx_vector)
            p_agg_prob = torch.log_softmax(p_agg_score, dim=-1)

            if self.training:
                selected_p_agg_idx = to_tensor(get_local_predicate_agg_with_idx(n_idx, [gold_box[idx] for idx in b_indices]))
                selected_p_agg_prob = torch.gather(p_agg_prob, 1, selected_p_agg_idx)
            else:
                selected_p_agg_prob, selected_p_agg_idx = torch.topk(p_agg_prob, 1)

            # Predict Col
            p_col_score = self.get_attention_weights(self.encoded_col[b_indices], p_ctx_vector)
            p_col_prob = torch.log_softmax(p_col_score, dim=-1)

            if self.training:
                selected_p_col_idx = to_tensor(get_local_predicate_col_with_idx(n_idx, [gold_box[idx] for idx in b_indices]))
                selected_p_col_prob = torch.gather(p_col_prob, 1, selected_p_col_idx)
            else:
                selected_p_col_prob, selected_p_col_idx = torch.topk(p_col_prob, 1)

            # Predict operator
            p_op_score = self.group_by_p_op(p_ctx_vector)
            p_op_prob = torch.log_softmax(p_op_score, dim=-1)

            if self.training:
                selected_p_op_idx = to_tensor(get_local_predicate_op_with_idx(n_idx, [gold_box[idx] for idx in b_indices]))
                selected_p_op_prob = torch.gather(p_op_prob, 1, selected_p_op_idx)
            else:
                selected_p_op_prob, group_by_p_op_idx = torch.topk(p_op_prob, 1)

            # Save and prev
            prev_agg[b_indices] = self.agg_emb(selected_p_agg_idx).squeeze(1)
            prev_op[b_indices] = self.op_emb(selected_p_op_idx).squeeze(1)
            prev_col[b_indices] = torch.stack([self.encoded_col[b_idx][selected_p_col_idx[idx]] for idx, b_idx in enumerate(b_indices)]).squeeze(1)

        # Predict Head Num
        head_num_score = self.group_by_head_num(att)
        head_num_prob = torch.log_softmax(head_num_score)
        if self.training:
            selected_head_num_idx = to_tensor(get_head_num(gold_box))
            selected_head_num_prob = torch.gather(head_num_prob, 1, selected_head_num_idx)
        else:
            selected_head_num_prob, selected_head_num_idx = torch.topk(head_num_prob, 1)

        # Predict Head column
        column_score = self.get_attention_weights(self.encoded_col, att)
        column_prob = torch.log_softmax(column_score, dim=-1)
        # Mask out those not in quantifiers
        if self.training:
            selected_head_col_idx = get_head_col(gold_box)
            selected_head_col_prob = []
            for b_idx in range(b_size):
                tmp_col_prob = torch.gather(column_prob[b_idx], 0, torch.tensor(selected_head_col_idx[b_idx]).cuda())
                selected_head_col_prob += [sum(tmp_col_prob)]
            selected_head_col_prob = torch.stack(selected_head_col_prob)
        else:
            selected_head_col_prob = []
            selected_head_col_idx = []
            for b_idx in range(b_size):
                tmp_col_prob, tmp_col_idx = torch.topk(column_prob[b_idx], selected_head_num_idx[b_idx]+1)
                selected_head_col_idx += [tmp_col_idx]
                selected_head_col_prob += [sum(tmp_col_prob)]
            selected_head_col_prob = torch.stack(selected_head_col_prob)

        # Predict Head Agg
        tmp = []
        for n_idx in range(selected_head_num_idx):
            b_indices = torch.tensor([b_idx for b_idx in range(b_size) if selected_head_num_idx[b_idx] + 1 > n_idx]).cuda()

            # Create Context
            tmp_col_idx = torch.tensor([selected_head_col_idx[b_indices[idx]][n_idx] for idx in range(len(b_indices))]).cuda()
            col_emb = torch.stack([self.encoded_col[b_idx][tmp_col_idx[idx]] for idx, b_idx in enumerate(b_indices)])
            new_context_vector = torch.cat([att[b_indices], col_emb], 1)
            h_ctx_vector = self.group_by_head_layer(new_context_vector)

            # Predict Agg
            head_agg_score = self.group_by_head_agg(h_ctx_vector)
            head_agg_prob = torch.log_softmax(head_agg_score, dim=-1)

            if self.training:
                selected_head_agg_idx = to_tensor(get_head_agg_with_idx(n_idx, gold_box))
                selected_head_agg_prob = torch.gather(head_agg_prob, 1, selected_head_agg_idx)
            else:
                selected_head_agg_prob, selected_head_agg_idx = torch.topk(head_agg_prob, 1)

        return loss, pred_boxes


    def decode_orderBy_box(self, att, state_vector, gold_box):
        b_size = len(att)

        # Predict quantifier type f num
        qf_num_score = self.order_by_qf_num(att)
        qf_num_prob = torch.log_softmax(qf_num_score, dim=-1)
        if self.training:
            selected_qf_num_idx = to_tensor(get_quantifier_num_with_type('f', gold_box))
            selected_qf_num_prob = torch.gather(qf_num_prob, 1, selected_qf_num_idx)
        else:
            selected_qf_num_prob, selected_qf_num_idx = torch.topk(qf_num_prob, 1)
        '''
        # Predict quantifier type s num
        qs_num_score = self.order_by_qs_num(att)
        qs_num_prob = torch.log_softmax(qs_num_score, dim=-1)
        if self.training:
            selected_qs_num_idx = to_tensor(get_quantifier_num_with_type('s', gold_box))
            selected_qs_num_prob = torch.gather(qs_num_prob, 1, selected_qs_num_idx)
        else:
            selected_qs_num_prob, selected_qs_num_idx = torch.topk(qs_num_prob, 1)
        '''
        # Predict from table
        table_score = self.get_attention_weights(self.encoded_tab, att)
        table_prob = torch.log_softmax(table_score, dim=-1)
        if self.training:
            selected_table_idx = get_quantifier_with_type('f', gold_box)
            selected_table_prob = []
            for b_idx in range(b_size):
                probs = torch.gather(table_prob[b_idx], 0, torch.tensor(selected_table_idx[b_idx]).cuda())
                selected_table_prob += [sum(probs)]
            selected_table_prob = torch.stack(selected_table_prob)
        else:
            selected_table_prob = []
            selected_table_idx = []
            for b_idx in range(b_size):
                tmp_table_prob, tmp_table_idx = torch.topk(table_prob[b_idx], selected_qf_num_idx[b_idx])
                selected_table_prob += [sum(tmp_table_prob)]
                selected_table_idx += [selected_table_idx]
            selected_table_prob = torch.stack(selected_table_prob)

        # Predict Head num
        head_num_score = self.order_by_head_num(att)
        head_num_prob = torch.log_softmax(head_num_score, dim=-1)
        if self.training:
            selected_head_num_idx = to_tensor(get_head_num(gold_box))
            selected_head_num_prob = torch.gather(head_num_prob, 1, selected_head_num_idx)
        else:
            selected_head_num_prob, selected_head_num_idx = torch.topk(head_num_prob, 1)

        # Predict Head column
        column_score = self.get_attention_weights(self.encoded_col, att)
        column_prob = torch.log_softmax(column_score, dim=-1)
        # Mask out those not in quantifiers
        if self.training:
            selected_head_col_idx = get_head_col(gold_box)
            selected_head_col_prob = []
            for b_idx in range(b_size):
                tmp_col_prob = torch.gather(column_prob[b_idx], 0, torch.tensor(selected_head_col_idx[b_idx]).cuda())
                selected_head_col_prob += [sum(tmp_col_prob)]
            selected_head_col_prob = torch.stack(selected_head_col_prob)
        else:
            selected_head_col_prob = []
            selected_head_col_idx = []
            for b_idx in range(b_size):
                tmp_col_prob, tmp_col_idx = torch.topk(column_prob[b_idx], selected_head_num_idx[b_idx]+1)
                selected_head_col_idx += [tmp_col_idx]
                selected_head_col_prob += [sum(tmp_col_prob)]
            selected_head_col_prob = torch.stack(selected_head_col_prob)

        # Predict Head Agg
        for n_idx in range(max(selected_head_num_idx)):
            b_indices = torch.tensor([b_idx for b_idx in range(b_size) if selected_head_num_idx[b_idx]+1 > n_idx]).cuda()

            # Create context
            tmp_col_idx = torch.tensor([selected_head_col_idx[b_indices[idx]][n_idx] for idx in range(len(b_indices))]).cuda()
            col_emb = torch.stack([self.encoded_col[b_idx][tmp_col_idx[idx]] for idx, b_idx in enumerate(b_indices)])
            new_context_vector = torch.cat([att[b_indices], col_emb], 1)
            h_ctx_vector = self.order_by_head_layer(new_context_vector)

            # Predict Agg
            head_agg_score = self.order_by_head_agg(h_ctx_vector)
            head_agg_prob = torch.log_softmax(head_agg_score, dim=-1)

            if self.training:
                selected_head_agg_idx = to_tensor(get_head_agg_with_idx(n_idx, gold_box))
                selected_head_agg_prob = torch.gather(head_agg_prob, 1, selected_head_agg_idx)
            else:
                selected_head_agg_prob, selected_head_agg_idx = torch.topk(head_agg_prob, 1)

        # predict is_asc
        is_asc_score = self.order_by_is_asc(att)
        is_asc_prob = torch.log_softmax(is_asc_score, dim=-1)
        if self.training:
            selected_is_asc_idx = to_tensor(get_is_asc(gold_box))
            selected_is_asc_prob = torch.gather(is_asc_prob, 1, selected_is_asc_idx)
        else:
            selected_is_asc_prob, selected_is_asc_idx = torch.topk(is_asc_prob, 1)

        # predict limit_num
        limit_num_score = self.order_by_limit_num(att)
        limit_num_prob = torch.log_softmax(limit_num_score, dim=-1)
        if self.training:
            selected_limit_num_idx = to_tensor(get_limit_num(gold_box))
            selected_limit_num_prob = torch.gather(limit_num_prob, 1, selected_is_asc_idx)
        else:
            selected_limit_num_prob, selected_limit_num_idx = torch.topk(limit_num_prob, 1)

        return loss, pred_boxes


    def encode_box(self, box, dropout):
        # Encode head
        tmp = []
        for head in box['head']:
            agg_id, col_id = head
            agg_emb = self.agg_emb(agg_id)
            col_emb = self.encoded_col[col_id]
            emb = agg_emb + col_emb
            tmp += [emb]
        head_emb = sum(tmp) / len(tmp)

        # Encode body (only quantifiers)
        tmp = []
        for table_id in box['quantifiers']:
            tab_emb = self.encoded_table[table_id]
            tmp += [tab_emb]
        body_emb = sum(tmp) / len(tmp)

        # Encode operator
        op_emb = self.op_emb(box['operator'])

        # Combine and encode
        ctx_vector = torch.cat([head_emb, body_emb, op_emb], 1)
        ctx_vector = self.box_encode_layer(ctx_vector)
        return ctx_vector


    def get_attention_weights(self, source, query, source_mask=None):
        # Compute weight
        weight_scores = torch.bmm(source, query.unsqueeze(-1)).squeeze(-1)

        # Masking
        if source_mask is not None:
            weight_scores.data.masked_fill_(source_mask, -float('inf'))

        # Softmax
        weight_probs = torch.softmax(weight_scores, dim=-1)

        return weight_probs


    def attention(self, encoded_src, encoded_src_attn, state_vector, mask=None):
        # Compute attention weights
        att_weight = self.get_attention_weights(encoded_src_attn, state_vector, mask)

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
        ctx_vector = self.attention(encoded_src, encoded_src_att, state_vector[0], src_mask)
        att = self.att_layer(torch.cat([state_vector[0], ctx_vector], 1))

        return state_vector, att


    def compute_lstm_input(self, box, att):
        # Encode Box
        encoded_box = self.encode_box(box) if box else torch.zeros(len(att), 600).cuda()
        return torch.cat([encoded_box, att], 1)
