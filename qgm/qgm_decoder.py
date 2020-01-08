import torch
import torch.nn as nn
import torch.nn.functional as F
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
BOX_OPS = ['select', 'groupBy', 'having', 'orderBy', 'intersect', 'union', 'except']
AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']
IUEN = ['select', 'intersect', 'union', 'except']
IUE = ['intersect', 'union', 'except']
IUE_INDICES = [BOX_OPS.index(key) for key in IUE]
GROUP_IDX = BOX_OPS.index('groupBy')
ORDER_IDX = BOX_OPS.index('orderBy')

class QGM_Decoder(nn.Module):
    def __init__(self):
        super(QGM_Decoder, self).__init__()
        emb_dim = 300

        # LSTM
        input_dim = 0 + 0 + 0
        self.lstm = nn.LSTMCell(input_dim, emb_dim)
        self.att_layer = nn.Sequential(
            torch.nn.Linear(emb_dim+emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        # Pointer Network
        self.src_encoding_layer = nn.Linear(emb_dim, emb_dim, bias=False)

        # Necessary
        self.iuen_linear = nn.Linear(emb_dim, 4)
        self.op_emb = nn.Embedding(len(BOX_OPS), emb_dim)
        self.agg_emb = nn.Embedding(len(AGG_OPS), emb_dim)
        #
        self.front_is_group_by = nn.Linear(emb_dim, 2)
        self.front_is_order_by = nn.Linear(emb_dim, 2)
        #
        self.back_is_group_by = nn.Linear(emb_dim, 2)
        self.back_is_order_by = nn.Linear(emb_dim, 2)

        # Select
        self.select_qf_num = nn.Linear(emb_dim, 5)
        self.select_qs_num = nn.Linear(emb_dim, 4)
        self.select_p_num = nn.Linear(emb_dim, 5)
        self.select_p_agg = nn.Linear(emb_dim, len(AGG_OPS))
        self.select_p_op = nn.Linear(emb_dim, len(WHERE_OPS))
        self.select_head_num = nn.Linear(emb_dim, 6)
        self.select_head_agg = nn.Linear(emb_dim, len(AGG_OPS))
        # Group by
        self.group_by_qf_num = nn.Linear(emb_dim, 4)
        self.group_by_qs_num = nn.Linear(emb_dim, 3)
        self.group_by_head_num = nn.Linear(emb_dim, 3)
        self.group_by_head_agg = nn.Linear(emb_dim, len(AGG_OPS))
        # Having
        self.group_by_p_num = nn.Linear(emb_dim, 5)
        self.group_by_p_agg = nn.Linear(emb_dim, len(AGG_OPS))
        self.group_by_p_op = nn.Linear(emb_dim, len(WHERE_OPS))
        # Order by
        self.order_by_q_num = nn.Linear(emb_dim, 3)
        self.order_by_head_num = nn.Linear(emb_dim, 3)
        self.order_by_is_asc = nn.Linear(emb_dim, 2)
        self.order_by_limit_num = nn.Linear(emb_dim, 3)
        self.order_by_agg = nn.Linear(emb_dim, len(AGG_OPS))

        # Store variables
        self.encoded_src = None
        self.encoded_src_att = None
        self.encoded_col = None
        self.encoded_tab = None


    def set_variables(self, encoded_src, encoded_src_att, encoded_col, encoded_tab):
        self.encoded_src = encoded_src
        self.encoded_src_att = encoded_src_att
        self.encoded_col = encoded_col
        self.encoded_tab = encoded_tab


    def decode(self, state_vector, qgm=None):
        # IUEN
        # Here (attention_vector 및 box 계산하여 context_vector 새로 계산)
        x = torch.zeros()
        state_vector, att = self.compute_context_vector(x, state_vector)

        iuen_score = self.iuen_linear(att)
        iuen_prob = torch.log_softmax(iuen_score, dim=-1)

        # Choose box_operator
        if self.is_train:
            op_key = [BOX_OPS[box['operator']] for box in qgm if box['operator'] in IUE_INDICES]
            assert len(op_key) == 1 or len(op_key) == 0
            selected_idx = IUEN.index(op_key) if op_key else IUEN.index('select')
            selected_prob = torch.gather(iuen_prob, 1, selected_idx)
        else:
            selected_prob, selected_idx = torch.topk(iuen_prob, 1)

        box_operator = BOX_OPS.index(IUEN[selected_idx])

        # Split boxes
        if self.is_train:
            split_idx = [idx for idx, box in enumerate(qgm) if box['operator'] in IUE_INDICES]
            assert len(split_idx) == 1
            front_boxes = qgm[:split_idx]
            rear_boxes = qgm[split_idx:]
        else:
            front_boxes = None
            rear_boxes = None

        state_vector = self.decode_box(state_vector, prev_box=None, gold_boxes=front_boxes)

        # Create boxes
        boxes = None

        if box_operator != BOX_OPS.index('select'):
            x = self.compute_lstm_input(boxes[-1], att)
            state_vector, att = self.compute_context_vector(x, state_vector)
            state_vector = self.decode_box(state_vector, prev_att=att, prev_bix=None, gold_boxes=rear_boxes)

        return state_vector


    def decode_box(self, state_vector, prev_att, prev_box, gold_boxes):
        '''To-Do: Need to split self.is_group_by for front and rear??'''

        x = self.compute_lstm_input(prev_box, prev_att)
        state_vector, att = self.compute_context_vector(x, state_vector)

        # Get scores for groupBy, orderBy
        groupBy_scores = self.is_group_by(state_vector)
        orderBy_scores = self.is_order_by(state_vector)

        # Change as probs
        front_groupBy_probs = torch.log_softmax(groupBy_scores)
        front_orderBy_probs = torch.log_softmax(orderBy_scores)

        # Get answer
        if self.is_train:
            ops = [box['operator'] for box in gold_boxes]
            group_selected_idx = GROUP_IDX in ops
            order_selected_idx = ORDER_IDX in ops
            group_selected_prob = torch.gather(front_groupBy_probs, 1, group_selected_idx)
            order_selected_prob = torch.gather(front_orderBy_probs, 1, order_selected_idx)
        else:
            group_selected_prob, group_selected_idx = torch.topk(front_groupBy_probs)
            order_selected_prob, order_selected_idx = torch.topk(front_orderBy_probs)

        is_groupBy = group_selected_idx == 1
        is_orderBy = order_selected_idx == 1

        # Predict each box
        # Here (attention_vector 및 box 계산하여 context_vector 새로 계산)
        context_vector = self.decode_select_box(state_vector)
        if is_groupBy:
            # Here (attention_vector 및 box 계산하여 context_vector 새로 계산)
            context_vector = self.decode_groupBy_box(state_vector)
        if is_orderBy:
            # Here (attention_vector 및 box 계산하여 context_vector 새로 계산)
            context_vector = self.decode_orderBy_box(state_vector)

        return context_vector


    def decode_select_box(self, state_vector, box):
        att = self.attention(self.encoded_src, self.encoded_src_att, state_vector[0])

        # Predict quantifier type f num
        qf_num_score = self.select_qf_num(att)
        qf_num_prob = torch.log_softmax(qf_num_score, dim=-1)
        if self.is_train:
            selected_qf_num_idx = len([item for item in box['body']['quantifier_types'] if item == 'f'])
            selected_qf_num_prob = torch.gather(qf_num_prob, 1, selected_qf_num_idx)
        else:
            selected_qf_num_prob, selected_qf_num_idx = torch.topk(qf_num_prob, 1)

        # Predict quantifier type s num
        qs_num_score = self.select_qs_num(att)
        qs_num_prob = torch.log_softmax(qs_num_score, dim=-1)
        if self.is_train:
            selected_qs_num_idx = len([item for item in box['body']['quantifier_types'] if item == 's'])
            selected_qs_num_prob = torch.gather(qs_num_prob, 1, selected_qs_num_idx)
        else:
            selected_qs_num_prob, selected_qs_num_idx = torch.topk(qs_num_prob, 1)

        # Predict from table
        table_score = self.pointerNetwork(self.encoded_tab, att)
        table_prob = torch.log_softmax(table_score, dim=-1)
        if self.is_train:
            selected_table_idx = box['body']['quantifiers']
            selected_table_prob = torch.gather(table_prob, 1, selected_table_idx)
        else:
            selected_table_prob, selected_table_idx = torch.topk(table_prob, selected_qf_num_idx)

        # Predict nested quantifier
        if self.is_train:
            box = [item for idx, item in enumerate(box['body']['quantifiers']) if
                   box['body']['quantifier_types'][idx] == 's']
        else:
            box = None
        # Context vector 바꿔줘야함
        something = self.decode(state_vector, box)

        ####################

        # Predict local predicates
            # For quantifier:
            # predict num
            # predict agg, col, operator
            # Predict Join predicates (Or use rule base or know)

        ####################

        # Predict Head num
        head_num_score = self.select_head_num(att)
        head_num_prob = torch.log_softmax(head_num_score)
        if self.is_train:
            selected_head_num_idx = len(box['head'])-1
            selected_head_num_prob = torch.gather(head_num_prob, 1, selected_head_num_idx)
        else:
            selected_head_num_prob, selected_head_num_idx = torch.topk(head_num_prob, 1)

        # Predict Head col
        column_score = self.pointerNetwork(self.encoded_col, att)
        column_prob = torch.log_sfotmax(column_score, dim=-1)
        # Mask out those not in quantifiers
        if self.is_train:
            selected_column_idx = [item[1] for item in box['head']]
            selected_column_prob = torch.gather(column_prob, 1, selected_column_idx)
        else:
            selected_column_prob, selected_column_idx = torch.topk(column_prob, selected_head_num_idx+1)

        # Batch_idx 단위로 한 번에
        tmp = []
        for c_idx in range(selected_column_idx):
            # concatenate att and encoded_col
            att_col = None
            agg_score = self.select_head_agg(att_col)
            agg_prob = torch.log_sfotmax(agg_score, dim=-1)
            if self.is_train:
                selected_agg_idx = box['head'][c_idx][0]
                selected_agg_prob = torch.gather(agg_prob, 1, selected_agg_idx)
            else:
                selected_agg_prob, selected_agg_idx = torch.topk(agg_prob, 1)
            tmp += [(selected_agg_prob, selected_agg_idx)]

        return state_vector


    def decode_groupBy_box(self, state_vector, box):
        att = self.attention(self.encoded_src, self.encoded_src_att, state_vector.hidden_state)

        # Predict quantifier type f num
        qf_num_score = self.group_by_qf_num(att)
        qf_num_prob = torch.log_softmax(qf_num_score, dim=-1)
        if self.is_train:
            selected_qf_num_idx = len([item for item in box['body']['quantifier_types'] if item == 'f'])
            selected_qf_num_prob = torch.gather(qf_num_prob, 1, selected_qf_num_idx)
        else:
            selected_qf_num_prob, selected_qf_num_idx = torch.topk(qf_num_prob, 1)

        # Predict quantifier type s num
        qs_num_score = self.group_by_qs_num(att)
        qs_num_prob = torch.log_softmax(qs_num_score, dim=-1)
        if self.is_train:
            selected_qs_num_idx = len([item for item in box['body']['quantifier_types'] if item == 's'])
            selected_qs_num_prob = torch.gather(qs_num_prob, 1 ,selected_qs_num_idx)
        else:
            selected_qs_num_prob, selected_qs_num_idx = torch.topk(qs_num_prob, 1)

        # Predict from table
        table_score = self.pointerNetwork(self.encoded_tab, att)
        table_prob = torch.log_softmax(table_score, dim=-1)
        if self.is_train:
            selected_table_idx = box['body']['quantifiers']
            selected_table_prob = torch.gather(table_prob, 1, selected_table_idx)
        else:
            selected_table_prob, selected_table_idx = torch.topk(table_prob, selected_qf_num_idx)

        # Predict nested quantifier
        if self.is_train:
            box = [item for idx, item in enumerate(box['body']['quantifiers']) if
                   box['body']['quantifier_types'][idx] == 's']
        else:
            box = None
        something = self.decode(state_vector, box)

        ############
        # Having

        # Predict local predicates num
        p_num_score = self.group_by_p_num(att)
        p_num_prob = torch.log_softmax(p_num_score, dim=-1)
        if self.is_train:
            selected_p_num_idx = None # Somehow
            selected_p_num_prob = torch.gather(p_num_prob, 1, selected_p_num_idx)
        else:
            selected_p_num_prob, selected_p_num_idx = torch.topk(p_num_prob, 1)

        # Predict predicates
            # Predict local predicates
            # For quantifier:
                # predict agg, col, operator
        ############

        # Predict Head Num
        head_num_score = self.group_by_head_num(att)
        head_num_prob = torch.log_softmax(head_num_score)
        if self.is_train:
            selected_head_num_idx = len(box['head'])-1
            selected_head_num_prob = torch.gather(head_num_prob, 1, selected_head_num_idx)
        else:
            selected_head_num_prob, selected_head_num_idx = torch.topk(head_num_prob, 1)

        # Predict Head column
        column_score = self.pointerNetwork(self.encoded_col, att)
        column_prob = torch.log_softmax(column_score, dim=-1)
        # Mask out those not in quantifiers

        if self.is_train:
            selected_column_idx = [item[1] for item in box['head']]
            selected_column_prob = torch.gather(column_prob, 1, selected_column_idx)
        else:
            selected_column_prob, selected_column_idx = torch.topk(column_prob, selected_head_num_idx+1)

        # Batch_idx 단위로 한 번에
        tmp = []
        for c_idx in range(selected_column_idx):
            # concatenate att and encoded_col
            att_col = None
            agg_score = self.group_by_head_agg(att_col)
            agg_prob = torch.log_sfotmax(agg_score, dim=-1)
            if self.is_train:
                selected_agg_idx = box['head'][c_idx][0]
                selected_agg_prob = torch.gather(agg_prob, 1, selected_agg_idx)
            else:
                selected_agg_prob, selected_agg_idx = torch.topk(agg_prob, 1)
            tmp += [(selected_agg_prob, selected_agg_idx)]

        return state_vector


    def decode_orderBy_box(self, state_vector, box):
        # Here (attention_vector 및 box 계산하여 context_vector 새로 계산)
        att = self.attention(self.encoded_src, self.encoded_src_att, state_vector.hidden_state)

        # Predict quantifiers - num
        num_score = self.order_by_num(att)
        num_prob = torch.log_softmax(num_score, dim=-1)
        if self.is_train:
            selected_num_idx = len(box['body']['quantifiers'])
            selected_num_prob = torch.gather(num_prob, 1, selected_num_idx)
        else:
            selected_num_prob, selected_num_idx = torch.topk(num_prob, 1)

        # Predict from table
        table_score = self.pointerNetwork(self.encoded_tab, att)
        table_prob = torch.log_softmax(table_score, dim=-1)
        if self.is_train:
            selected_table_idx = box['body']['quantifiers']
            selected_table_prob = torch.gather(table_prob, 1, selected_table_idx)
        else:
            selected_table_prob, selected_table_idx = torch.topk(table_prob, selected_num_idx)

        # Predict Head - num
        head_num_score = self.order_by_head_num(att)
        head_num_prob = torch.log_softmax(head_num_score, dim=-1)
        if self.is_train:
            selected_head_num_idx = len(box['head'])-1
            selected_head_num_prob = torch.gather(head_num_prob, 1, selected_head_num_idx)
        else:
            selected_head_num_prob, selected_head_num_idx = torch.topk(head_num_prob, 1)

        # Top n개의 컬럼을 뽑고, 컬럼에 맞춰서 agg를 따로 뽑기...
        # predict column
        column_score = self.pointerNetwork(self.encoded_col, att)
        column_prob = torch.log_softmax(column_score, dim=-1)
        # Mask out those not in quantifiers

        if self.is_train:
            selected_column_idx = [item[1] for item in box['head']]
            selected_column_prob = torch.gather(column_prob, 1, selected_column_idx)
        else:
            selected_column_prob, selected_column_idx = torch.topk(column_prob, selected_head_num_idx+1)

        # batch_idx 단위로 한번에 해야겠네.
        tmp = []
        for c_idx in range(selected_column_idx):
            # concatenate atten and encoded_col
            att_col = None
            agg_score = self.order_by_agg(att_col)
            agg_prob = torch.log_softmax(agg_score, dim=-1)
            if self.is_train:
                selected_agg_idx = box['head'][c_idx][0]
                selected_agg_prob = torch.gather(agg_prob, 1, selected_agg_idx)
            else:
                selected_agg_prob, selected_agg_idx = torch.topk(agg_prob, 1)
            tmp += [(selected_agg_prob, selected_agg_idx)]

        # predict is_asc
        is_asc_score = self.order_by_is_asc(att)
        is_asc_prob = torch.log_softmax(is_asc_score, dim=-1)
        if self.is_train:
            selected_is_asc_idx = int(box['is_asc'])
            selected_is_asc_prob = torch.gather(is_asc_prob, 1, selected_is_asc_idx)
        else:
            selected_is_asc_prob, selected_is_asc_idx = torch.topk(is_asc_prob, 1)

        # predict limit_num
        limit_num_score = self.order_by_limit_num(att)
        limit_num_prob = torch.log_softmax(limit_num_score, dim=-1)
        if self.is_train:
            selected_limit_num_idx = box['limit_num']
            selected_limit_num_prob = torch.gather(limit_num_prob, 1, selected_is_asc_idx)
        else:
            selected_limit_num_prob, selected_limit_num_idx = torch.topk(limit_num_prob, 1)

        return state_vector


    def encode_box(self, box, encoded_col, encoded_table, linear_layer, dropout):
        # Encode head
        tmp = []
        for head in box['head']:
            agg_id, col_id = head
            agg_emb = self.agg_emb(agg_id)
            col_emb = encoded_col[col_id]
            emb = agg_emb + col_emb
            tmp += [emb]
        head_emb = sum(tmp) / len(tmp)

        # Encode body (only quantifiers)
        tmp = []
        for table_id in box['quantifiers']:
            tab_emb = encoded_table[table_id]
            tmp += [tab_emb]
        body_emb = sum(tmp) / len(tmp)

        # Encode operator
        op_emb = self.op_emb(box['operator'])

        # Combine and encode
        context_vector = torch.cat([head_emb, body_emb, op_emb], 1)
        context_vector = F.relu(linear_layer(context_vector))
        context_vector = dropout(context_vector)

        return context_vector


    def get_attention_weights(self, source, query, source_mask=None):
        # Compute weight
        weight_scores = torch.bmm(source, query)

        # Masking
        if source_mask:
            weight_scores.data.masked_fill_(source_mask, -float('inf'))

        # Softmax
        weight_probs = torch.softmax(weight_scores, dim=-1)

        return weight_probs


    def attention(self, encoded_src, encoded_src_attn, state_vector, mask=None):
        # Compute attention weights
        att_weight = self.get_attention_weights(encoded_src_attn, state_vector, mask)

        # Compute new context_vector
        ctx_vector = torch.bmm(att_weight, encoded_src)

        return ctx_vector


    def compute_context_vector(self, x, encoded_src, encoded_src_attn, state_vector, mask=None):
        # Compute LSTM
        state_vector = self.lstm(x, state_vector)

        # Compute attention
        ctx_vector = self.attention(encoded_src, encoded_src_attn, state_vector, mask)
        att = self.att_layer(torch.cat([state_vector[0], ctx_vector], 1))

        return state_vector, att


    def compute_lstm_input(self, box, att):
        # Encode Box
        encoded_box = self.encode_box(box)
        return torch.cat([encoded_box, att])

