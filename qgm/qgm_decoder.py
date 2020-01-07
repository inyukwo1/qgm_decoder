import torch
import torch.nn as nn
import torch.nn.functional as F
BOX_OPS = ['select', 'groupBy', 'having', 'orderBy', 'intersect', 'union', 'except']
AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']
IUEN = ['select', 'intersect', 'union', 'except']
IUE = ['intersect', 'union', 'except']
IUE_IDX = [BOX_OPS.index(key) for key in IUE]

class QGM_Decoder(nn.Module):
    def __init__(self):
        super(QGM_Decoder, self).__init__()
        emb_dim = 300

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


    def decode(self, encoded_src, encoded_col, encoded_table, context_vector, qgm=None):
        '''
        1. predict box operator
            1-1. predict operator

        2: predict box - body - quantifiers
            2-1. predict num
            2-2. predict type
            2-3. if type == 's':
                recursive call
               if type == 'f':
                choose from DB table

        3: predict box - body - predicates
            3-1. predict local predicates
            3-1-1. predict num
            3-1-2. predict type
            3-1-3. predict agg, col, operator

            3-2. predict join predicates
            3-2-1. if more than one quantifiers, predict join column

        4: predict box - head
            4-1. predict num
            4-2. predict agg
            4-3. predict col


        Use coarse-to-fine design.
            1. use encoded_src and encoded_context to find
                if IUE
                if groupBy
                if orderBy


        iue 골랐으면, 그 operator로 src attention 해서 context vector 구하고서 groupBy, orderBy 선택
            즉, 첫번째는 select로, 두번째는 iue로 attention 하는거임


        :param encoded_nl:
        :param encoded_context:
        :return: [qgm box]
        '''
        # IUEN
        iuen_score = self.iuen_linear(context_vector)
        iuen_prob = torch.log_softmax(iuen_score, dim=-1)

        # Choose box_operator
        if self.is_train:
            op_key = [BOX_OPS[box['operator']] for box in qgm if box['operator'] in IUE_IDX]
            assert len(op_key) == 1 or len(op_key) == 0
            selected_idx = IUEN.index(op_key) if op_key else IUEN.index('select')
            selected_prob = torch.gather(iuen_prob, 1, selected_idx)
        else:
            selected_prob, selected_idx = torch.topk(iuen_prob, 1)

        box_operator = BOX_OPS.index(IUEN[selected_idx])

        # Get scores for groupBy, orderBy
        front_groupBy_scores = self.front_is_group_by(context_vector)
        front_orderBy_scores = self.front_is_order_by(context_vector)

        # Change as probs
        front_groupBy_probs = torch.log_softmax(front_groupBy_scores)
        front_orderBy_probs = torch.log_softmax(front_orderBy_scores)

        # Get answer
        if self.is_train:
            group_selected_idx = None
            order_selected_idx = None
            group_selected_prob = torch.gather(front_groupBy_probs, 1, group_selected_idx)
            order_selected_prob = torch.gather(front_orderBy_probs, 1, order_selected_idx)
        else:
            group_selected_prob, group_selected_idx = torch.topk(front_groupBy_probs)
            order_selected_prob, order_selected_idx = torch.topk(front_orderBy_probs)

        front_is_groupBy = group_selected_idx == 1
        front_is_orderBy = order_selected_idx == 1

        # Predict each box
        context_vector = self.decode_select_box(encoded_src, encoded_col, encoded_table, context_vector)
        if front_is_groupBy:
            context_vector = self.decode_groupBy_box(encoded_src, encoded_col, encoded_table, context_vector)
        if front_is_orderBy:
            context_vector = self.decode_orderBy_box(encoded_src, encoded_col, encoded_table, context_vector)

        # Back query
        if box_operator != BOX_OPS.index('select'):

            # Need to change context_vector somehow

            # Get scores for groupBy, orderBy
            back_groupBy_scores = self.back_is_group_by(context_vector)
            back_orderBy_scores = self.back_is_order_by(context_vector)

            # Change as probs
            back_groupBy_probs = torch.log_softmax(back_groupBy_scores)
            back_orderBy_probs = torch.log_softmax(back_orderBy_scores)

            # Get answers
            if self.is_train:
                group_selected_idx = None
                order_selected_idx = None
                group_selected_prob = torch.gather(back_groupBy_probs, 1, group_selected_idx)
                order_selected_prob = torch.gather(back_orderBy_probs, 1, order_selected_idx)
            else:
                group_selected_prob, group_selected_idx = torch.topk(back_groupBy_probs)
                order_selected_prob, order_selected_idx = torch.topk(back_orderBy_probs)

            back_is_groupBy = group_selected_idx == 1
            back_is_orderBy = order_selected_idx == 1

            # Predict each box
            context_vector = self.decode_select_box(encoded_src, encoded_col, encoded_table, context_vector)
            if back_is_groupBy:
                context_vector = self.decode_groupBy_box(encoded_src, encoded_col, encoded_table, context_vector)
            if back_is_orderBy:
                context_vector = self.decode_orderBy_box(encoded_src, encoded_col, encoded_table, context_vector)

        return context_vector


    def decode_select_box(self, encoded_src, encoded_col, encoded_table, context_vector):
        # predict Body
            # Predict quantifiers
                # Predict num
                # Predict type
                # Choose from table if type == f
                # Recursive call if type == s

            # Predict predicates
                # Predict local predicates
                # For quantifier:
                    # predict num
                    # predict agg, col, operator
                # Predict Join predicates (Or use rule base or know)

        # Predict Head
            # predict num
            # predict agg, col

        return context_vector


    def decode_groupBy_box(self, encoded_src, encoded_col, encoded_table, context_vector):
        # predict Body
            # Predict quantifiers
                # Predict num
                # Predict type
                # Choose from table if type == f
                # Recursive call if type == s

            # Predict predicates
                # Predict local predicates
                # For quantifier:
                    # predict num
                    # predict agg, col, operator

        # Predict Head
            # predict num
            # predict agg, col
        return context_vector


    def decode_orderBy_box(self, encoded_src, encoded_col, encoded_table, context_vector):
        # predict Body
            # Predict quantifiers
                # Predict num
                # Predict type
                # Choose from table if type == f
                # Recursive call if type == s

            # predict is_asc
            # predict limit_num

        # Predict Head
            # predict num
            # predict agg, col

        return context_vector


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
