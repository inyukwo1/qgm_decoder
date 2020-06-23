from qgm.qgm import QGM, QGMColumn
from qgm.qgm_action import QGM_ACTION


def action_id_to_action(action_id):
    return QGM_ACTION.get_instance().action_id_to_action(action_id)


def qgm_construct_select_col_num(qgm: QGM):
    if qgm.is_gold:
        select_col_num = len(qgm.base_boxes.select_cols)
        select_col_num_ans = QGM_ACTION.get_instance().prob_action_id_mapping(
            "select_col_num", str(select_col_num)
        )
        yield "select_col_num", select_col_num_ans
    else:

        def set_select_col_num(action_id):
            qgm.prev_action_id = action_id

        yield "select_col_num", set_select_col_num


def qgm_construct_select_col(qgm: QGM, select_col_idx):
    qgm_column = qgm.base_boxes.select_cols[select_col_idx]
    setwise_col_id = qgm_column.setwise_column_id
    tab_id = qgm_column.table_id

    if qgm.is_gold:
        yield "C", setwise_col_id
        yield "T", tab_id
    else:

        def set_col_id(setwise_col_id):
            qgm.prev_action_id = setwise_col_id

        def set_tab_id(tab_id):
            # TODO do we have to infer origin_column_id for QGMColumn?
            new_column = QGMColumn(qgm.base_boxes)
            new_column.setwise_column_id = qgm.prev_action_id
            new_column.table_id = tab_id
            qgm.base_boxes.select_cols.append(new_column)

        yield "C", set_col_id
        yield "T", set_tab_id


def qgm_construct_infer_predicate_col_or_not(qgm: QGM, predicate_col_idx):
    if qgm.is_gold:
        action = (
            "do" if predicate_col_idx < len(qgm.base_boxes.predicate_cols) else "done"
        )
        yield "predicate_col_done", action
    else:

        def set_infer_col_or_not(action_id):
            qgm.prev_action_id = action_id

        yield "predicate_col_done", set_infer_col_or_not


def qgm_condtruct_predicate_col(qgm: QGM, predicate_col_idx):
    qgm_column = qgm.base_boxes.predicate_cols[predicate_col_idx]
    setwise_col_id = qgm_column.setwise_column_id
    tab_id = qgm_column.table_id

    if qgm.is_gold:
        yield "C", setwise_col_id
        yield "T", tab_id
    else:

        def set_col_id(setwise_col_id):
            qgm.prev_action_id = setwise_col_id

        def set_tab_id(tab_id):
            # TODO do we have to infer origin_column_id for QGMColumn?
            new_column = QGMColumn(qgm.base_boxes)
            new_column.setwise_column_id = qgm.prev_action_id
            new_column.table_id = tab_id
            qgm.base_boxes.select_cols.append(new_column)

        yield "C", set_col_id
        yield "T", set_tab_id


def qgm_construct_predicate_op(qgm: QGM, predicate_col_pointer, op_idx):
    qgm_prediate_box = qgm.base_boxes.predicate_box
    if qgm.is_gold:
        conj, op = qgm_prediate_box.local_predicates[op_idx]
        if op_idx != 0:
            assert conj in {"and", "or"}
            yield "predicate_and_or", conj
        yield "predicate_op", op
    else:

        def set_predicate_conj(action_id):
            qgm.prev_action_id = action_id

        def set_predicate_op(action_id):
            conj = action_id_to_action(qgm.prev_action_id)
            qgm.prev_action_id = action_id
            op = action_id_to_action(action_id)
            qgm_prediate_box.add_local_predicate(
                conj, op, predicate_col_pointer, "value"
            )  # TODO value prediction

        if op_idx != 0:
            yield "predicate_and_or", set_predicate_conj
        yield "predicate_op", set_predicate_op


def qgm_construct_projection_agg(qgm: QGM, projection_col_pointer, agg_idx):
    qgm_projection_box = qgm.base_boxes.projection_box
    if qgm.is_gold:
        agg = qgm_projection_box.projections[agg_idx].agg
        yield "projection_agg", agg
    else:

        def set_projection_agg(action_id):
            qgm.prev_action_id = action_id
            agg = action_id_to_action(action_id)
            qgm_projection_box.add_projection(agg, projection_col_pointer)

        yield "projection_agg", set_projection_agg


def qgm_construct(self: QGM):
    yield qgm_construct_select_col_num(self)
    select_col_num = (
        len(self.base_boxes.select_cols)
        if self.is_gold
        else int(QGM_ACTION.get_instance().action_id_to_action(self.prev_action_id))
    )
    for select_col_idx in range(select_col_num):
        yield qgm_construct_select_col(self, select_col_idx)
    for predicate_col_idx in range(10):  # predicate at most 10
        yield qgm_construct_infer_predicate_col_or_not(self, predicate_col_idx)
        if self.is_gold:
            if predicate_col_idx == len(self.base_boxes.predicate_cols):
                break
        else:
            if action_id_to_action(self.prev_action_id) == "done":
                break
        yield qgm_condtruct_predicate_col(self, predicate_col_idx)

    predicate_col_pointer = 0
    for op_idx in range(10):
        if predicate_col_pointer >= len(self.base_boxes.predicate_cols):
            break
        yield qgm_construct_predicate_op(self, predicate_col_pointer, op_idx)
        predicate_col_pointer += 1

    projection_col_pointer = 0
    for agg_idx in range(10):
        if projection_col_pointer >= len(self.base_boxes.select_cols):
            break
        yield qgm_construct_projection_agg(self, projection_col_pointer, agg_idx)
        projection_col_pointer += 1
