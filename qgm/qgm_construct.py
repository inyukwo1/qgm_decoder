from qgm.qgm import (
    QGM,
    QGMColumn,
    QGMSubqueryBox,
    QGMPredicateBox,
    QGMLocalPredicate,
    QGMProjection,
)
from qgm.qgm_action import QGM_ACTION
from typing import Union


def action_id_to_action(action_id):
    return QGM_ACTION.get_instance().action_id_to_action(action_id)


def qgm_construct_select_col_num(qgm: QGM):
    if qgm.is_gold:
        select_col_num = len(qgm.base_boxes.select_cols)
        yield "select_col_num", str(select_col_num), None
    else:

        def set_select_col_num(action_id):
            qgm.prev_action_ids += [action_id]

        yield "select_col_num", set_select_col_num, None


def qgm_construct_select_col(qgm: QGM, select_col_idx):
    if qgm.is_gold:
        qgm_column = qgm.base_boxes.select_cols[select_col_idx]
        setwise_col_id = qgm_column.setwise_column_id
        tab_id = qgm_column.table_id
        yield "C", setwise_col_id, None
        yield "T", tab_id, None
    else:

        def set_col_id(setwise_col_id):
            qgm.prev_action_ids += [setwise_col_id]

        def set_tab_id(tab_id):
            # TODO do we have to infer origin_column_id for QGMColumn?
            new_column = QGMColumn(qgm.base_boxes)
            new_column.setwise_column_id = qgm.prev_action_ids[-1]
            new_column.table_id = tab_id
            new_column.infer_origin_column_id()
            qgm.base_boxes.select_cols.append(new_column)

        yield "C", set_col_id, None
        yield "T", set_tab_id, None


def qgm_construct_infer_predicate_col_or_not(qgm: QGM, predicate_col_idx):
    if qgm.is_gold:
        action = (
            "do" if predicate_col_idx < len(qgm.base_boxes.predicate_cols) else "done"
        )
        yield "predicate_col_done", action, None
    else:

        def set_infer_col_or_not(action_id):
            qgm.prev_action_ids += [action_id]

        yield "predicate_col_done", set_infer_col_or_not, None


def qgm_condtruct_predicate_col(qgm: QGM, predicate_col_idx):
    if qgm.is_gold:
        qgm_column = qgm.base_boxes.predicate_cols[predicate_col_idx]
        setwise_col_id = qgm_column.setwise_column_id
        tab_id = qgm_column.table_id
        yield "C", setwise_col_id, None
        yield "T", tab_id, None
    else:

        def set_col_id(setwise_col_id):
            qgm.prev_action_ids += [setwise_col_id]

        def set_tab_id(tab_id):
            # TODO do we have to infer origin_column_id for QGMColumn?
            new_column = QGMColumn(qgm.base_boxes)
            new_column.setwise_column_id = qgm.prev_action_ids[-1]
            new_column.table_id = tab_id
            new_column.infer_origin_column_id()
            qgm.prev_action_ids += [tab_id]
            qgm.base_boxes.predicate_cols.append(new_column)

        yield "C", set_col_id, None
        yield "T", set_tab_id, None


def qgm_construct_subquery(
    subquery_box: QGMSubqueryBox,
    predicate_col_pointer_getter,
    predicate_col_pointer_increaser,
):
    qgm = subquery_box.find_qgm_root()
    if qgm.is_gold:
        agg = subquery_box.projection.agg
        yield "subquery_agg", agg, predicate_col_pointer_getter()
        predicate_col_pointer_increaser()
        for op_idx in range(len(subquery_box.local_predicates)):
            yield "subquery_predicate_col_done", "do", None
            yield from qgm_construct_predicate_op(
                subquery_box,
                predicate_col_pointer_getter,
                predicate_col_pointer_increaser,
                op_idx,
            )
        yield "subquery_predicate_col_done", "done", None
    else:

        def set_projection_agg(action_id):
            qgm.prev_action_ids += [action_id]
            subquery_box.projection = QGMProjection(
                subquery_box,
                QGM_ACTION.action_id_to_action(action_id),
                predicate_col_pointer_getter(),
            )

        def set_subquery_predicate_col_done(action_id):
            qgm.prev_action_ids += [action_id]

        yield "subquery_agg", set_projection_agg, predicate_col_pointer_getter()
        predicate_col_pointer_increaser()
        for op_idx in range(10):
            yield "subquery_predicate_col_done", set_subquery_predicate_col_done, None
            subquery_predicate_col_done = action_id_to_action(qgm.prev_action_ids[-1])
            assert subquery_predicate_col_done in {"do", "done"}
            if subquery_predicate_col_done == "done":
                break
            yield from qgm_construct_predicate_op(
                subquery_box,
                predicate_col_pointer_getter,
                predicate_col_pointer_increaser,
                op_idx,
            )


def qgm_construct_predicate_op(
    subquery_or_predicate_box: Union[QGMSubqueryBox, QGMPredicateBox],
    predicate_col_pointer_getter,
    predicate_col_pointer_increaser,
    op_idx,
):
    qgm = subquery_or_predicate_box.find_qgm_root()
    current_col_pointer = predicate_col_pointer_getter()

    if qgm.is_gold:
        if op_idx >= len(subquery_or_predicate_box.local_predicates):
            print("x")
        conj, local_predicate = subquery_or_predicate_box.local_predicates[op_idx]
        if op_idx != 0:
            assert conj in {"and", "or"}
            yield "predicate_conj", conj, current_col_pointer
        yield "predicate_op", local_predicate.op, current_col_pointer
        predicate_col_pointer_increaser()
        if local_predicate.op != "between":
            value_or_subquery_action = (
                "value" if local_predicate.is_right_hand_value() else "subquery"
            )
            yield "value_or_subquery", value_or_subquery_action, None
            if value_or_subquery_action == "subquery":
                yield from qgm_construct_subquery(
                    subquery_or_predicate_box.local_predicates[op_idx][
                        1
                    ].value_or_subquery,
                    predicate_col_pointer_getter,
                    predicate_col_pointer_increaser,
                )
        else:
            value_or_subquery_action = (
                "value"
                if local_predicate.if_op_between_right_hand_first_value()
                else "subquery"
            )
            yield "value_or_subquery", value_or_subquery_action, None
            if value_or_subquery_action == "subquery":
                yield from qgm_construct_subquery(
                    subquery_or_predicate_box.local_predicates[op_idx][
                        1
                    ].value_or_subquery[0],
                    predicate_col_pointer_getter,
                    predicate_col_pointer_increaser,
                )
            value_or_subquery_action = (
                "value"
                if local_predicate.if_op_between_right_hand_second_value()
                else "subquery"
            )
            yield "value_or_subquery", value_or_subquery_action, None
            if value_or_subquery_action == "subquery":
                yield from qgm_construct_subquery(
                    subquery_or_predicate_box.local_predicates[op_idx][
                        1
                    ].value_or_subquery[1],
                    predicate_col_pointer_getter,
                    predicate_col_pointer_increaser,
                )

    else:

        def set_predicate_conj(action_id):
            qgm.prev_action_ids += [action_id]

        def set_predicate_op(action_id):
            qgm.prev_action_ids += [action_id]

        def set_value_or_subquery(action_id):
            qgm.prev_action_ids += [action_id]

        if op_idx != 0:
            yield "predicate_conj", set_predicate_conj, current_col_pointer
        yield "predicate_op", set_predicate_op, current_col_pointer

        predicate_col_pointer_increaser()
        if op_idx != 0:
            conj = action_id_to_action(qgm.prev_action_ids[-2])
        else:
            conj = ""
        op = action_id_to_action(qgm.prev_action_ids[-1])

        if op != "between":
            new_local_predicate = QGMLocalPredicate(
                subquery_or_predicate_box, op, current_col_pointer, None
            )
            subquery_or_predicate_box.local_predicates.append(
                (conj, new_local_predicate)
            )
            yield "value_or_subquery", set_value_or_subquery, None
            value_or_subquery_action = action_id_to_action(qgm.prev_action_ids[-1])
            assert value_or_subquery_action in {"value", "subquery"}
            if value_or_subquery_action == "value":
                new_local_predicate.value_or_subquery = "value"  # TODO value prediction
            else:
                subquery_box = QGMSubqueryBox(new_local_predicate)
                new_local_predicate.value_or_subquery = subquery_box
                yield from qgm_construct_subquery(
                    subquery_box,
                    predicate_col_pointer_getter,
                    predicate_col_pointer_increaser,
                )
        else:
            new_local_predicate = QGMLocalPredicate(
                subquery_or_predicate_box, op, current_col_pointer, [None, None]
            )
            subquery_or_predicate_box.local_predicates.append(
                (conj, new_local_predicate)
            )
            yield "value_or_subquery", set_value_or_subquery, None
            value_or_subquery_action = action_id_to_action(qgm.prev_action_ids[-1])
            assert value_or_subquery_action in {"value", "subquery"}
            if value_or_subquery_action == "value":
                new_local_predicate.value_or_subquery[
                    0
                ] = "value"  # TODO value prediction
            else:
                subquery_box = QGMSubqueryBox(new_local_predicate)
                new_local_predicate.value_or_subquery[0] = subquery_box
                yield from qgm_construct_subquery(
                    subquery_box,
                    predicate_col_pointer_getter,
                    predicate_col_pointer_increaser,
                )
            yield "value_or_subquery", set_value_or_subquery, None
            value_or_subquery_action = action_id_to_action(qgm.prev_action_ids[-1])
            assert value_or_subquery_action in {"value", "subquery"}
            if value_or_subquery_action == "value":
                new_local_predicate.value_or_subquery[
                    1
                ] = "value"  # TODO value prediction
            else:
                subquery_box = QGMSubqueryBox(new_local_predicate)
                new_local_predicate.value_or_subquery[1] = subquery_box
                yield from qgm_construct_subquery(
                    subquery_box,
                    predicate_col_pointer_getter,
                    predicate_col_pointer_increaser,
                )


def qgm_construct_projection_agg(qgm: QGM, projection_col_pointer, agg_idx):
    qgm_projection_box = qgm.base_boxes.projection_box
    if qgm.is_gold:
        agg = qgm_projection_box.projections[agg_idx].agg
        yield "projection_agg", agg, projection_col_pointer
    else:

        def set_projection_agg(action_id):
            qgm.prev_action_ids += [action_id]
            agg = action_id_to_action(action_id)
            qgm_projection_box.add_projection_using_base_col(
                agg, projection_col_pointer
            )

        yield "projection_agg", set_projection_agg, projection_col_pointer


def qgm_construct(self: QGM):
    yield from qgm_construct_select_col_num(self)
    select_col_num = (
        len(self.base_boxes.select_cols)
        if self.is_gold
        else int(QGM_ACTION.action_id_to_action(self.prev_action_ids[-1]))
    )
    for select_col_idx in range(select_col_num):
        yield from qgm_construct_select_col(self, select_col_idx)
    for predicate_col_idx in range(10):  # predicate at most 10
        yield from qgm_construct_infer_predicate_col_or_not(self, predicate_col_idx)
        if self.is_gold:
            if predicate_col_idx == len(self.base_boxes.predicate_cols):
                break
        else:
            if action_id_to_action(self.prev_action_ids[-1]) == "done":
                break
        yield from qgm_condtruct_predicate_col(self, predicate_col_idx)

    predicate_col_pointer = {"pointer": 0}

    def predicate_col_pointer_getter():
        return predicate_col_pointer["pointer"]

    def predicate_col_pointer_increaser():
        predicate_col_pointer["pointer"] += 1

    for op_idx in range(10):
        if predicate_col_pointer["pointer"] >= len(self.base_boxes.predicate_cols):
            break
        yield from qgm_construct_predicate_op(
            self.base_boxes.predicate_box,
            predicate_col_pointer_getter,
            predicate_col_pointer_increaser,
            op_idx,
        )

    projection_col_pointer = 0
    for agg_idx in range(10):
        if projection_col_pointer >= len(self.base_boxes.select_cols):
            break
        yield from qgm_construct_projection_agg(self, projection_col_pointer, agg_idx)
        projection_col_pointer += 1
    yield None, None, None


QGM.qgm_construct = qgm_construct
