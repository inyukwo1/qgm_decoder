from qgm.qgm import (
    QGM,
    QGMBaseBox,
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
        yield "select_col_num", None


def qgm_construct_select_col(qgm: QGM, select_col_idx):
    if qgm.is_gold:
        qgm_column = qgm.base_boxes.select_cols[select_col_idx]
        setwise_col_id = qgm_column.setwise_column_id
        tab_id = qgm_column.table_id
        yield "C", setwise_col_id, None
        yield "T", tab_id, None
    else:
        new_column = QGMColumn(qgm.base_boxes)
        yield "C", None
        new_column.setwise_column_id = qgm.prev_symbol_actions[-1][1]
        yield "T", None
        new_column.table_id = qgm.prev_symbol_actions[-1][1]
        new_column.infer_origin_column_id()
        qgm.base_boxes.select_cols.append(new_column)


def qgm_construct_infer_predicate_col_or_not(qgm: QGM, predicate_col_idx):
    if qgm.is_gold:
        action = (
            "do" if predicate_col_idx < len(qgm.base_boxes.predicate_cols) else "done"
        )
        yield "predicate_col_done", action, None
    else:
        yield "predicate_col_done", None


def qgm_condtruct_predicate_col(qgm: QGM, predicate_col_idx):
    if qgm.is_gold:
        qgm_column = qgm.base_boxes.predicate_cols[predicate_col_idx]
        setwise_col_id = qgm_column.setwise_column_id
        tab_id = qgm_column.table_id
        yield "C", setwise_col_id, None
        yield "T", tab_id, None
    else:
        new_column = QGMColumn(qgm.base_boxes)
        yield "C", None
        new_column.setwise_column_id = qgm.prev_symbol_actions[-1][1]
        yield "T", None
        new_column.table_id = qgm.prev_symbol_actions[-1][1]

        new_column.infer_origin_column_id()
        qgm.base_boxes.predicate_cols.append(new_column)


def qgm_construct_subquery(
    subquery_box: QGMSubqueryBox,
    predicate_col_pointer_getter,
    predicate_col_pointer_increaser,
):
    qgm = subquery_box.find_qgm_root()
    if qgm.is_gold:
        if subquery_box.projection.key_column is not None:
            current_col_pointer = None
            yield "col_previous_key_stack", "key", None
            yield "T", subquery_box.projection.key_column.table_id, None
            yield "C", subquery_box.projection.key_column.setwise_column_id, None
        elif subquery_box.projection.col_pointer != predicate_col_pointer_getter():
            assert (
                subquery_box.projection.col_pointer
                == predicate_col_pointer_getter() - 1
            )
            current_col_pointer = subquery_box.projection.col_pointer
            yield "col_previous_key_stack", "previous", None
        else:
            assert False
            current_col_pointer = predicate_col_pointer_getter()
            predicate_col_pointer_increaser()
            yield "col_previous_key_stack", "stack", None
        agg = subquery_box.projection.agg
        yield "subquery_agg", agg, current_col_pointer
        for op_idx in range(len(subquery_box.local_predicates)):
            yield "subquery_op_done", "do", None
            yield from qgm_construct_predicate_op(
                subquery_box,
                predicate_col_pointer_getter,
                predicate_col_pointer_increaser,
                op_idx,
            )
        yield "subquery_op_done", "done", None
    else:

        yield "col_previous_key_stack", None

        prev_key_stack = qgm.prev_symbol_actions[-1][1]
        if prev_key_stack == "key":
            current_col_pointer = None
            yield "T", None
            key_tab_id = qgm.prev_symbol_actions[-1][1]
            yield "C", None
            key_col_id = qgm.prev_symbol_actions[-1][1]
        else:  # TODO we don't assume stack
            current_col_pointer = predicate_col_pointer_getter() - 1

        yield "subquery_agg", current_col_pointer
        agg = qgm.prev_symbol_actions[-1][1]
        if prev_key_stack == "key":
            subquery_box.set_projection_by_key(agg, key_col_id, key_tab_id)
        else:
            subquery_box.set_projection_by_pointer(agg, current_col_pointer)
        for op_idx in range(10):
            yield "subquery_op_done", None
            subquery_op_done = qgm.prev_symbol_actions[-1][1]
            assert subquery_op_done in {"do", "done"}
            if subquery_op_done == "done":
                break
            yield from qgm_construct_predicate_op(
                subquery_box,
                predicate_col_pointer_getter,
                predicate_col_pointer_increaser,
                op_idx,
            )
    yield from qgm_construct_orderby(
        subquery_box, predicate_col_pointer_getter, predicate_col_pointer_increaser
    )


def qgm_construct_predicate_op(
    subquery_or_predicate_box: Union[QGMSubqueryBox, QGMPredicateBox],
    predicate_col_pointer_getter,
    predicate_col_pointer_increaser,
    op_idx,
):
    qgm = subquery_or_predicate_box.find_qgm_root()

    if qgm.is_gold:
        conj, local_predicate = subquery_or_predicate_box.local_predicates[op_idx]
        if local_predicate.key_column is None:
            yield "col_previous_key_stack", "stack", None
            current_col_pointer = predicate_col_pointer_getter()
            predicate_col_pointer_increaser()
        else:
            yield "col_previous_key_stack", "key", None
            yield "T", local_predicate.key_column.table_id, None
            yield "C", local_predicate.key_column.setwise_column_id, None
            current_col_pointer = None
        if op_idx != 0:
            assert conj in {"and", "or"}
            yield "predicate_conj", conj, current_col_pointer
        yield "predicate_op", local_predicate.op, current_col_pointer

        def yield_value_or_subquery(
            is_value_or_subquery, value_or_subquery, col_pointer
        ):
            value_or_subquery_action = "value" if is_value_or_subquery else "subquery"
            yield "value_or_subquery", value_or_subquery_action, col_pointer
            if value_or_subquery_action == "subquery":
                yield from qgm_construct_subquery(
                    value_or_subquery,
                    predicate_col_pointer_getter,
                    predicate_col_pointer_increaser,
                )

        if local_predicate.op != "between":
            yield from yield_value_or_subquery(
                local_predicate.is_right_hand_value(),
                subquery_or_predicate_box.local_predicates[op_idx][1].value_or_subquery,
                current_col_pointer,
            )
        else:
            yield from yield_value_or_subquery(
                local_predicate.if_op_between_right_hand_first_value(),
                subquery_or_predicate_box.local_predicates[op_idx][1].value_or_subquery[
                    0
                ],
                current_col_pointer,
            )
            yield from yield_value_or_subquery(
                local_predicate.if_op_between_right_hand_second_value(),
                subquery_or_predicate_box.local_predicates[op_idx][1].value_or_subquery[
                    1
                ],
                current_col_pointer,
            )

    else:
        yield "col_previous_key_stack", None
        prev_key_stack = qgm.prev_symbol_actions[-1][1]
        if prev_key_stack == "key":
            current_col_pointer = None
            yield "T", None
            key_tab_id = qgm.prev_symbol_actions[-1][1]
            yield "C", None
            key_col_id = qgm.prev_symbol_actions[-1][1]
        else:
            current_col_pointer = predicate_col_pointer_getter()
            predicate_col_pointer_increaser()
            # If "prev", We don't want to do "prev", so we redirect for "stack"
        conj = ""
        if op_idx != 0:
            yield "predicate_conj", current_col_pointer
            conj = qgm.prev_symbol_actions[-1][1]
        yield "predicate_op", current_col_pointer
        op = qgm.prev_symbol_actions[-1][1]

        if op != "between":
            new_local_predicate = QGMLocalPredicate(
                subquery_or_predicate_box, op, current_col_pointer, None, None
            )
            if prev_key_stack == "key":
                key_column = QGMColumn(new_local_predicate)
                key_column.setwise_column_id = key_col_id
                key_column.table_id = key_tab_id
                key_column.infer_origin_column_id()
                new_local_predicate.key_column = key_column
            subquery_or_predicate_box.local_predicates.append(
                (conj, new_local_predicate)
            )
            yield "value_or_subquery", current_col_pointer
            value_or_subquery_action = qgm.prev_symbol_actions[-1][1]
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
                subquery_or_predicate_box, op, current_col_pointer, None, [None, None]
            )
            if prev_key_stack == "key":
                key_column = QGMColumn(new_local_predicate)
                key_column.setwise_column_id = key_col_id
                key_column.table_id = key_tab_id
                key_column.infer_origin_column_id()
                new_local_predicate.key_column = key_column
            subquery_or_predicate_box.local_predicates.append(
                (conj, new_local_predicate)
            )
            yield "value_or_subquery", current_col_pointer
            value_or_subquery_action = qgm.prev_symbol_actions[-1][1]
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
            yield "value_or_subquery", current_col_pointer
            value_or_subquery_action = qgm.prev_symbol_actions[-1][1]
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
        yield "projection_agg", projection_col_pointer
        agg = qgm.prev_symbol_actions[-1][1]
        qgm_projection_box.add_projection_using_base_col(
            agg, projection_col_pointer, None
        )


def qgm_construct_orderby(
    qgm_basebox_or_subquerybox: Union[QGMBaseBox, QGMSubqueryBox],
    predicate_col_pointer_getter,
    predicate_col_pointer_increaser,
):
    qgm = qgm_basebox_or_subquerybox.find_qgm_root()
    if qgm.is_gold:
        if qgm_basebox_or_subquerybox.orderby_box is None:
            yield "orderby_exists", "no", None
            return
        else:
            yield "orderby_exists", "yes", None
        current_col_pointer = predicate_col_pointer_getter()
        yield "orderby_direction", qgm_basebox_or_subquerybox.orderby_box.direction, current_col_pointer
        yield "orderby_agg", qgm_basebox_or_subquerybox.orderby_box.agg, current_col_pointer
        predicate_col_pointer_increaser()
    else:

        yield "orderby_exists", None
        orderby_exists = qgm.prev_symbol_actions[-1][1]
        assert orderby_exists in {"no", "yes"}
        if orderby_exists == "no":
            return
        current_col_pointer = predicate_col_pointer_getter()
        yield "orderby_direction", current_col_pointer
        direction = qgm.prev_symbol_actions[-1][1]
        yield "orderby_agg", current_col_pointer
        agg = qgm.prev_symbol_actions[-1][1]

        qgm_basebox_or_subquerybox.add_orderby_using_base_col(
            direction, agg, "value", current_col_pointer
        )
        predicate_col_pointer_increaser()


def qgm_construct_groupby(
    qgm_basebox_or_subquerybox: Union[QGMBaseBox, QGMSubqueryBox],
    predicate_col_pointer_getter,
    predicate_col_pointer_increaser,
):
    qgm = qgm_basebox_or_subquerybox.find_qgm_root()
    if qgm.is_gold:
        if qgm_basebox_or_subquerybox.groupby_box is None:
            yield "groupby_exists", "no", None
            return
        else:
            yield "groupby_exists", "yes", None
            if qgm_basebox_or_subquerybox.groupby_box.is_key:
                yield "col_previous_key_stack", "key", None
                yield "T", qgm_basebox_or_subquerybox.groupby_box.column.table_id, None
                yield "C", qgm_basebox_or_subquerybox.groupby_box.column.setwise_column_id, None
            else:
                yield "col_previous_key_stack", "previous", None
                yield "C", qgm_basebox_or_subquerybox.groupby_box.column.setwise_column_id, None
                yield "T", qgm_basebox_or_subquerybox.groupby_box.column.table_id, None
            # TODO we assume groupby col is prev col or key

            if qgm_basebox_or_subquerybox.groupby_box.having_predicate is None:
                yield "having_exists", "no", None
            else:
                yield "having_exists", "yes", None
                current_col_pointer = predicate_col_pointer_getter()
                yield "having_agg", qgm_basebox_or_subquerybox.groupby_box.having_agg, current_col_pointer
                yield "predicate_op", qgm_basebox_or_subquerybox.groupby_box.having_predicate.op, current_col_pointer
                predicate_col_pointer_increaser()
    else:

        yield "groupby_exists", None
        groupby_exists = qgm.prev_symbol_actions[-1][1]
        assert groupby_exists in {"no", "yes"}
        if groupby_exists == "no":
            return
        yield "col_previous_key_stack", None
        prev_key_stack = qgm.prev_symbol_actions[-1][1]
        if prev_key_stack == "key":
            yield "T", None
            tab_id = qgm.prev_symbol_actions[-1][1]
            yield "C", None
            col_id = qgm.prev_symbol_actions[-1][1]
            is_key = True

        else:
            # TODO we assume there is no stack
            yield "C", None
            col_id = qgm.prev_symbol_actions[-1][1]
            yield "T", None
            tab_id = qgm.prev_symbol_actions[-1][1]
            is_key = False

        yield "having_exists", None
        having_exists = qgm.prev_symbol_actions[-1][1]
        assert having_exists in {"no", "yes"}
        if having_exists == "no":
            qgm_basebox_or_subquerybox.add_groupby_using_base_col(
                col_id, tab_id, is_key
            )
            return

        current_col_pointer = predicate_col_pointer_getter()
        yield "having_agg", current_col_pointer
        agg = qgm.prev_symbol_actions[-1][1]
        yield "predicate_op", current_col_pointer
        op = qgm.prev_symbol_actions[-1][1]

        having_col_pointer = predicate_col_pointer_getter()
        qgm_basebox_or_subquerybox.add_groupby_using_base_col(
            col_id, tab_id, is_key, having_col_pointer, agg, op, "value"
        )

        predicate_col_pointer_increaser()


def qgm_construct(self: QGM):
    yield from qgm_construct_select_col_num(self)
    select_col_num = (
        len(self.base_boxes.select_cols)
        if self.is_gold
        else int(self.prev_symbol_actions[-1][1])
    )
    for select_col_idx in range(select_col_num):
        yield from qgm_construct_select_col(self, select_col_idx)
    for predicate_col_idx in range(20):  # predicate at most 10
        yield from qgm_construct_infer_predicate_col_or_not(self, predicate_col_idx)
        if self.is_gold:
            if predicate_col_idx == len(self.base_boxes.predicate_cols):
                break
        else:
            if self.prev_symbol_actions[-1][1] == "done":
                break
        yield from qgm_condtruct_predicate_col(self, predicate_col_idx)

    projection_col_pointer = 0
    for agg_idx in range(10):
        if projection_col_pointer >= len(self.base_boxes.select_cols):
            break
        yield from qgm_construct_projection_agg(self, projection_col_pointer, agg_idx)
        projection_col_pointer += 1

    predicate_col_pointer = {"pointer": 0}

    def predicate_col_pointer_getter():
        return predicate_col_pointer["pointer"]

    def predicate_col_pointer_increaser():
        predicate_col_pointer["pointer"] += 1

    if self.is_gold:
        for op_idx in range(len(self.base_boxes.predicate_box.local_predicates)):
            yield "op_done", "do", None
            yield from qgm_construct_predicate_op(
                self.base_boxes.predicate_box,
                predicate_col_pointer_getter,
                predicate_col_pointer_increaser,
                op_idx,
            )
        yield "op_done", "done", None
    else:

        for op_idx in range(10):
            yield "op_done", None
            _, op_done = self.prev_symbol_actions[-1]
            assert op_done in {"do", "done"}
            if op_done == "done":
                break
            yield from qgm_construct_predicate_op(
                self.base_boxes.predicate_box,
                predicate_col_pointer_getter,
                predicate_col_pointer_increaser,
                op_idx,
            )
    yield from qgm_construct_groupby(
        self.base_boxes, predicate_col_pointer_getter, predicate_col_pointer_increaser
    )
    yield from qgm_construct_orderby(
        self.base_boxes, predicate_col_pointer_getter, predicate_col_pointer_increaser
    )
    if self.is_gold:
        yield None, None, None
    else:
        yield None, None


QGM.qgm_construct = qgm_construct
