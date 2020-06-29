from sql_ds.sql_ds import (
    SQLDataStructure,
    SQLWithOrder,
    SQLOrderClause,
    SQLBySet,
    SQLWithGroup,
    SQLGroupClause,
    SQLHavingClause,
    SQLHavingWhereClauseOne,
    SQLSelectClause,
    SQLFromClause,
    SQLJoinClause,
    SQLWhereClause,
    SQLLeftHand,
    SQLColumnWithAgg,
    SQLTable,
    SQLColumn,
    SQLValue,
    SQLBase,
    WHERE_OPS,
    UNIT_OPS,
    AGG_OPS,
)
from qgm.qgm import (
    QGM,
    QGMProjectionBox,
    QGMPredicateBox,
    QGMColumn,
    QGMSubqueryBox,
    QGMBaseBox,
)
from typing import Union


def _sql_left_hand_import_from_agg_setwise_colid(
    parent, from_clause, agg, setwise_colid, tab_id
):
    sql_left_hand = SQLLeftHand(parent)
    sql_col_agg = SQLColumnWithAgg(sql_left_hand)
    sql_left_hand.sql_column_with_agg = sql_col_agg
    sql_col_agg.agg = agg
    sql_column = SQLColumn(sql_col_agg)
    sql_col_agg.sql_column = sql_column
    sql_column.setwise_column_id = setwise_colid
    sql_column._find_parent_table(from_clause, tab_id)
    sql_column.infer_origin_col_id()
    return sql_left_hand


def _sql_where_clause_import_from_qgm_predicate_box_or_subquery_box(
    sql_with_group: SQLWithGroup,
    qgm_predicate_or_subquery_box: Union[QGMPredicateBox, QGMSubqueryBox],
):
    where_clause = SQLWhereClause(sql_with_group)
    for conj, qgm_local_predicate in qgm_predicate_or_subquery_box.local_predicates:
        qgm_column: QGMColumn = qgm_local_predicate.find_col()
        where_clause_one = SQLHavingWhereClauseOne(where_clause)
        sql_left_hand = _sql_left_hand_import_from_agg_setwise_colid(
            where_clause_one,
            sql_with_group.sql_from_clause,
            "none",
            qgm_column.setwise_column_id,
            qgm_column.table_id,
        )
        where_clause_one.left_hand = sql_left_hand
        where_clause_one.op = qgm_local_predicate.op

        def import_value_or_subquery(value_or_subquery):
            if isinstance(value_or_subquery, QGMSubqueryBox):  # subquery
                sql_with_order = _sql_with_order_import_from_qgm_subquery_box(
                    where_clause_one, value_or_subquery
                )
                value_or_sql_with_order = sql_with_order
            else:
                sqlvalue = SQLValue(where_clause_one)
                sqlvalue.value = value_or_subquery
                value_or_sql_with_order = sqlvalue
            return value_or_sql_with_order

        if where_clause_one.op == "between":
            where_clause_one.right_hand = import_value_or_subquery(
                qgm_local_predicate.value_or_subquery[0]
            )

            where_clause_one.right_hand2 = import_value_or_subquery(
                qgm_local_predicate.value_or_subquery[1]
            )
        else:
            where_clause_one.right_hand = import_value_or_subquery(
                qgm_local_predicate.value_or_subquery
            )
        where_clause.sql_where_clause_one_chain.append((conj, where_clause_one))
    return where_clause


def _sql_select_clause_import_from_qgm_projection_box(
    sql_with_group: SQLWithGroup, qgm_projection_box: QGMProjectionBox
):
    select_clause = SQLSelectClause(sql_with_group)
    for qgm_projection in qgm_projection_box.projections:
        qgm_column: QGMColumn = qgm_projection.find_col()
        sql_left_hand = _sql_left_hand_import_from_agg_setwise_colid(
            select_clause,
            sql_with_group.sql_from_clause,
            qgm_projection.agg,
            qgm_column.setwise_column_id,
            qgm_column.table_id,
        )
        select_clause.sql_left_hand_list.append(sql_left_hand)
    return select_clause


def _sql_from_clause_import_from_qgm(sql_with_group: SQLWithGroup, qgm: QGM):
    from_clause = SQLFromClause(sql_with_group)
    from_clause.tables_chain = []
    from_clause.join_clauses_chain = []
    for qgm_col in qgm.base_boxes.select_cols:
        from_clause.extend_using_shortest_path(qgm_col.table_id)
    for qgm_col in qgm.base_boxes.predicate_cols:
        from_clause.extend_using_shortest_path(qgm_col.table_id)
    return from_clause


def _sql_with_group_import_from_qgm(sql_by_set: SQLBySet, qgm: QGM):
    sql_with_group = SQLWithGroup(sql_by_set)

    sql_with_group.sql_from_clause = _sql_from_clause_import_from_qgm(
        sql_with_group, qgm
    )
    sql_with_group.sql_select_clause = _sql_select_clause_import_from_qgm_projection_box(
        sql_with_group, qgm.base_boxes.projection_box
    )
    sql_with_group.sql_where_clause = _sql_where_clause_import_from_qgm_predicate_box_or_subquery_box(
        sql_with_group, qgm.base_boxes.predicate_box
    )
    # TODO support group clause

    return sql_with_group


def _sql_with_group_import_from_qgm_subquery_box(
    sql_by_set: SQLBySet, qgm_subquery_box: QGMSubqueryBox
):
    sql_with_group = SQLWithGroup(sql_by_set)
    from_clause = SQLFromClause(sql_with_group)
    from_clause.tables_chain = []
    from_clause.join_clauses_chain = []
    from_clause.extend_using_shortest_path(
        qgm_subquery_box.projection.find_col().table_id
    )
    for conj, predicate in qgm_subquery_box.local_predicates:
        from_clause.extend_using_shortest_path(predicate.find_col().table_id)
    sql_with_group.sql_from_clause = from_clause

    select_clause = SQLSelectClause(sql_with_group)
    qgm_column = qgm_subquery_box.projection.find_col()
    sql_left_hand = _sql_left_hand_import_from_agg_setwise_colid(
        select_clause,
        sql_with_group.sql_from_clause,
        qgm_subquery_box.projection.agg,
        qgm_column.setwise_column_id,
        qgm_column.table_id,
    )
    select_clause.sql_left_hand_list.append(sql_left_hand)
    sql_with_group.sql_select_clause = select_clause

    sql_with_group.sql_where_clause = _sql_where_clause_import_from_qgm_predicate_box_or_subquery_box(
        sql_with_group, qgm_subquery_box
    )

    return sql_with_group


def _sql_by_set_import_from_qgm_or_subquery(
    sql_with_order: SQLWithOrder, qgm_or_subquery: Union[QGM, QGMSubqueryBox]
):
    sql_by_set = SQLBySet(sql_with_order)
    # TODO currently don't support set operator
    if isinstance(qgm_or_subquery, QGM):
        sql_by_set.sql_with_group_chain.append(
            ("", _sql_with_group_import_from_qgm(sql_by_set, qgm_or_subquery))
        )
    else:
        sql_by_set.sql_with_group_chain.append(
            (
                "",
                _sql_with_group_import_from_qgm_subquery_box(
                    sql_by_set, qgm_or_subquery
                ),
            )
        )

    return sql_by_set


def _sql_order_clause_import_from_qgm(
    sql_with_order, qgm_basebox_or_subquerybox: Union[QGMBaseBox, QGMSubqueryBox]
):
    if qgm_basebox_or_subquerybox.orderby_box is None:
        return None
    qgm_order_clause = qgm_basebox_or_subquerybox.orderby_box
    sql_order_clause = SQLOrderClause(sql_with_order)
    sql_order_clause.limit_num = qgm_order_clause.limit_num
    sql_order_clause.sql_order_direction = qgm_order_clause.direction
    lefthand = _sql_left_hand_import_from_agg_setwise_colid(
        sql_order_clause,
        sql_with_order.get_from_clause(),
        qgm_order_clause.agg,
        qgm_order_clause.find_col().setwise_column_id,
        qgm_order_clause.find_col().table_id,
    )
    # TODO assume one column
    sql_order_clause.sql_column_with_agg_list.append(lefthand)

    return sql_order_clause


def _sql_with_order_import_from_qgm_subquery_box(
    parent: SQLBase, qgm_subquery_box: QGMSubqueryBox
):
    sql_with_order = SQLWithOrder(parent)
    sql_with_order.sql_by_set = _sql_by_set_import_from_qgm_or_subquery(
        sql_with_order, qgm_subquery_box
    )
    sql_with_order.sql_order_clause = _sql_order_clause_import_from_qgm(
        sql_with_order, qgm_subquery_box
    )
    return sql_with_order


def _sql_with_order_import_from_qgm(sql_ds: SQLDataStructure, qgm: QGM):
    sql_with_order = SQLWithOrder(sql_ds)
    sql_with_order.sql_by_set = _sql_by_set_import_from_qgm_or_subquery(
        sql_with_order, qgm
    )
    sql_with_order.sql_order_clause = _sql_order_clause_import_from_qgm(
        sql_with_order, qgm.base_boxes
    )
    return sql_with_order


def sql_import_from_qgm(self: SQLDataStructure, qgm: QGM):
    self.db = qgm.db
    self.sql_with_order = _sql_with_order_import_from_qgm(self, qgm)


SQLDataStructure.import_from_qgm = sql_import_from_qgm
