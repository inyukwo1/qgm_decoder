from qgm.qgm import QGM, QGMColumn, QGMBase, QGMSubqueryBox, QGMPredicateBox
from sql_ds.sql_ds import (
    SQLDataStructure,
    SQLColumn,
    SQLHavingWhereClauseOne,
    SQLColumnWithAgg,
    SQLValue,
    SQLWithOrder,
)
from typing import List, Tuple, Union


def qgm_import_from_sql_ds(sql_ds: SQLDataStructure):
    if (
        sql_ds.has_subquery_from()
        or sql_ds.has_subquery_select()
        or sql_ds.has_set_operator()
    ):  # TODO currently not supporting conditions
        return None
    new_qgm = QGM(sql_ds.db, True)
    select_clause_columns: List[
        SQLColumnWithAgg
    ] = sql_ds.sql_with_order.get_select_clauses()
    predicates: List[
        Tuple[str, SQLHavingWhereClauseOne]
    ] = sql_ds.sql_with_order.get_where_clauses()
    for sql_column_with_agg in select_clause_columns:
        new_column = QGMColumn(new_qgm.base_boxes)
        new_column.import_from_sql_column(sql_column_with_agg.sql_column)
        new_qgm.base_boxes.projection_box.add_projection(
            new_column, sql_column_with_agg.agg
        )

    for conj, prediate in predicates:
        append_local_predicate_from_conj_predicate(
            new_qgm.base_boxes.predicate_box, conj, prediate
        )
    group_clause = sql_ds.sql_with_order.sql_by_set.sql_with_group_chain[0][
        1
    ].sql_group_clause
    if group_clause is not None:
        assert len(sql_ds.sql_with_order.sql_by_set.sql_with_group_chain) == 1
        new_column = QGMColumn(new_qgm.base_boxes)
        new_column.import_from_sql_column(group_clause.sql_column)
        having_clause = group_clause.sql_having_clause
        if len(having_clause.sql_having_clause_one_chain) == 0:
            new_qgm.base_boxes.add_groupby(new_column)
        else:
            having_column = QGMColumn(new_qgm.base_boxes)
            assert len(having_clause.sql_having_clause_one_chain) == 1
            having_clause_one = having_clause.sql_having_clause_one_chain[0][1]
            having_column.import_from_sql_column(
                having_clause_one.left_hand.sql_column_with_agg.sql_column
            )

            new_qgm.base_boxes.add_groupby(
                new_column,
                having_column,
                having_clause_one.left_hand.sql_column_with_agg.agg,
                having_clause_one.op,
                None,
            )
            if having_clause_one.op == "between":
                value_or_subquery = (
                    qgm_import_value_or_subquery_from_sql_ds(
                        new_qgm.base_boxes.groupby_box.having_predicate,
                        having_clause_one.right_hand,
                    ),
                    qgm_import_value_or_subquery_from_sql_ds(
                        new_qgm.base_boxes.groupby_box.having_predicate,
                        having_clause_one.right_hand2,
                    ),
                )
            else:
                value_or_subquery = qgm_import_value_or_subquery_from_sql_ds(
                    new_qgm.base_boxes.groupby_box.having_predicate,
                    having_clause_one.right_hand,
                )
            new_qgm.base_boxes.groupby_box.having_predicate.value_or_subquery = (
                value_or_subquery
            )

    order_clause = sql_ds.sql_with_order.sql_order_clause
    if order_clause is not None:
        assert len(order_clause.sql_column_with_agg_list) == 1
        assert order_clause.sql_column_with_agg_list[0].op == "none"
        sql_column_with_agg = order_clause.sql_column_with_agg_list[
            0
        ].sql_column_with_agg
        new_column = QGMColumn(new_qgm.base_boxes)
        new_column.import_from_sql_column(sql_column_with_agg.sql_column)
        new_qgm.base_boxes.add_orderby(
            order_clause.sql_order_direction,
            sql_column_with_agg.agg,
            order_clause.limit_num,
            new_column,
        )

    return new_qgm


def append_local_predicate_from_conj_predicate(
    parent: Union[QGMSubqueryBox, QGMPredicateBox],
    conj: str,
    predicate: SQLHavingWhereClauseOne,
):
    new_column = QGMColumn(parent.find_base_box())
    if (
        not predicate.left_hand.no_op_and_agg()
    ):  # TODO currently we process just simple cols
        return None
    new_column.import_from_sql_column(
        predicate.left_hand.sql_column_with_agg.sql_column
    )
    if predicate.not_op:
        assert predicate.op == "in"
        op = "not in"
    else:
        op = predicate.op
    parent.add_local_predicate(conj, new_column, op, None)
    if predicate.op == "between":
        value_or_subquery = (
            qgm_import_value_or_subquery_from_sql_ds(parent, predicate.right_hand),
            qgm_import_value_or_subquery_from_sql_ds(parent, predicate.right_hand2),
        )
    else:
        value_or_subquery = qgm_import_value_or_subquery_from_sql_ds(
            parent, predicate.right_hand
        )
    parent.local_predicates[-1][1].value_or_subquery = value_or_subquery


def qgm_import_value_or_subquery_from_sql_ds(
    parent: QGMBase, sql_value_or_subquery: Union[SQLValue, SQLWithOrder]
):
    if isinstance(sql_value_or_subquery, SQLValue):
        return sql_value_or_subquery.value
    else:
        subquery_box = QGMSubqueryBox(parent)
        qgm_base = subquery_box.find_base_box()
        sql_subquery: SQLWithOrder = sql_value_or_subquery
        select_clause_columns: List[
            SQLColumnWithAgg
        ] = sql_subquery.get_select_clauses()
        predicates: List[
            Tuple[str, SQLHavingWhereClauseOne]
        ] = sql_subquery.get_where_clauses()
        assert (
            len(select_clause_columns) == 1
        )  # TODO currently we dont support multi col selection in subquery
        sql_column_with_agg = select_clause_columns[0]
        new_column = QGMColumn(qgm_base)
        new_column.import_from_sql_column(sql_column_with_agg.sql_column)
        subquery_box.set_projection(new_column, sql_column_with_agg.agg)
        for conj, predicate in predicates:
            append_local_predicate_from_conj_predicate(subquery_box, conj, predicate)

        group_clause = sql_subquery.sql_by_set.sql_with_group_chain[0][
            1
        ].sql_group_clause
        if group_clause is not None:
            assert len(sql_subquery.sql_by_set.sql_with_group_chain) == 1
            new_column = QGMColumn(qgm_base)
            new_column.import_from_sql_column(group_clause.sql_column)
            having_clause = group_clause.sql_having_clause
            if len(having_clause.sql_having_clause_one_chain) == 0:
                subquery_box.add_groupby(new_column)
            else:
                having_column = QGMColumn(qgm_base)
                assert len(having_clause.sql_having_clause_one_chain) == 1
                having_clause_one = having_clause.sql_having_clause_one_chain[0][1]
                having_column.import_from_sql_column(
                    having_clause_one.left_hand.sql_column_with_agg.sql_column
                )

                subquery_box.add_groupby(
                    new_column,
                    having_column,
                    having_clause_one.left_hand.sql_column_with_agg.agg,
                    having_clause_one.op,
                    None,
                )
                if having_clause_one.op == "between":
                    value_or_subquery = (
                        qgm_import_value_or_subquery_from_sql_ds(
                            subquery_box.groupby_box.having_predicate,
                            having_clause_one.right_hand,
                        ),
                        qgm_import_value_or_subquery_from_sql_ds(
                            subquery_box.groupby_box.having_predicate,
                            having_clause_one.right_hand2,
                        ),
                    )
                else:
                    value_or_subquery = qgm_import_value_or_subquery_from_sql_ds(
                        subquery_box.groupby_box.having_predicate,
                        having_clause_one.right_hand,
                    )
                subquery_box.groupby_box.having_predicate.value_or_subquery = (
                    value_or_subquery
                )

        order_clause = sql_subquery.sql_order_clause
        if order_clause is not None:
            assert len(order_clause.sql_column_with_agg_list) == 1
            assert order_clause.sql_column_with_agg_list[0].op == "none"
            sql_column_with_agg = order_clause.sql_column_with_agg_list[
                0
            ].sql_column_with_agg
            new_column = QGMColumn(subquery_box.find_base_box())
            new_column.import_from_sql_column(sql_column_with_agg.sql_column)
            subquery_box.add_orderby(
                order_clause.sql_order_direction,
                sql_column_with_agg.agg,
                order_clause.limit_num,
                new_column,
            )
        return subquery_box


def qgm_column_import_from_sql_column(self: QGMColumn, sql_column: SQLColumn):
    self.origin_column_id = sql_column.origin_column_id
    self.setwise_column_id = sql_column.setwise_column_id
    self.table_id = sql_column.sql_table.table_id


QGMColumn.import_from_sql_column = qgm_column_import_from_sql_column
