from qgm.qgm import QGM, QGMColumn
from sql_ds.sql_ds import (
    SQLDataStructure,
    SQLColumn,
    SQLHavingWhereClauseOne,
    SQLColumnWithAgg,
)
from typing import List, Tuple


def qgm_import_from_sql_ds(sql_ds: SQLDataStructure):
    if (
        sql_ds.has_subquery()
        or sql_ds.has_set_operator()
        or sql_ds.has_grouping()
        or sql_ds.has_ordering()
        or sql_ds.has_table_without_col()
    ):  # TODO currently not supporting conditions
        return None
    new_qgm = QGM(sql_ds.db, True)
    select_clause_columns: List[SQLColumnWithAgg] = sql_ds.get_select_clauses()
    predicates: List[Tuple[str, SQLHavingWhereClauseOne]] = sql_ds.get_where_clauses()
    for sql_column_with_agg in select_clause_columns:
        new_column = QGMColumn(new_qgm.base_boxes)
        new_column.import_from_sql_column(sql_column_with_agg.sql_column)
        new_qgm.base_boxes.add_select_column(new_column, sql_column_with_agg.agg)

    for conj, prediate in predicates:
        new_column = QGMColumn(new_qgm.base_boxes)
        if (
            not prediate.left_hand.no_op_and_agg()
        ):  # TODO currently we process just simple cols
            return None
        new_column.import_from_sql_column(
            prediate.left_hand.sql_column_with_agg.sql_column
        )
        if prediate.not_op:
            assert prediate.op == "in"
            op = "not in"
        else:
            op = prediate.op
        if prediate.op == "between":
            value = (prediate.right_hand.value, prediate.right_hand2.value)
        else:
            value = prediate.right_hand.value
        new_qgm.base_boxes.add_local_predicate(conj, new_column, op, value)
    return new_qgm


def qgm_column_import_from_sql_column(self: QGMColumn, sql_column: SQLColumn):
    self.origin_column_id = sql_column.origin_column_id
    self.setwise_column_id = sql_column.setwise_column_id
    self.is_primary_key = sql_column.is_primary_key
    self.table_id = sql_column.sql_table.table_id


QGMColumn.import_from_sql_column = qgm_column_import_from_sql_column
