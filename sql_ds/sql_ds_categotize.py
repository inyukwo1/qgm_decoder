from sql_ds.sql_ds import SQLDataStructure, SQLWithOrder


def has_table_without_col(self: SQLDataStructure):
    # TODO process order clause
    for (
        set_op,
        sql_with_group,
    ) in self.sql_with_order.sql_by_set.sql_with_group_chain:
        from_clause = sql_with_group.sql_from_clause
        table_ids = [sqltable.table_id for sqltable in from_clause.tables_chain]
        checked_table = [False for _ in range(len(from_clause.tables_chain))]

        for sql_left_hand in sql_with_group.sql_select_clause.sql_left_hand_list:
            table_id = sql_left_hand.sql_column_with_agg.sql_column.sql_table.table_id
            checked_table[table_ids.index(table_id)] = True
            if sql_left_hand.sql_column_with_agg2 is not None:
                table_id2 = (
                    sql_left_hand.sql_column_with_agg2.sql_column.sql_table.table_id
                )
                checked_table[table_ids.index(table_id2)] = True
        for (
            conj,
            sql_where_clause,
        ) in sql_with_group.sql_where_clause.sql_where_clause_one_chain:
            table_id = (
                sql_where_clause.left_hand.sql_column_with_agg.sql_column.sql_table.table_id
            )
            checked_table[table_ids.index(table_id)] = True

        if sql_with_group.sql_group_clause is not None:
            table_id = sql_with_group.sql_group_clause.sql_column.sql_table.table_id
            checked_table[table_ids.index(table_id)] = True
            if sql_with_group.sql_group_clause.sql_having_clause is not None:
                for (
                    conj,
                    sql_where_clause,
                ) in (
                    sql_with_group.sql_group_clause.sql_having_clause.sql_having_clause_one_chain
                ):
                    table_id = (
                        sql_where_clause.left_hand.sql_column_with_agg.sql_column.sql_table.table_id
                    )
                    checked_table[table_ids.index(table_id)] = True
        if False in checked_table:
            return True

    return False


def has_aggregator_in_main_select(self: SQLDataStructure):
    for (
        set_op,
        sql_with_group,
    ) in self.sql_with_order.sql_by_set.sql_with_group_chain:
        for sql_left_hand in sql_with_group.sql_select_clause.sql_left_hand_list:
            if sql_left_hand.sql_column_with_agg.agg != "none":
                return True
    return False


def has_subquery_where(self: SQLDataStructure):
    for (
        set_op,
        sql_with_group,
    ) in self.sql_with_order.sql_by_set.sql_with_group_chain:
        for (
            conj,
            where_clause,
        ) in sql_with_group.sql_where_clause.sql_where_clause_one_chain:
            if isinstance(where_clause.right_hand, SQLWithOrder):
                return True
            elif isinstance(where_clause.right_hand2, SQLWithOrder):
                return True
    return False


def has_subquery_from(self: SQLDataStructure):
    for (
        set_op,
        sql_with_group,
    ) in self.sql_with_order.sql_by_set.sql_with_group_chain:
        if sql_with_group.sql_from_clause.from_subquery is not None:
            return True


def has_subquery_select(self: SQLDataStructure):
    return False  # TODO for now we don't support


def has_subquery(self: SQLDataStructure):
    return (
        self.has_subquery_where()
        or self.has_subquery_from()
        or self.has_subquery_select()
    )


def has_set_operator(self: SQLDataStructure):

    for (
        set_op,
        sql_with_group,
    ) in self.sql_with_order.sql_by_set.sql_with_group_chain:
        for (
            conj,
            where_clause,
        ) in sql_with_group.sql_where_clause.sql_where_clause_one_chain:
            if where_clause.op in {"not in", "in"}:
                return True
    return self.sql_with_order.sql_by_set.has_set_operator()


def has_grouping(self: SQLDataStructure):
    # TODO not checking grouping in subquery
    for (
        set_op,
        sql_with_group,
    ) in self.sql_with_order.sql_by_set.sql_with_group_chain:
        if sql_with_group.sql_group_clause is not None:
            return True
    return False


def has_ordering(self: SQLDataStructure):
    # TODO not checking ordering in subquery
    return self.sql_with_order.sql_order_clause is not None


SQLDataStructure.has_table_without_col = has_table_without_col
SQLDataStructure.has_aggregator_in_main_select = has_aggregator_in_main_select
SQLDataStructure.has_subquery_where = has_subquery_where
SQLDataStructure.has_subquery_from = has_subquery_from
SQLDataStructure.has_subquery_select = has_subquery_select
SQLDataStructure.has_subquery = has_subquery
SQLDataStructure.has_set_operator = has_set_operator
SQLDataStructure.has_grouping = has_grouping
SQLDataStructure.has_ordering = has_ordering
