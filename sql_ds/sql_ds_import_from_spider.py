from sql_ds.sql_ds import (
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
    WHERE_OPS,
    UNIT_OPS,
    AGG_OPS,
)


def sql_with_order_import_from_spider_sql(self: SQLWithOrder, sql):
    self.sql_by_set = SQLBySet(self)
    self.sql_by_set.import_from_spider_sql(sql)
    if sql["orderBy"]:
        self.sql_order_clause = SQLOrderClause(self)
        self.sql_order_clause.import_from_spider_sql(sql["orderBy"], sql["limit"])


def sql_order_clause_import_from_spider_sql(
    self: SQLOrderClause, sql_orderBy, sql_limit
):
    for val_unit in sql_orderBy[1]:
        new_sql_left_hand = SQLLeftHand(self)
        new_sql_left_hand.import_from_spider_sql(val_unit)
        self.sql_column_with_agg_list.append(new_sql_left_hand)

    self.limit_num = sql_limit
    self.sql_order_direction = sql_orderBy[0]


def sql_by_set_import_from_spider_sql(self: SQLBySet, sql):
    # For now we assume at most one set operator exist
    sql_with_group = SQLWithGroup(self)
    sql_with_group.import_from_spider_sql(sql)
    self.sql_with_group_chain.append(("", sql_with_group))
    if sql["intersect"]:
        sql_with_group = SQLWithGroup(self)
        sql_with_group.import_from_spider_sql(sql["intersect"])
        self.sql_with_group_chain.append(("intersect", sql_with_group))
        assert sql["intersect"]["intersect"] is None
        assert sql["intersect"]["union"] is None
        assert sql["intersect"]["except"] is None
        assert sql["union"] is None
        assert sql["except"] is None
    elif sql["union"]:
        sql_with_group = SQLWithGroup(self)
        sql_with_group.import_from_spider_sql(sql["union"])
        self.sql_with_group_chain.append(("union", sql_with_group))
        assert sql["union"]["intersect"] is None
        assert sql["union"]["union"] is None
        assert sql["union"]["except"] is None
        assert sql["intersect"] is None
        assert sql["except"] is None
    elif sql["except"]:
        sql_with_group = SQLWithGroup(self)
        sql_with_group.import_from_spider_sql(sql["except"])
        self.sql_with_group_chain.append(("except", sql_with_group))
        assert sql["except"]["intersect"] is None
        assert sql["except"]["union"] is None
        assert sql["except"]["except"] is None
        assert sql["intersect"] is None
        assert sql["union"] is None


def sql_with_group_import_from_spider_sql(self: SQLWithGroup, sql):
    self.sql_from_clause = SQLFromClause(self)
    self.sql_from_clause.import_from_spider_sql(sql["from"])
    self.sql_select_clause = SQLSelectClause(self)
    self.sql_select_clause.import_from_spider_sql(sql["select"])
    self.sql_where_clause = SQLWhereClause(self)
    self.sql_where_clause.import_from_spider_sql(sql["where"])
    if len(sql["groupBy"]) > 0:
        self.sql_group_clause = SQLGroupClause(self)
        self.sql_group_clause.import_from_spider_sql(sql["groupBy"], sql["having"])


def sql_group_clause_import_from_spider_sql(
    self: SQLGroupClause, sql_groupby, sql_having
):
    col_unit = sql_groupby[0]
    self.sql_column = SQLColumn(self)
    self.sql_column.import_from_spider_sql(col_unit[1])
    self.sql_having_clause = SQLHavingClause(self)
    self.sql_having_clause.import_from_spider_sql(sql_having)


def sql_having_clause_import_from_spider_sql(self: SQLHavingClause, sql_having):
    for idx in range(0, len(sql_having), 2):
        sql_having_clause_one = SQLHavingWhereClauseOne(self)
        sql_having_clause_one.import_from_spider_sql(sql_having[idx])
        if idx == 0:
            self.sql_having_clause_one_chain.append(("", sql_having_clause_one))
        else:
            self.sql_having_clause_one_chain.append(
                (sql_having[idx - 1], sql_having_clause_one)  # and / or
            )


def sql_having_where_clause_one_import_from_spider_sql(
    self: SQLHavingWhereClauseOne, sql_cond_unit
):
    def parse_val_or_subquery(val):
        if isinstance(val, dict):  # subquery
            val_or_subquery = SQLWithOrder(self)
            val_or_subquery.import_from_spider_sql(val)
        else:
            val_or_subquery = SQLValue(self)
            val_or_subquery.import_from_spider_sql(val)
        return val_or_subquery

    self.not_op = sql_cond_unit[0]

    self.left_hand = SQLLeftHand(self)
    self.left_hand.import_from_spider_sql(sql_cond_unit[2])
    op_id = sql_cond_unit[1]
    self.op = WHERE_OPS[op_id]
    val1 = sql_cond_unit[3]
    self.right_hand = parse_val_or_subquery(val1)
    val2 = sql_cond_unit[4]
    if val2 is not None:
        assert self.op == "between"
        self.right_hand2 = parse_val_or_subquery(val2)


def sql_select_clause_import_from_spider_sql(self: SQLSelectClause, sql_select):
    self.distinct = sql_select[0]
    for agg_id, val_unit in sql_select[1]:
        sql_left_hand = SQLLeftHand(self)
        sql_left_hand.import_from_spider_sql(val_unit, agg_id)
        self.sql_left_hand_list.append(sql_left_hand)


def sql_from_clause_import_from_spider_sql(self: SQLFromClause, sql_from):
    if sql_from["table_units"][0][0] == "sql":  # nested from
        assert len(sql_from["table_units"]) == 1
        assert len(sql_from["conds"]) == 0
        self.from_subquery = SQLBySet(self)
        self.from_subquery.import_from_spider_sql(sql_from["table_units"][0][1])
    else:
        self.tables_chain = []
        self.join_clauses_chain = []
        for idx, table_unit in enumerate(sql_from["table_units"]):
            assert table_unit[0] == "table_unit"
            sql_table = SQLTable(self)
            abbrev_table_id = idx + 1
            sql_table.import_from_spider_sql(table_unit[1], abbrev_table_id)
            self.tables_chain.append(sql_table)
        for idx in range(0, len(sql_from["conds"]), 2):
            if idx != 0:
                assert sql_from["conds"][idx - 1] == "and"
            cond = sql_from["conds"][idx]

            sql_join_clause = SQLJoinClause(self)
            sql_join_clause.import_from_spider_sql(cond)
            self.join_clauses_chain.append(sql_join_clause)


def sql_join_clause_import_from_spider_sql(self: SQLJoinClause, sql_cond):
    assert sql_cond[0] is False
    assert sql_cond[1] == 2
    val_unit = sql_cond[2]
    assert val_unit[0] == 0
    col_unit_1 = val_unit[1]
    col1 = col_unit_1[1]
    col_unit_2 = sql_cond[3]
    col2 = col_unit_2[1]
    assert sql_cond[4] is None
    self.column1 = SQLColumn(self)
    self.column1.import_from_spider_sql(col1)
    self.column2 = SQLColumn(self)
    self.column2.import_from_spider_sql(col2)


def sql_where_clause_import_from_spider_sql(self: SQLWhereClause, sql_condition):
    for idx in range(0, len(sql_condition), 2):
        sql_where_clause_one = SQLHavingWhereClauseOne(self)
        sql_where_clause_one.import_from_spider_sql(sql_condition[idx])
        if idx == 0:
            self.sql_where_clause_one_chain.append(("", sql_where_clause_one))
        else:
            self.sql_where_clause_one_chain.append(
                (sql_condition[idx - 1], sql_where_clause_one)
            )


def sql_left_hand_import_from_spider_sql(self: SQLLeftHand, sql_val_unit, agg=None):
    self.op = UNIT_OPS[sql_val_unit[0]]
    self.sql_column_with_agg = SQLColumnWithAgg(self)
    self.sql_column_with_agg.import_from_spider_sql(sql_val_unit[1], agg)
    if self.op != "none":
        self.sql_column_with_agg2 = SQLColumnWithAgg(self)
        self.sql_column_with_agg2.import_from_spider_sql(sql_val_unit[2], agg)


def sql_column_with_agg_import_from_spider_sql(
    self: SQLColumnWithAgg, sql_col_unit, agg=None
):
    if agg:
        self.agg = AGG_OPS[agg]

    else:
        self.agg = AGG_OPS[sql_col_unit[0]]
    self.distinct = sql_col_unit[2]
    self.sql_column = SQLColumn(self)
    self.sql_column.import_from_spider_sql(sql_col_unit[1])


def sql_table_import_from_spider_sql(self: SQLTable, table_id, abbrev_table_id):
    self.table_id = table_id
    self.abbrev_table_id = abbrev_table_id


def sql_column_import_from_spider_sql(self: SQLColumn, col_id):
    self.origin_column_id = col_id
    from_clause = self.infer_from_clause()
    self.fill_aux_info(from_clause)


def sql_value_import_from_spider_sql(self: SQLValue, value):
    if isinstance(value, float) and value - float(int(value)) == 0.0:
        value = int(value)
    self.value = value


SQLWithOrder.import_from_spider_sql = sql_with_order_import_from_spider_sql
SQLOrderClause.import_from_spider_sql = sql_order_clause_import_from_spider_sql
SQLBySet.import_from_spider_sql = sql_by_set_import_from_spider_sql
SQLWithGroup.import_from_spider_sql = sql_with_group_import_from_spider_sql
SQLGroupClause.import_from_spider_sql = sql_group_clause_import_from_spider_sql
SQLHavingClause.import_from_spider_sql = sql_having_clause_import_from_spider_sql
SQLHavingWhereClauseOne.import_from_spider_sql = (
    sql_having_where_clause_one_import_from_spider_sql
)
SQLSelectClause.import_from_spider_sql = sql_select_clause_import_from_spider_sql
SQLFromClause.import_from_spider_sql = sql_from_clause_import_from_spider_sql
SQLJoinClause.import_from_spider_sql = sql_join_clause_import_from_spider_sql
SQLWhereClause.import_from_spider_sql = sql_where_clause_import_from_spider_sql
SQLLeftHand.import_from_spider_sql = sql_left_hand_import_from_spider_sql
SQLColumnWithAgg.import_from_spider_sql = sql_column_with_agg_import_from_spider_sql
SQLTable.import_from_spider_sql = sql_table_import_from_spider_sql
SQLColumn.import_from_spider_sql = sql_column_import_from_spider_sql
SQLValue.import_from_spider_sql = sql_value_import_from_spider_sql
