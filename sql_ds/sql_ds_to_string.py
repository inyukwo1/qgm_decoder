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
)


def beutify(sql_query):
    sql_query = " ".join(sql_query.split())
    sql_query = sql_query.replace(";", "")
    sql_query = sql_query.replace(" ASC", "")
    sql_query = sql_query.replace('"', "'")
    sql_query = sql_query.replace("COUNT ", "count")
    sql_query = sql_query.replace("SUM ", "sum")
    sql_query = sql_query.replace("MAX ", "max")
    sql_query = sql_query.replace("MIN ", "min")
    sql_query = sql_query.replace("( ", "(")
    sql_query = sql_query.replace(" )", ")")
    sql_query = sql_query.replace(" , ", ", ")
    sql_query = sql_query.replace("IN(SELECT", "IN (SELECT")
    sql_query = sql_query.replace("AVG (", "AVG(")

    start_pos = 0
    sql_query_split = sql_query.split(" ")
    while "ON" in sql_query_split[start_pos:]:
        on_pos = sql_query_split.index("ON", start_pos)
        try:
            if int(sql_query_split[on_pos + 1][1]) > int(
                sql_query_split[on_pos + 3][1]
            ):
                (sql_query_split[on_pos + 1], sql_query_split[on_pos + 3],) = (
                    sql_query_split[on_pos + 3],
                    sql_query_split[on_pos + 1],
                )
        except:
            pass

        start_pos = on_pos + 1
    for q_idx in range(len(sql_query_split)):
        if "distinct(" in sql_query_split[q_idx]:
            new_str = sql_query_split[q_idx].replace("distinct(", "DISTINCT ")[:-1]
            sql_query_split[q_idx] = new_str
    sql_query = " ".join(sql_query_split)
    return sql_query


def sql_with_order_to_string(self: SQLWithOrder):
    sql_query = self.sql_by_set.to_string()
    if self.sql_order_clause:
        sql_query += " {}".format(self.sql_order_clause.to_string())
    sql_query = beutify(sql_query)
    return sql_query


def sql_order_clause_to_string(self: SQLOrderClause):
    column_list_string = ", ".join(
        [sql_left_hand.to_string() for sql_left_hand in self.sql_column_with_agg_list]
    )
    sql_query = "ORDER BY {}".format(column_list_string)
    if self.sql_order_direction == "desc":
        sql_query += " DESC"
    if self.limit_num is not None:
        sql_query += " LIMIT {}".format(self.limit_num)
    return sql_query


def sql_by_set_to_string(self: SQLBySet):
    sql_query = ""
    for set_operator, sql_with_group in self.sql_with_group_chain:
        sql_query += " {} {}".format(set_operator.upper(), sql_with_group.to_string())
    return sql_query


def sql_with_group_to_string(self: SQLWithGroup):
    sql_query = "{} {} {}".format(
        self.sql_select_clause.to_string(),
        self.sql_from_clause.to_string(),
        self.sql_where_clause.to_string(),
    )
    if self.sql_group_clause is not None:
        sql_query += " {}".format(self.sql_group_clause.to_string())
    return sql_query


def sql_group_clause_to_string(self: SQLGroupClause):
    sql_query = "GROUP BY {} {}".format(
        self.sql_column.to_string(), self.sql_having_clause.to_string()
    )
    return sql_query


def sql_having_clause_to_string(self: SQLHavingClause):
    if len(self.sql_having_clause_one_chain) == 0:
        return ""
    sql_query = "HAVING "
    for conjunction, sql_having_one in self.sql_having_clause_one_chain:
        sql_query += " {} {}".format(conjunction.upper(), sql_having_one.to_string())
    return sql_query


def sql_having_where_clause_one_to_string(self: SQLHavingWhereClauseOne):
    if isinstance(self.right_hand, SQLValue):
        right_hand_string = self.right_hand.to_string()
    else:
        right_hand_string = "({})".format(self.right_hand.to_string())
    if self.op == "between":

        if isinstance(self.right_hand2, SQLValue):
            right_hand_string2 = self.right_hand2.to_string()
        else:
            right_hand_string2 = "({})".format(self.right_hand2.to_string())
        return "{} BETWEEN {} AND {}".format(
            self.left_hand.to_string(), right_hand_string, right_hand_string2,
        )
    else:
        if self.not_op:
            return "{} NOT {} {}".format(
                self.left_hand.to_string(), self.op.upper(), right_hand_string,
            )
        else:
            return "{} {} {}".format(
                self.left_hand.to_string(), self.op.upper(), right_hand_string,
            )


def sql_select_clause_to_string(self: SQLSelectClause):
    if self.distinct:
        sql_query = "SELECT DISTINCT "
    else:
        sql_query = "SELECT "
    return sql_query + "{}".format(
        " , ".join(
            [sql_left_hand.to_string() for sql_left_hand in self.sql_left_hand_list]
        )
    )


def sql_from_clause_to_string(self: SQLFromClause):
    if self.from_subquery is not None:
        return "FROM ({})".format(self.from_subquery.to_string())
    else:
        if len(self.tables_chain) == 1:
            return "FROM {}".format(self.tables_chain[0].to_string(False))
        sql_query = "FROM"
        for idx, sql_table in enumerate(self.tables_chain):
            if idx != 0:
                sql_query += " JOIN {} ON {}".format(
                    sql_table.to_string(True),
                    self.join_clauses_chain[idx - 1].to_string(),
                )
            else:
                sql_query += " {}".format(sql_table.to_string(True))
        # sql_query += " ON {}".format(
        #     " AND ".join(
        #         [join_clause.to_string() for join_clause in self.join_clauses_chain]
        #     )
        # )

        return sql_query


def sql_join_clause_to_string(self: SQLJoinClause):
    return "{} = {}".format(self.column1.to_string(), self.column2.to_string())


def sql_where_clause_to_string(self: SQLWhereClause):
    if len(self.sql_where_clause_one_chain) == 0:
        return ""
    sql_query = "WHERE "
    for conjunction, sql_having_one in self.sql_where_clause_one_chain:
        sql_query += " {} {}".format(conjunction.upper(), sql_having_one.to_string())
    return sql_query


def sql_left_hand_to_string(self: SQLLeftHand):
    if self.op == "none":
        return self.sql_column_with_agg.to_string()
    else:
        return "{} {} {}".format(
            self.sql_column_with_agg.to_string(),
            self.op,
            self.sql_column_with_agg2.to_string(),
        )


def sql_column_with_agg_to_string(self: SQLColumnWithAgg):
    if self.agg == "none":
        if self.distinct:
            return "DISTINCT {}".format(self.sql_column.to_string())
        else:
            return self.sql_column.to_string()
    else:
        if self.distinct:
            return "{}(DISTINCT {})".format(self.agg, self.sql_column.to_string())

        else:
            return "{}({})".format(self.agg, self.sql_column.to_string())


def sql_table_to_string(self: SQLTable, with_abb):
    table_name = self.table_name()
    if with_abb:
        return "{} AS T{}".format(table_name, self.abbrev_table_id)
    else:
        return table_name


def sql_column_to_string(self: SQLColumn):
    if (
        not self.multi_table_ancester
        or self.sql_table is None
        or self.origin_column_id == 0
    ):
        return self.column_name()
    else:
        return "T{}.{}".format(self.sql_table.abbrev_table_id, self.column_name())


def sql_value_to_string(self: SQLValue):
    return str(self.value)


SQLWithOrder.to_string = sql_with_order_to_string
SQLOrderClause.to_string = sql_order_clause_to_string
SQLBySet.to_string = sql_by_set_to_string
SQLWithGroup.to_string = sql_with_group_to_string
SQLGroupClause.to_string = sql_group_clause_to_string
SQLHavingClause.to_string = sql_having_clause_to_string
SQLHavingWhereClauseOne.to_string = sql_having_where_clause_one_to_string
SQLSelectClause.to_string = sql_select_clause_to_string
SQLFromClause.to_string = sql_from_clause_to_string
SQLJoinClause.to_string = sql_join_clause_to_string
SQLWhereClause.to_string = sql_where_clause_to_string
SQLLeftHand.to_string = sql_left_hand_to_string
SQLColumnWithAgg.to_string = sql_column_with_agg_to_string
SQLTable.to_string = sql_table_to_string
SQLColumn.to_string = sql_column_to_string
SQLValue.to_string = sql_value_to_string
