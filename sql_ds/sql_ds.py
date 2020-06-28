from typing import Tuple, List, Optional, Union
from typing_extensions import Literal

CLAUSE_KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
)
JOIN_KEYWORDS = ("join", "on", "as")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")


class SQLBase:
    def __init__(self, parent):
        self.parent = parent
        self.db = parent.db  # db is spider style db

    def infer_from_clause(self):
        ancestor = self
        while ancestor.get_from_clause() is None:
            ancestor = ancestor.parent
        from_clause = ancestor.get_from_clause()
        return from_clause

    def import_from_spider_sql(self, *args):
        pass

    def to_string(self) -> str:
        pass

    def get_from_clause(self):
        return None

    def get_table_of_original_col(self, original_col_id):
        return self.db["column_names"][original_col_id][0]


class SQLValue(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.value = None


class SQLColumn(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.origin_column_id: int = None
        self.setwise_column_id: int = None
        self.is_primary_key: bool = None
        self.sql_table: SQLTable = None
        self.multi_table_ancester = False

    def fill_aux_info(self, from_clause):
        setwise_column_id, tab_id = self._infer_setwise_col_tab_id(from_clause)
        self.setwise_column_id = setwise_column_id
        self._find_parent_table(from_clause, tab_id)
        # TODO fill is_primary_key

    def infer_origin_col_id(self):
        if self.setwise_column_id == 0:
            self.origin_column_id = 0
            return
        # infer origin col id using setwise col id & tab id
        for origin_col_id, (tab_id, col_name) in enumerate(self.db["column_names"]):
            if (
                tab_id == self.sql_table.table_id
                and col_name == self.db["col_set"][self.setwise_column_id]
            ):
                self.origin_column_id = origin_col_id
                return
        assert False, "cannot found origin col if"

    def _infer_setwise_col_tab_id(self, from_clause):
        db = self.db
        from_sql_tables = from_clause.tables_chain
        new_col_id = db["col_set"].index(db["column_names"][self.origin_column_id][1])
        if self.origin_column_id == 0:  # COUNT
            tab_id = self._infer_table_star(from_sql_tables)
        else:
            tab_id = db["column_names"][self.origin_column_id][0]
        return new_col_id, tab_id

    def _find_parent_table(self, from_clause, tab_id):
        def try_find_parent_table():
            for sql_table in from_clause.tables_chain:
                if sql_table.table_id == tab_id:
                    self.sql_table = sql_table
                    return True
            return False

        if tab_id == -1:
            self.sql_table = None
            return

        if len(from_clause.tables_chain) > 1:
            self.multi_table_ancester = True
        if try_find_parent_table():
            return
        from_clause.extend_using_shortest_path(tab_id)
        if try_find_parent_table():
            return
        assert False

    def _infer_table_star(self, from_sql_tables):
        if len(from_sql_tables) == 1:
            return from_sql_tables[0].table_id
        else:
            assert False  # TODO make rules for this!

    def column_name(self):
        return self.db["column_names_original"][self.origin_column_id][1]


class SQLTable(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.table_id: int = None
        # TODO
        self.abbrev_table_id: int = None

    def table_name(self):
        return self.db["table_names_original"][self.table_id]


class SQLColumnWithAgg(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.agg: str = None  # AGG_OPS
        self.distinct: bool = False
        self.sql_column: SQLColumn = None


class SQLLeftHand(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.op: str = "none"  # UNIT_OPS
        self.sql_column_with_agg: SQLColumnWithAgg = None
        self.sql_column_with_agg2: SQLColumnWithAgg = None

    def no_op_and_agg(self):
        return self.op == "none" and self.sql_column_with_agg.agg == "none"


class SQLWhereClause(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.sql_where_clause_one_chain: List[Tuple[str, SQLHavingWhereClauseOne]] = []


class SQLJoinClause(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.column1: SQLColumn = None
        self.column2: SQLColumn = None


class SQLFromClause(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.tables_chain: Optional[List[SQLTable]] = None
        self.join_clauses_chain: Optional[List[SQLJoinClause]] = None
        self.from_subquery: Optional[SQLBySet] = None

    def get_from_clause(self):
        if self.tables_chain is not None:
            return self
        else:
            return self.from_subquery.get_from_clause()

    def extend_using_shortest_path(self, tab_id):  # if success, return True
        if len(self.tables_chain) == 0:
            new_sql_table = SQLTable(self)
            new_sql_table.table_id = tab_id
            new_sql_table.abbrev_table_id = 1
            self.tables_chain += [new_sql_table]
            return True
        candidates = [{"current": tab_id, "visited": [tab_id], "join_paths": [],}]

        def explore_neighbor(candidate):
            explored_candidates = []
            for new_tab_id in range(len(self.db["table_names"])):
                if new_tab_id in candidate["visited"]:
                    continue
                for f, p in self.db["foreign_keys"]:
                    f_tab = self.get_table_of_original_col(f)
                    p_tab = self.get_table_of_original_col(p)
                    if (f_tab == new_tab_id and p_tab == candidate["current"]) or (
                        p_tab == new_tab_id and f_tab == candidate["current"]
                    ):
                        new_sql_join = SQLJoinClause(self)
                        new_sql_join.column1 = SQLColumn(new_sql_join)
                        new_sql_join.column1.origin_column_id = f
                        new_sql_join.column2 = SQLColumn(new_sql_join)
                        new_sql_join.column2.origin_column_id = p
                    elif p_tab == new_tab_id and f_tab == candidate["current"]:
                        new_sql_join = SQLJoinClause(self)
                        new_sql_join.column2 = SQLColumn(new_sql_join)
                        new_sql_join.column2.origin_column_id = f
                        new_sql_join.column1 = SQLColumn(new_sql_join)
                        new_sql_join.column1.origin_column_id = p
                    else:
                        continue

                    new_candidate = {
                        "current": new_tab_id,
                        "visited": [new_tab_id] + candidate["visited"][:],
                        "join_paths": [new_sql_join] + candidate["join_paths"][:],
                    }
                    explored_candidates.append(new_candidate)
            return explored_candidates

        def explore(candidates):
            new_candidates = []
            for candidate in candidates:
                explored_candidates = explore_neighbor(candidate)
                for explored_candidate_one in explored_candidates:
                    from_table_ids = [
                        sql_table.table_id for sql_table in self.tables_chain
                    ]
                    if explored_candidate_one["current"] in from_table_ids:
                        return True, explored_candidate_one
                new_candidates += explored_candidates
            return False, new_candidates

        for i in range(4):  # maximum 4 hop
            res, new_candidates_or_one = explore(candidates)
            if res:
                selected_candidate = new_candidates_or_one
                for new_tab_id in selected_candidate["visited"][1:]:
                    new_sql_table = SQLTable(self)
                    new_sql_table.table_id = new_tab_id
                    new_sql_table.abbrev_table_id = len(self.tables_chain) + 1
                    self.tables_chain += [new_sql_table]
                for join_clause in selected_candidate["join_paths"]:
                    join_clause.column1.fill_aux_info(self)
                    join_clause.column2.fill_aux_info(self)
                self.join_clauses_chain += selected_candidate["join_paths"]
                return True
            else:
                candidates = new_candidates_or_one
        return False


class SQLSelectClause(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.sql_left_hand_list: List[SQLLeftHand] = []
        self.distinct: bool = False


class SQLHavingWhereClauseOne(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.not_op: bool = False
        self.left_hand: SQLLeftHand = None
        self.op: Literal[WHERE_OPS] = "none"
        self.right_hand: Union[SQLValue, SQLWithOrder] = None
        self.right_hand2: Optional[
            Union[SQLValue, SQLWithOrder]
        ] = None  # Only used when op is between


class SQLHavingClause(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.sql_having_clause_one_chain: List[Tuple[str, SQLHavingWhereClauseOne]] = []


class SQLGroupClause(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        # TODO currently we don't support agg / multi columns
        self.sql_column: SQLColumn = None
        self.sql_having_clause: SQLHavingClause = None


class SQLWithGroup(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.sql_select_clause: SQLSelectClause = None
        self.sql_from_clause: SQLFromClause = None
        self.sql_where_clause: SQLWhereClause = None
        self.sql_group_clause: Optional[SQLGroupClause] = None

    def get_from_clause(self):
        return self.sql_from_clause.get_from_clause()


class SQLBySet(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        # Chaining via set operator (e.g. intersect, union, ...)
        self.sql_with_group_chain: List[Tuple[str, SQLWithGroup]] = []

    def get_from_clause(self):
        if len(self.sql_with_group_chain) == 1:
            return self.sql_with_group_chain[0][1].get_from_clause()
        else:
            return None

    def has_set_operator(self):
        return len(self.sql_with_group_chain) > 1


class SQLOrderClause(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.sql_column_with_agg_list: List[SQLLeftHand] = []
        self.limit_num: Optional[int] = None
        self.sql_order_direction: Literal[ORDER_OPS] = "desc"


class SQLWithOrder(SQLBase):
    def __init__(self, parent):
        SQLBase.__init__(self, parent)
        self.sql_by_set: SQLBySet = None
        self.sql_order_clause: Optional[SQLOrderClause] = None

    def get_from_clause(self):
        return self.sql_by_set.get_from_clause()

    def get_select_clauses(self):
        select_clauses = []
        for (set_op, sql_with_group,) in self.sql_by_set.sql_with_group_chain:
            # TODO currently support no op in select
            for left_hand in sql_with_group.sql_select_clause.sql_left_hand_list:
                select_clauses.append(left_hand.sql_column_with_agg)
        return select_clauses

    def get_where_clauses(self):
        conj_where_clauses = []
        for (set_op, sql_with_group,) in self.sql_by_set.sql_with_group_chain:
            # TODO currently support no op in select
            for (
                conj,
                where_clause,
            ) in sql_with_group.sql_where_clause.sql_where_clause_one_chain:
                conj_where_clauses.append((conj, where_clause))
        return conj_where_clauses


class SQLDataStructure:
    def __init__(self, db=None):
        self.db = db
        self.sql_with_order: SQLWithOrder = None

    @classmethod
    def import_from_spider_sql(cls, sql, db):
        sql_data_structure = SQLDataStructure(db)
        sql_data_structure.sql_with_order = SQLWithOrder(sql_data_structure)
        sql_data_structure.sql_with_order.import_from_spider_sql(sql)
        return sql_data_structure

    def import_from_qgm(self, qgm):
        # implemented in sql_ds_import_from_qgm.py
        pass

    def to_string(self):
        sql_query = self.sql_with_order.to_string()
        return sql_query

    # Implemented in sql_ds_categorize.py
    def has_table_without_col(self):
        pass

    def has_aggregator_in_main_select(self):
        pass

    def has_subquery_where(self):
        pass

    def has_subquery_from(self):
        pass

    def has_subquery_select(self):
        pass

    def has_subquery(self):
        pass

    def has_set_operator(self):
        pass

    def has_grouping(self):
        pass

    def has_ordering(self):
        pass
