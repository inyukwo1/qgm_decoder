from typing import List, Tuple


OPS = (
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "like",
    "is",
    "exists",
    "in",
    "not in",
)


AGGS = ("none", "max", "min", "count", "sum", "avg")

PROBLEM_ACTIONS = [
    ("select_col_num", ["1", "2", "3", "4", "5"]),
    ("projection_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("predicate_col_done", ["done", "do"]),
    ("predicate_and_or", ["and", "or"]),
    (
        "predicate_op",
        [
            "between",
            "=",
            ">",
            "<",
            ">=",
            "<=",
            "!=",
            "like",
            "is",
            "exists",
            "in",
            "not in",
        ],
    ),
    ("value_or_subquery", ["value", "subquery"]),
    ("subquery_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("subquery_predicate_col_done", ["done", "do"]),
    ("subquery_predicate_and_or", ["and", "or"]),
    (
        "subquery_predicate_op",
        [
            "between",
            "=",
            ">",
            "<",
            ">=",
            "<=",
            "!=",
            "like",
            "is",
            "exists",
            "in",
            "not in",
        ],
    ),
    ("C", []),
    ("T", []),
]


class QGMBase:
    def __init__(self, parent):
        self.parent = parent
        self.db = parent.db


class QGMColumn(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.origin_column_id: int = None
        self.setwise_column_id: int = None
        self.is_primary_key: bool = None
        self.table_id: int = None

    def import_from_sql_column(self, sql_column):
        # implemented in qgm_import_from_sql_ds.py
        pass


class QGMLocalPredicate(QGMBase):
    def __init__(
        self, parent, op, col_pointer, value_or_subquery
    ):  # can be tuple if op == between
        QGMBase.__init__(self, parent)
        self.op = op
        self.col_pointer = col_pointer
        self.value_or_subquery = value_or_subquery

    def find_col(self):
        predicatebox = self.parent
        assert isinstance(predicatebox, QGMPredicateBox)
        basebox = predicatebox.parent
        assert isinstance(basebox, QGMBaseBox)
        return basebox.predicate_cols[self.col_pointer]


class QGMPredicateBox(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.local_predicates: List[Tuple[str, QGMLocalPredicate]] = []

    def add_local_predicate(self, conj, op, col_pointer, value_or_subquery):
        new_local_predicate = QGMLocalPredicate(
            self, op, col_pointer, value_or_subquery
        )
        self.local_predicates.append((conj, new_local_predicate))


class QGMProjection(QGMBase):
    def __init__(self, parent, agg, col_pointer):
        QGMBase.__init__(self, parent)
        self.agg = agg
        self.col_pointer = col_pointer

    def find_col(self):
        projectionbox = self.parent
        assert isinstance(projectionbox, QGMProjectionBox)
        basebox = projectionbox.parent
        assert isinstance(basebox, QGMBaseBox)
        return basebox.select_cols[self.col_pointer]


class QGMProjectionBox(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.projections = []

    def add_projection(self, agg, col_pointer):
        new_projection = QGMProjection(self, agg, col_pointer)
        self.projections.append(new_projection)


class QGMBaseBox(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.select_cols: List[QGMColumn] = []
        self.predicate_cols: List[QGMColumn] = []
        self.predicate_box = QGMPredicateBox(self)
        self.projection_box = QGMProjectionBox(self)

    def add_select_column(self, qgm_column: QGMColumn, agg):
        new_col_pointer = len(self.select_cols)
        self.select_cols.append(qgm_column)
        self.projection_box.add_projection(agg, new_col_pointer)

    def add_local_predicate(self, conj, qgm_column: QGMColumn, op, value_or_subquery):
        new_col_pointer = len(self.predicate_cols)
        self.predicate_cols.append(qgm_column)
        self.predicate_box.add_local_predicate(
            conj, op, new_col_pointer, value_or_subquery
        )


class QGM:
    def __init__(self, db):  # we use spider-like db
        self.is_gold = None
        self.db = db
        self.pointer = None
        self.prev_action_id = None
        self.base_boxes = QGMBaseBox(self)

    @classmethod
    def import_from_sql_ds(cls):
        pass

    def export_to_text(self):
        pass

    def fast_comparison(self, other_qgm: "QGM"):
        pass
