from typing import List, Tuple, Optional


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


class GlobalPredicate:
    def __init__(self):
        self.type = None
        self.op = None
        self.left = None
        self.right = None


class LocalPredicate:
    def __init__(self):
        self.left = None
        self.op = None
        self.right = None


class Quantifier:
    def __init__(self):
        self.type = None
        self.ref = None


class QGMBox:
    def __init__(self):
        self.quantifiers = []
        self.db = None


class QGMBox_SELECT_GROUPBY(QGMBox):
    def __init__(self):
        super(QGMBox_SELECT_GROUPBY, self).__init__()
        self.head = []  # (agg, QGMBox_column)
        self.quantifiers = []
        self.global_predicates = []
        self.local_predicates = []  # (conj, LocalPredicate)
        self.group_columns = []

    @@property
    def is_group(self):
        return len(self.group_columns) > 0


class QGMBox_SETOP(QGMBox):
    def __init__(self):
        super(QGMBox_SETOP, self).__init__()
        self.op = None
        self.left = None
        self.right = None


class QGMBox_Column(QGMBox):
    def __init__(self):
        super(QGMBox_Column, self).__init__()
        self.column_id = None


class QGMBox_Table(QGMBox):
    def __init__(self):
        super(QGMBox_Table, self).__init__()
        self.head_cols: List[QGMBox_Column] = []
        self.table_id = None


class QGM:
    def __init__(self, db, is_gold):  # we use spider-like db
        self.is_gold = is_gold
        self.db = db
        self.refs = []
        self.table_boxes = []
        self.top_box = None

    def __eq__(self, other):
        for ref, other_ref in zip(self.refs, other.refs):
            if ref != other_ref:
                return False
        return self.top_box == other.top_box

    @classmethod
    def import_from_sql(cls):
        pass

    def export_to_sql(self):
        pass

    def fast_comparison(self, other_qgm: "QGM"):
        pass

    def qgm_rewrite(self):
        pass

    def qgm_simplification(self):
        pass

    def to_action_sequence(self):
        pass
