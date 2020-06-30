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

SYMBOL_ACTIONS = [
    ("select_col_num", ["1", "2", "3", "4", "5"]),
    ("projection_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("predicate_col_done", ["done", "do"]),
    ("predicate_conj", ["and", "or"]),
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
    ("op_done", ["done", "do"]),
    ("subquery_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("subquery_op_done", ["done", "do"]),
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
    ("orderby_exists", ["no", "yes"]),
    ("orderby_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("orderby_direction", ["asc", "desc"]),
    ("groupby_exists", ["no", "yes"]),
    ("having_exists", ["no", "yes"]),
    ("having_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("C", []),
    ("T", []),
]


class QGMBase:
    def __init__(self, parent):
        self.parent = parent
        self.db = parent.db

    def find_base_box(self):
        ancestor = self.parent
        while not isinstance(ancestor, QGMBaseBox):
            ancestor = ancestor.parent
        return ancestor

    def find_qgm_root(self):
        ancestor = self.parent
        while not isinstance(ancestor, QGM):
            ancestor = ancestor.parent
        return ancestor


class QGMColumn(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.origin_column_id: int = None
        self.setwise_column_id: int = None
        self.is_primary_key: bool = None
        self.table_id: int = None

    def __eq__(self, other):
        return (
            self.origin_column_id == other.origin_column_id
            and self.setwise_column_id == other.setwise_column_id
            and self.is_primary_key == other.is_primary_key
            and self.table_id == other.table_id
        )

    def infer_origin_column_id(self):
        if self.setwise_column_id == 0:
            self.origin_column_id = 0
            return
        col_name = self.db["col_set"][self.setwise_column_id]
        for idx, (tab_id, ori_col_name) in enumerate(self.db["column_names"]):
            if tab_id == self.table_id and col_name == ori_col_name:
                self.origin_column_id = idx
                return
        assert False

    def import_from_sql_column(self, sql_column):
        # implemented in qgm_import_from_sql_ds.py
        pass


class QGMSubqueryBox(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.projection: QGMProjection = None
        self.local_predicates: List[Tuple[str, QGMLocalPredicate]] = []
        self.groupby_box: QGMGroupbyBox = None
        self.orderby_box: QGMOrderbyBox = None

    def __eq__(self, other):
        for (my_conj, my_predicate), (other_conj, other_predicate) in zip(
            self.local_predicates, other.local_predicates
        ):
            if my_conj != other_conj or my_predicate != other_predicate:
                return False
        return self.projection == other.projection

    def set_projection(self, qgm_column, agg):
        new_col_pointer = self.find_base_box().add_predicate_column_returning_col_pointer(  # TODO projection? predicate?
            qgm_column
        )
        new_projection = QGMProjection(self, agg, new_col_pointer)
        self.projection = new_projection

    def add_local_predicate(self, conj, qgm_column: QGMColumn, op, value_or_subquery):
        new_col_pointer = self.find_base_box().add_predicate_column_returning_col_pointer(
            qgm_column
        )
        self.add_local_predicate_using_base_col(
            conj, op, new_col_pointer, value_or_subquery
        )

    def add_local_predicate_using_base_col(
        self, conj, op, col_pointer, value_or_subquery
    ):
        new_local_predicate = QGMLocalPredicate(
            self, op, col_pointer, value_or_subquery
        )
        self.local_predicates.append((conj, new_local_predicate))

    def add_orderby(self, direction, agg, limit_num, qgm_column: QGMColumn):
        new_col_pointer = self.find_base_box().add_predicate_column_returning_col_pointer(
            qgm_column
        )
        self.add_orderby_using_base_col(direction, agg, limit_num, new_col_pointer)

    def add_orderby_using_base_col(self, direction, agg, limit_num, col_pointer):
        new_orderby_box = QGMOrderbyBox(self, direction, agg, limit_num, col_pointer)
        self.orderby_box = new_orderby_box

    def add_groupby(
        self,
        qgm_column: QGMColumn,
        having_column: QGMColumn = None,
        having_agg=None,
        having_op=None,
        having_value_or_subquery=None,
    ):
        new_col_pointer = self.find_base_box().add_predicate_column_returning_col_pointer(
            qgm_column
        )
        having_col_pointer = (
            self.find_base_box().add_predicate_column_returning_col_pointer(
                having_column
            )
            if having_column is not None
            else None
        )
        self.add_groupby_using_base_col(
            new_col_pointer,
            having_col_pointer,
            having_agg,
            having_op,
            having_value_or_subquery,
        )

    def add_groupby_using_base_col(
        self,
        col_pointer,
        having_col_pointer=None,
        having_agg=None,
        having_op=None,
        having_value_or_subquery=None,
    ):
        having_agg = having_agg
        having_op = having_op
        having_value_or_subquery = having_value_or_subquery
        new_groupby_box = QGMGroupbyBox(
            self,
            col_pointer,
            having_col_pointer,
            having_agg,
            having_op,
            having_value_or_subquery,
        )
        self.groupby_box = new_groupby_box


class QGMLocalPredicate(QGMBase):
    def __init__(
        self, parent, op, col_pointer, value_or_subquery
    ):  # can be tuple if op == between
        QGMBase.__init__(self, parent)
        self.op = op
        self.col_pointer = col_pointer
        self.value_or_subquery = value_or_subquery

    def __eq__(self, other):
        return (
            self.op == other.op
            and self.col_pointer == other.col_pointer
            and (
                (
                    not isinstance(self.value_or_subquery, QGMSubqueryBox)
                    and not isinstance(other.value_or_subquery, QGMSubqueryBox)
                )
                or self.value_or_subquery == other.value_or_subquery
            )
        )

    def if_op_between_right_hand_first_value(self):
        assert self.op == "between"
        return not isinstance(self.value_or_subquery[0], QGMSubqueryBox)

    def if_op_between_right_hand_second_value(self):
        assert self.op == "between"
        return not isinstance(self.value_or_subquery[1], QGMSubqueryBox)

    def is_right_hand_value(self):
        return not isinstance(self.value_or_subquery, QGMSubqueryBox)

    def find_col(self):
        ancient = self.parent
        while not isinstance(ancient, QGMBaseBox):
            ancient = ancient.parent
        return ancient.predicate_cols[self.col_pointer]


class QGMPredicateBox(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.local_predicates: List[Tuple[str, QGMLocalPredicate]] = []

    def __eq__(self, other):
        for (my_conj, my_predicate), (other_conj, other_predicate) in zip(
            self.local_predicates, other.local_predicates
        ):
            if my_conj != other_conj or my_predicate != other_predicate:
                return False
        return True

    def add_local_predicate(self, conj, qgm_column: QGMColumn, op, value_or_subquery):
        new_col_pointer = self.find_base_box().add_predicate_column_returning_col_pointer(
            qgm_column
        )
        self.add_local_predicate_using_base_col(
            conj, op, new_col_pointer, value_or_subquery
        )

    def add_local_predicate_using_base_col(
        self, conj, op, col_pointer, value_or_subquery
    ):
        new_local_predicate = QGMLocalPredicate(
            self, op, col_pointer, value_or_subquery
        )
        self.local_predicates.append((conj, new_local_predicate))


class QGMProjection(QGMBase):
    def __init__(self, parent, agg, col_pointer):
        QGMBase.__init__(self, parent)
        self.agg = agg
        self.col_pointer = col_pointer

    def __eq__(self, other):
        return self.agg == other.agg and self.col_pointer == other.col_pointer

    def find_col(self) -> QGMColumn:
        ancient = self.parent
        contained_subquery = False
        while not isinstance(ancient, QGMBaseBox):
            if isinstance(ancient, QGMSubqueryBox):
                contained_subquery = True
            ancient = ancient.parent

        return (
            ancient.predicate_cols[self.col_pointer]
            if contained_subquery
            else ancient.select_cols[self.col_pointer]
        )


class QGMProjectionBox(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.projections: List[QGMProjection] = []

    def __eq__(self, other):
        for my_projection, other_projection in zip(self.projections, other.projections):
            if my_projection != other_projection:
                return False
        return True

    def add_projection(self, qgm_column: QGMColumn, agg):
        new_col_pointer = self.find_base_box().add_projection_column_returning_col_pointer(
            qgm_column
        )
        self.add_projection_using_base_col(agg, new_col_pointer)

    def add_projection_using_base_col(self, agg, col_pointer):
        new_projection = QGMProjection(self, agg, col_pointer)
        self.projections.append(new_projection)


class QGMOrderbyBox(QGMBase):
    def __init__(self, parent, direction, agg, limit_num, col_pointer):
        QGMBase.__init__(self, parent)
        self.agg = agg
        self.col_pointer = col_pointer
        self.direction = direction
        self.limit_num = limit_num

    def __eq__(self, other):
        return (
            self.agg == other.agg
            and self.col_pointer == other.col_pointer
            and self.direction == other.direction
        )

    def find_col(self):
        ancient = self.parent
        while not isinstance(ancient, QGMBaseBox):
            ancient = ancient.parent

        return ancient.predicate_cols[self.col_pointer]


class QGMGroupbyBox(QGMBase):
    def __init__(
        self,
        parent,
        col_pointer,
        having_col_pointer=None,
        having_agg=None,
        having_op=None,
        having_value_or_subquery=None,
    ):
        QGMBase.__init__(self, parent)
        self.col_pointer = col_pointer
        self.having_agg: Optional[str] = None
        self.having_predicate: Optional[QGMLocalPredicate] = None
        if having_col_pointer is not None:
            self.having_agg = having_agg
            self.having_predicate = QGMLocalPredicate(
                self, having_op, having_col_pointer, having_value_or_subquery
            )

    def __eq__(self, other):
        return (
            self.col_pointer == other.col_pointer
            and self.having_agg == other.having_agg
            and self.having_predicate == other.having_predicate
        )

    def find_col(self):
        ancient = self.parent
        while not isinstance(ancient, QGMBaseBox):
            ancient = ancient.parent

        return ancient.predicate_cols[self.col_pointer]


class QGMBaseBox(QGMBase):
    def __init__(self, parent):
        QGMBase.__init__(self, parent)
        self.select_cols: List[QGMColumn] = []
        self.predicate_cols: List[QGMColumn] = []
        self.predicate_box = QGMPredicateBox(self)
        self.projection_box = QGMProjectionBox(self)
        self.groupby_box: QGMGroupbyBox = None
        self.orderby_box: QGMOrderbyBox = None

    def __eq__(self, other):
        for my_select_col, other_select_col in zip(self.select_cols, other.select_cols):
            if my_select_col != other_select_col:
                return False

        for my_predicate_col, other_predicate_col in zip(
            self.predicate_cols, other.predicate_cols
        ):
            if my_predicate_col != other_predicate_col:
                return False
        return (
            self.predicate_box == other.predicate_box
            and self.projection_box == other.projection_box
        )

    def add_projection_column_returning_col_pointer(self, qgm_column: QGMColumn):
        new_col_pointer = len(self.select_cols)
        self.select_cols.append(qgm_column)
        return new_col_pointer

    def add_predicate_column_returning_col_pointer(self, qgm_column: QGMColumn):
        new_col_pointer = len(self.predicate_cols)
        self.predicate_cols.append(qgm_column)
        return new_col_pointer

    def add_orderby(self, direction, agg, limit_num, qgm_column: QGMColumn):
        new_col_pointer = self.add_predicate_column_returning_col_pointer(qgm_column)
        self.add_orderby_using_base_col(direction, agg, limit_num, new_col_pointer)

    def add_orderby_using_base_col(self, direction, agg, limit_num, col_pointer):
        new_orderby_box = QGMOrderbyBox(self, direction, agg, limit_num, col_pointer)
        self.orderby_box = new_orderby_box

    def add_groupby(
        self,
        qgm_column: QGMColumn,
        having_column: QGMColumn = None,
        having_agg=None,
        having_op=None,
        having_value_or_subquery=None,
    ):
        new_col_pointer = self.add_predicate_column_returning_col_pointer(qgm_column)
        having_col_pointer = (
            self.add_predicate_column_returning_col_pointer(having_column)
            if having_column is not None
            else None
        )
        self.add_groupby_using_base_col(
            new_col_pointer,
            having_col_pointer,
            having_agg,
            having_op,
            having_value_or_subquery,
        )

    def add_groupby_using_base_col(
        self,
        col_pointer,
        having_col_pointer=None,
        having_agg=None,
        having_op=None,
        having_value_or_subquery=None,
    ):
        having_agg = having_agg
        having_op = having_op
        having_value_or_subquery = having_value_or_subquery
        new_groupby_box = QGMGroupbyBox(
            self,
            col_pointer,
            having_col_pointer,
            having_agg,
            having_op,
            having_value_or_subquery,
        )
        self.groupby_box = new_groupby_box


class QGM:
    def __init__(self, db, is_gold):  # we use spider-like db
        self.is_gold = is_gold
        self.db = db
        self.pointer = None
        self.prev_action_ids = []
        self.base_boxes = QGMBaseBox(self)

    def __eq__(self, other):
        return self.base_boxes == other.base_boxes

    @classmethod
    def import_from_sql_ds(cls):
        pass

    def qgm_construct(self):
        pass

    def export_to_text(self):
        pass

    def fast_comparison(self, other_qgm: "QGM"):
        pass

    def qgm_canonicalize(self):
        pass
