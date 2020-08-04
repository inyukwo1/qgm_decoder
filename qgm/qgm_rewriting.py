from qgm.qgm import QGM, QGMSubqueryBox


def qgm_subquerybox_orderby_to_agg(qgm_subquery_box: QGMSubqueryBox):
    qgm_projection = qgm_subquery_box.projection
    orderby_box = qgm_subquery_box.orderby_box
    if orderby_box is not None:
        if orderby_box.limit_num == 1:
            if qgm_projection.find_col() == orderby_box.find_col():
                if orderby_box.direction == "asc":
                    qgm_projection.agg = "min"
                else:
                    qgm_projection.agg = "max"
                assert (
                    orderby_box.col_pointer
                    == len(qgm_projection.find_base_box().predicate_cols) - 1
                )
                qgm_projection.find_base_box().predicate_cols.pop()
                qgm_subquery_box.orderby_box = None

    for conj, predicate in qgm_subquery_box.local_predicates:
        if isinstance(predicate.value_or_subquery, QGMSubqueryBox):
            qgm_subquerybox_orderby_to_agg(predicate.value_or_subquery)


def qgm_orderby_to_agg(qgm: QGM):
    qgm_projections = qgm.base_boxes.projection_box.projections
    orderby_box = qgm.base_boxes.orderby_box
    if orderby_box is not None:
        if orderby_box.limit_num == 1:
            if (
                len(qgm_projections) == 1
                and qgm_projections[0].find_col() == orderby_box.find_col()
            ):
                if orderby_box.direction == "asc":
                    qgm_projections[0].agg = "min"
                else:
                    qgm_projections[0].agg = "max"
                assert orderby_box.col_pointer == len(qgm.base_boxes.predicate_cols) - 1
                qgm.base_boxes.predicate_cols.pop()
                qgm.base_boxes.orderby_box = None
    for conj, predicate in qgm.base_boxes.predicate_box.local_predicates:
        if isinstance(predicate.value_or_subquery, QGMSubqueryBox):
            qgm_subquerybox_orderby_to_agg(predicate.value_or_subquery)


def qgm_canonicalize(self: QGM):
    qgm_orderby_to_agg(self)


QGM.qgm_canonicalize = qgm_canonicalize
