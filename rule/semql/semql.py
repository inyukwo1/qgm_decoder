from typing import NewType
from rule.grammar import Grammar
from rule.semql.semql_loss import SemQL_Loss

import rule.utils as utils


SKETCH_SYMBOLS = ["C", "T"]

# Singleton
class SemQL(Grammar):
    semql: "SemQL" = None

    def __init__(self, emb_dim=300):
        super(SemQL, self).__init__("./rule/semql/semql.manifesto", emb_dim)
        SemQL.semql = self

    def create_loss_object(self):
        return SemQL_Loss(self.symbol_to_sid)

    @classmethod
    def create_data(cls, data):
        raise NotImplementedError("need implementation")
        return None

    @classmethod
    def create_data(cls, qgm_boxes):
        # Simple query only
        qgm_box = qgm_boxes[0]

        actions = "Root({}) ".format(0 if qgm_box["body"]["local_predicates"] else 1)

        # Sel
        sel_len = len(qgm_box["head"])
        actions += "Sel({}) ".format(sel_len - 1)
        for idx, head in enumerate(qgm_box["head"]):
            actions += "A({}) ".format(head[0])
            actions += "C({}) ".format(head[1])
            actions += "T({}) ".format(head[2])

        # Filter
        p_len = len(qgm_box["body"]["local_predicates"])
        for idx, predicate in enumerate(qgm_box["body"]["local_predicates"]):
            if idx + 1 < p_len:
                actions += "Filter({}) ".format(0)
            actions += "Filter({}) ".format(predicate[2] + 1)
            actions += "A({}) ".format(predicate[0])
            actions += "C({}) ".format(predicate[1][0])
            actions += "T({}) ".format(predicate[1][1])

        actions = actions.strip(" ")
        return actions

    def cal_acc(self, pred_actions, gold_actions):
        assert len(pred_actions) == len(gold_actions), "Num diff: {}, {}".format(
            len(pred_actions), len(gold_actions)
        )
        keys = [
            "total",
            "detail",
            "sketch",
            "head_num",
            "head_agg",
            "head_col",
            "quantifier_num",
            "quantifier_tab",
            "local_predicate_num",
            "local_predicate_op",
            "local_predicate_agg",
            "local_predicate_col",
        ]
        acc = {key: 0.0 for key in keys}

        is_correct_list = []
        for p_actions, g_actions in zip(pred_actions, gold_actions):
            # Sketch
            p_sketch = [item for item in p_actions if item[0] not in SKETCH_SYMBOLS]
            g_sketch = [item for item in g_actions if item[0] not in SKETCH_SYMBOLS]
            sketch_is_correct = p_sketch == g_sketch

            # Detail
            p_detail = [item for item in p_actions if item[0] in SKETCH_SYMBOLS]
            g_detail = [item for item in g_actions if item[0] in SKETCH_SYMBOLS]
            detail_is_correct = p_detail == g_detail

            # Correctness
            total_is_correct = sketch_is_correct and detail_is_correct
            is_correct_list += [total_is_correct]

            acc["detail"] += detail_is_correct
            acc["sketch"] += sketch_is_correct
            acc["total"] += total_is_correct

            # More specific accs
            # Head num: Check sel
            p_head_num = utils.get_symbol_indices(p_actions, "Sel")
            g_head_num = utils.get_symbol_indices(g_actions, "Sel")
            head_num_is_correct = p_head_num == g_head_num

            # Head agg: Check A after sel
            head_agg_is_correct = (
                head_col_is_correct
            ) = head_tab_is_correct = head_num_is_correct
            if head_num_is_correct:
                # Head agg: check A after sel
                p_head_agg = utils.filter_action(p_actions, "A", ["Sel"])
                g_head_agg = utils.filter_action(g_actions, "A", ["Sel"])
                head_agg_is_correct = p_head_agg == g_head_agg

                # Head col: check C after sel
                p_head_col = utils.filter_action(p_actions, "C", ["Sel"])
                g_head_col = utils.filter_action(g_actions, "C", ["Sel"])
                head_col_is_correct = p_head_col == g_head_col

                # head tab: check T after sel
                p_head_tab = utils.filter_action(p_actions, "T", ["Sel"])
                g_head_tab = utils.filter_action(g_actions, "T", ["Sel"])
                head_tab_is_correct = p_head_tab == g_head_tab

            acc["head_num"] += head_num_is_correct
            acc["head_agg"] += head_agg_is_correct
            acc["head_col"] += head_col_is_correct

            # predicate num: check filter num
            # Quantifier Num: Count number of Q
            p_predicate_num = utils.count_symbol(p_actions, "Filter")
            g_predicate_num = utils.count_symbol(g_actions, "Filter")
            predicate_num_is_correct = p_predicate_num == g_predicate_num

            predicate_op_is_correct = (
                predicate_agg_is_correct
            ) = predicate_num_is_correct
            predicate_col_is_correct = (
                predicate_tab_is_correct
            ) = predicate_num_is_correct

            if predicate_num_is_correct:
                # predicate ops
                p_predicate_op = utils.get_symbol_indices(p_actions, "Filter")
                g_predicate_op = utils.get_symbol_indices(g_actions, "Filter")
                predicate_op_is_correct = p_predicate_op == g_predicate_op

                # predicate agg: check A after filter
                p_predicate_agg = utils.filter_action(p_actions, "A", ["Filter"])
                g_predicate_agg = utils.filter_action(g_actions, "A", ["Filter"])
                predicate_agg_is_correct = p_predicate_agg == g_predicate_agg

                # predicate col: check C after filter
                p_predicate_col = utils.filter_action(p_actions, "C", ["Filter"])
                g_predicate_col = utils.filter_action(g_actions, "C", ["Filter"])
                predicate_col_is_correct = p_predicate_col == g_predicate_col

                # predicate tab: check t after filter
                p_predicate_tab = utils.filter_action(p_actions, "T", ["Filter"])
                g_predicate_tab = utils.filter_action(g_actions, "T", ["Filter"])
                predicate_tab_is_correct = p_predicate_tab == g_predicate_tab

            acc["local_predicate_num"] += predicate_num_is_correct
            acc["local_predicate_agg"] += predicate_agg_is_correct
            acc["local_predicate_col"] += predicate_col_is_correct
            acc["local_predicate_op"] += predicate_op_is_correct

            # quantifier tab: check all t
            p_quantifier_num = len(set(utils.get_symbol_indices(p_actions, "T")))
            g_quantifier_num = len(set(utils.get_symbol_indices(g_actions, "T")))
            quan_num_is_correct = p_quantifier_num == g_quantifier_num

            # quantifier num: check all t num as set
            quan_tab_is_correct = head_tab_is_correct and predicate_tab_is_correct

            acc["quantifier_num"] += quan_num_is_correct
            acc["quantifier_tab"] += quan_tab_is_correct

        for key in acc.keys():
            acc[key] = acc[key] / len(gold_actions)

        return acc, is_correct_list
