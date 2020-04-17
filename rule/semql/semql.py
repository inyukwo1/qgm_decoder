from typing import NewType
from rule.grammar import Grammar
from rule.semql.semql_loss import SemQL_Loss

import rule.utils as utils


SKETCH_SYMBOLS = ["A", "C", "T"]

# Singleton
class SemQL(Grammar):
    semql: "SemQL" = None

    def __init__(self, emb_dim=300):
        super(SemQL, self).__init__("./rule/semql/semql.manifesto", emb_dim)
        SemQL.semql = self

    def create_loss_object(self):
        return SemQL_Loss(self.symbol_to_sid)

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
            "head_tab",
            "quantifier_tab_set_num",
            "quantifier_tab",
            "quantifier_tab_set",
            "local_predicate_num",
            "local_predicate_op",
            "local_predicate_agg",
            "local_predicate_col",
            "local_predicate_tab",
        ]
        acc = {key: 0.0 for key in keys}
        cnt = {key: 0 for key in keys}

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

            cnt["detail"] += 1
            acc["detail"] += detail_is_correct
            cnt["sketch"] += 1
            acc["sketch"] += sketch_is_correct
            cnt["total"] += 1
            acc["total"] += total_is_correct

            # More specific accs
            # Head num: Check sel
            p_head = [item for item in p_actions if item[0] == "Sel"]
            g_head = [item for item in g_actions if item[0] == "Sel"]

            p_head_num = len(p_head)
            g_head_num = len(g_head)
            head_num_is_correct = p_head_num == g_head_num

            cnt["head_num"] += 1
            acc["head_num"] += head_num_is_correct

            # Head agg: check A after sel
            p_head_agg = utils.filter_action(p_actions, "A", "Sel", ["A", "C", "T"])
            g_head_agg = utils.filter_action(g_actions, "A", "Sel", ["A", "C", "T"])
            head_agg_is_correct = p_head_agg == g_head_agg

            cnt["head_agg"] += 1
            acc["head_agg"] += head_agg_is_correct

            # Head col: check C after sel
            p_head_col = utils.filter_action(p_actions, "C", "Sel", ["A", "C", "T"])
            g_head_col = utils.filter_action(g_actions, "C", "Sel", ["A", "C", "T"])
            head_col_is_correct = p_head_col == g_head_col

            cnt["head_col"] += 1
            acc["head_col"] += head_col_is_correct

            # head tab: check T after sel
            p_head_tab = utils.filter_action(p_actions, "T", "Sel", ["A", "C", "T"])
            g_head_tab = utils.filter_action(g_actions, "T", "Sel", ["A", "C", "T"])
            head_tab_is_correct = p_head_tab == g_head_tab

            cnt["head_tab"] += 1
            acc["head_tab"] += head_tab_is_correct

            # predicate num: check filter num
            # Quantifier Num: Count number of Q
            p_predicate = [item for item in p_actions if item[0] == "Filter"]
            g_predicate = [item for item in g_actions if item[0] == "Filter"]
            p_predicate_num = len(p_predicate)
            g_predicate_num = len(g_predicate)
            predicate_num_is_correct = p_predicate_num == g_predicate_num

            if p_predicate or g_predicate:
                cnt["local_predicate_num"] += 1
                acc["local_predicate_num"] += predicate_num_is_correct

                # predicate ops
                predicate_op_is_correct = p_predicate == g_predicate
                cnt["local_predicate_op"] += 1
                acc["local_predicate_op"] += predicate_op_is_correct

                # predicate agg: check A after filter
                p_predicate_agg = utils.filter_action(p_actions, "A", "Filter", ["A", "C", "T"])
                g_predicate_agg = utils.filter_action(g_actions, "A", "Filter", ["A", "C", "T"])
                predicate_agg_is_correct = p_predicate_agg == g_predicate_agg
                cnt["local_predicate_agg"] += 1
                acc["local_predicate_agg"] += predicate_agg_is_correct

                # predicate col: check C after filter
                p_predicate_col = utils.filter_action(p_actions, "C", "Filter", ["A", "C", "T"])
                g_predicate_col = utils.filter_action(g_actions, "C", "Filter", ["A", "C", "T"])
                predicate_col_is_correct = p_predicate_col == g_predicate_col
                cnt["local_predicate_col"] += 1
                acc["local_predicate_col"] += predicate_col_is_correct

                # predicate tab: check t after filter
                p_predicate_tab = utils.filter_action(p_actions, "T", "Filter", ["A", "C", "T"])
                g_predicate_tab = utils.filter_action(g_actions, "T", "Filter", ["A", "C", "T"])
                predicate_tab_is_correct = p_predicate_tab == g_predicate_tab
                cnt["local_predicate_tab"] += 1
                acc["local_predicate_tab"] += predicate_tab_is_correct

            # quantifier tab: check all t
            p_quantifier = [item for item in p_actions if item[0] == "T"]
            g_quantifier = [item for item in g_actions if item[0] == "T"]
            p_quantifier_num = len(set(p_quantifier))
            g_quantifier_num = len(set(g_quantifier))
            quan_num_is_correct = p_quantifier_num == g_quantifier_num
            cnt["quantifier_tab_set_num"] += 1
            acc["quantifier_tab_set_num"] += quan_num_is_correct

            # quantifier num: check all t num as set
            quan_tab_is_correct = p_quantifier == g_quantifier
            cnt["quantifier_tab"] += 1
            acc["quantifier_tab"] += quan_tab_is_correct

            quan_tab_set_is_correct = set(p_quantifier) == set(g_quantifier)
            cnt["quantifier_tab_set"] += 1
            acc["quantifier_tab_set"] += quan_tab_set_is_correct

        for key in acc.keys():
            acc[key] = acc[key] / cnt[key]

        return acc, is_correct_list
