from typing import NewType
from rule.grammar import Grammar
from rule.noqgm.noqgm_loss import NOQGM_Loss

import rule.utils as utils

SKETCH_SYMBOLS = ["C", "T"]

# Singleton
class NOQGM(Grammar):
    noqgm: "NOQGM" = None

    def __init__(self, emb_dim=300):
        super(NOQGM, self).__init__("./rule/noqgm/noqgm.manifesto", emb_dim)
        NOQGM.noqgm = self

    def create_loss_object(self):
        return NOQGM_Loss(self.symbol_to_sid)

    @classmethod
    def create_data(cls, sql, db):
        # json to strings of actions
        # No nested in the from clause
        for item in sql["from"]["table_units"]:
            if item[0] == "sql":
                return None
        if sql["intersect"] or sql["union"] or sql["except"]:
            # Multiple
            return None
        elif sql["having"] or sql["groupby"] or sql["orderby"]:
            return None
        else:
            # Single
            # Root
            col_num = 0

            action = "Root(0) "
            action += "Sel({}) ".format(len(sql["select"][1]))
            for select in sql["select"][1]:
                ori_col_id = select[1][1][1]
                new_col_id = db["col_set"].index(db["column_names"][ori_col_id][1])
                if ori_col_id == 0:
                    tab_id = sql["from"]["table_units"][0][1]
                else:
                    tab_id = db["column_names"][ori_col_id][0]
                action += "C({}) ".format(new_col_id)
                action += "T({}) ".format(tab_id)
                action += "A({}) ".format(select[0])
                col_num += 1
            action += "Filter({}) ".format((len(sql["where"]) + 1) // 2)

            for idx in range(0, len(sql["where"]), 2):
                where_cond = sql["where"][idx]
                if isinstance(where_cond[3], dict):
                    return None
                op_id = where_cond[1] + 4 if where_cond[0] else where_cond[1]
                action += "B({}) ".format(op_id)
                ori_col_id = where_cond[2][1][1]
                new_col_id = db["col_set"].index(db["column_names"][ori_col_id][1])
                if ori_col_id == 0:
                    tab_id = sql["from"]["table_units"][0][1]
                else:
                    tab_id = db["column_names"][ori_col_id][0]
                action += "C({}) ".format(new_col_id)
                action += "T({}) ".format(tab_id)
                action += "A({}) ".format(where_cond[2][1][0])
                col_num += 1

        return action[:-1]

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
