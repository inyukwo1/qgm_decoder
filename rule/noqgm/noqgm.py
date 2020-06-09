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
    def create_data(cls, sql, db, dataset="spider"):
        # json to strings of actions
        # No nested in the from clause
        for item in sql["from"]["table_units"]:
            if item[0] == "sql":
                return None
        if sql["intersect"] or sql["union"] or sql["except"]:
            # Multiple
            return None
        elif sql["having"]:
            return None
        elif sql["groupby"] or sql["orderby"]:
            return True
        else:
            if dataset == "spider":
                # Single
                # Root
                if sql["where"]:
                    action = "Root(0) "
                else:
                    action = "Root(1) "
                # Sel
                assert sql["select"], "Something is weird {}".format(sql["select"])
                action += "Sel({}) ".format(len(sql["select"][1]) - 1)
                for select in sql["select"][1]:
                    action += "A({}) ".format(select[0])
                    ori_col_id = select[1][1][1]
                    new_col_id = db["col_set"].index(db["column_names"][ori_col_id][1])
                    if ori_col_id == 0:
                        tab_id = sql["from"]["table_units"][0][1]
                    else:
                        tab_id = db["column_names"][ori_col_id][0]
                    action += "C({}) ".format(new_col_id)
                    action += "T({}) ".format(tab_id)

                for idx in range(0, len(sql["where"]), 2):
                    where_cond = sql["where"][idx]
                    if isinstance(where_cond[3], dict):
                        return None
                    if len(sql["where"]) > idx + 1:
                        assert sql["where"][idx + 1] in ["or", "and"]
                        action += "Filter({}) ".format(
                            0 if sql["where"][idx + 1] == "or" else 1
                        )
                    op_id = where_cond[1] + 6 if where_cond[0] else where_cond[1] + 2
                    action += "Filter({}) ".format(op_id)
                    action += "A({}) ".format(where_cond[2][1][0])
                    ori_col_id = where_cond[2][1][1]
                    new_col_id = db["col_set"].index(db["column_names"][ori_col_id][1])
                    if ori_col_id == 0:
                        tab_id = sql["from"]["table_units"][0][1]
                    else:
                        tab_id = db["column_names"][ori_col_id][0]
                    action += "C({}) ".format(new_col_id)
                    action += "T({}) ".format(tab_id)
            elif dataset == "wikisql":
                # Single
                # Root
                action = "Root(0) "
                # Sel
                assert sql["select"], "Something is weird {}".format(sql["select"])
                for select in sql["select"][1]:
                    action += "A({}) ".format(select[0])
                    ori_col_id = select[1][1][1]
                    new_col_id = db["col_set"].index(db["column_names"][ori_col_id][1])
                    action += "C({}) ".format(new_col_id)

                for idx in range(0, len(sql["where"]), 2):
                    where_cond = sql["where"][idx]
                    if isinstance(where_cond[3], dict):
                        return None
                    if len(sql["where"]) > idx + 1:
                        assert sql["where"][idx + 1] in ["or", "and"]
                        action += "Filter(0) "
                    op_id = where_cond[1] - 1
                    action += "Filter({}) ".format(op_id)
                    ori_col_id = where_cond[2][1][1]
                    new_col_id = db["col_set"].index(db["column_names"][ori_col_id][1])
                    action += "C({}) ".format(new_col_id)

        return action[:-1]

    def to_sql(self, noqgm, data):
        def get_num(action):
            left = action.index("(")
            right = action.index(")")
            return int(action[left : right + 1])

        def get_symbol(action):
            return action.split("(")[0]

        def parse_ACT(noqgm, last_idx):
            assert get_symbol(noqgm[last_idx]) == "A"
            assert get_symbol(noqgm[last_idx + 1]) == "C"
            assert get_symbol(noqgm[last_idx + 2]) == "T"

            # Aggregation
            agg_ops = ["none", "max", "min", "count", "sum", "avg"]
            agg_num = get_num(noqgm[last_idx])

            # Column and Table
            col_name = str(get_num(noqgm[last_idx + 1]))
            tab_name = str(get_num(noqgm[last_idx + 2]))

            if agg_num != 0:
                tmp = "{}()".format(agg_ops[agg_num], tab_name, col_name)
            else:
                tmp = "{}.{}".format(tab_name, col_name)
            return tmp

        def parse_where_clause(noqgm, cur_idx):
            where_ops = [
                "or",
                "and",
                "Not",
                "Between",
                "=",
                ">",
                "<",
                ">=",
                "<=",
                "!=",
                "In",
                "Like",
                "Is",
                "Exists",
                "Not In",
                "Not Like",
            ]
            where = "{}"
            while cur_idx < len(noqgm):
                action = noqgm[cur_idx]
                if get_symbol(action) == "Filter":
                    # Do something here
                    num = get_num(action)
                    if num in [0, 1]:
                        where = where.format(
                            "{} {} {}".format("{}", where_ops[num], "{}")
                        )
                    else:
                        where = where.format(
                            "{} {} 'value'".format("{}", where_ops[num])
                        )
                    cur_idx += 1
                elif get_symbol(action) == "A":
                    tmp = parse_ACT(noqgm, cur_idx)
                    where = where.format(tmp)
                    cur_idx += 3
                else:
                    break
            return where, cur_idx

        def parse_select_clause(noqgm, cur_idx):
            select = ""
            while cur_idx < len(noqgm):
                action = noqgm[cur_idx]
                if get_symbol(action) == "Sel":
                    cur_idx += 1
                elif get_symbol(action) == "A":
                    tmp = parse_ACT(noqgm, cur_idx)
                    select += tmp
                    cur_idx += 3
                else:
                    break
            return select, cur_idx

        def parse_from_clause(noqgm, sql):
            # Get all tables
            # Connect all tables
            return "NOTHING"

        sql = "SELECT {} FROM {}"
        assert "Root" in noqgm[0], "weird : {}".format(noqgm)
        root_num = get_num(noqgm[0])
        cur_idx, select_clause = parse_select_clause(noqgm, 1)
        if root_num == 0:
            sql += " WHERE {}"
            # Where clause
            cur_idx, where_clause = parse_where_clause(noqgm, cur_idx)
        from_clause = parse_from_clause(noqgm)

        sql = sql.format(select_clause, from_clause, where_clause)
        return sql

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
