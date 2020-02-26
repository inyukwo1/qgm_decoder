from rule.grammar import Grammar
from rule.qgm.qgm_loss import QGM_Loss

import rule.utils as utils

SKETCH_SYMBOLS = ["A", "C", "T"]


class QGM(Grammar):
    def __init__(self, emb_dim=300):
        super(QGM, self).__init__("./rule/qgm/qgm.manifesto", emb_dim)
        self.loss = QGM_Loss(self.symbol_to_sid)

    def create_loss_object(self):
        return QGM_Loss(self.symbol_to_sid)

    def create_data(self, qgm_boxes):
        # Simple query only
        qgm_box = qgm_boxes[0]

        actions = "B({}) ".format(0 if qgm_box["body"]["local_predicates"] else 1)

        # Q
        q_len = len(qgm_box["body"]["quantifiers"])
        for idx, quantifier in enumerate(qgm_box["body"]["quantifiers"]):
            actions += "Q({}) ".format(1 if idx + 1 == q_len else 0)
            actions += "T({}) ".format(quantifier)

        # H
        h_len = len(qgm_box["head"])
        for idx, head in enumerate(qgm_box["head"]):
            actions += "H({}) ".format(1 if idx + 1 == h_len else 0)
            actions += "A({}) ".format(head[0])
            actions += "C({}) ".format(head[1])

        # P
        p_len = len(qgm_box["body"]["local_predicates"])
        for idx, predicate in enumerate(qgm_box["body"]["local_predicates"]):
            actions += "P({}) ".format(1 if idx + 1 == p_len else 0)
            actions += "O({}) ".format(predicate[2])
            actions += "A({}) ".format(predicate[0])
            actions += "C({}) ".format(predicate[1])

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

        # Parse actions
        parsed_actions = []
        for actions in gold_actions:
            tmp = []
            for item in actions.split(" "):
                symbol = item.split("(")[0]
                idx = item.split("(")[1].split(")")[0]
                tmp += [(symbol, int(idx))]
            parsed_actions += [tmp]
        gold_actions = parsed_actions

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
            p_head_num = utils.count_symbol(p_actions, "H")
            g_head_num = utils.count_symbol(g_actions, "H")
            head_num_is_correct = p_head_num == g_head_num

            # Head others
            head_agg_is_correct = head_col_is_correct = head_num_is_correct
            if head_num_is_correct:
                # Head Agg: Check A after H
                pred_head_agg = utils.filter_action(p_actions, "A", ["H"])
                gold_head_agg = utils.filter_action(g_actions, "A", ["H"])
                head_agg_is_correct = pred_head_agg == gold_head_agg
                # Head col: Check C After H A
                pred_head_col = utils.filter_action(p_actions, "C", ["H"])
                gold_head_col = utils.filter_action(g_actions, "C", ["H"])
                head_col_is_correct = pred_head_col == gold_head_col

            acc["head_num"] += head_num_is_correct
            acc["head_agg"] += head_agg_is_correct
            acc["head_col"] += head_col_is_correct

            # Quantifier Num: Count number of Q
            pred_quan_num = utils.count_symbol(p_actions, "Q")
            gold_quan_num = utils.count_symbol(g_actions, "Q")
            quan_num_is_correct = pred_quan_num == gold_quan_num

            # Quantifier others
            quan_tab_is_correct = quan_num_is_correct
            if quan_num_is_correct:
                # Quantifier Tab: Check T After Q
                pred_quan_tab = utils.filter_action(p_actions, "T", ["Q"])
                gold_quan_tab = utils.filter_action(g_actions, "T", ["Q"])
                quan_tab_is_correct = pred_quan_tab == gold_quan_tab

            acc["quantifier_num"] += quan_num_is_correct
            acc["quantifier_tab"] += quan_tab_is_correct

            # Predicate Num: Count number of P
            pred_predicate_num = utils.count_symbol(p_actions, "P")
            gold_predicate_num = utils.count_symbol(g_actions, "P")
            predicate_num_is_correct = pred_predicate_num == gold_predicate_num

            # Others
            predicate_op_is_correct = (
                predicate_agg_is_correct
            ) = predicate_col_is_correct = predicate_num_is_correct

            # Predicate Num: Count number of P
            if predicate_num_is_correct:
                # Predicate op: Check O After P
                pred_predicate_op = utils.filter_action(p_actions, "O", ["P"])
                gold_predicate_op = utils.filter_action(g_actions, "O", ["P"])
                predicate_op_is_correct = pred_predicate_op == gold_predicate_op

                # Predicate Agg: Check A After O
                pred_predicate_agg = utils.filter_action(p_actions, "A", ["O"])
                gold_predicate_agg = utils.filter_action(g_actions, "A", ["O"])
                predicate_agg_is_correct = pred_predicate_agg == gold_predicate_agg

                # Predicate Col: Check C After O A
                pred_predicate_col = utils.filter_action(p_actions, "C", ["O"])
                gold_predicate_col = utils.filter_action(g_actions, "C", ["O"])
                predicate_col_is_correct = pred_predicate_col == gold_predicate_col

            acc["local_predicate_num"] += predicate_num_is_correct
            acc["local_predicate_agg"] += predicate_agg_is_correct
            acc["local_predicate_col"] += predicate_col_is_correct
            acc["local_predicate_op"] += predicate_op_is_correct

        for key in acc.keys():
            acc[key] = acc[key] / len(gold_actions)

        return acc, is_correct_list
