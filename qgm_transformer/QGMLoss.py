import torch


class Loss:
    def __init__(self, grammar):
        keys = [
            "head_agg",
            "head_col",
            "head_num",
            "predicate_agg",
            "predicate_col",
            "predicate_op",
            "predicate_num",
            "quantifier_num",
            "quantifier_tab",
            "sketch",
            "detail",
            "total",
        ]
        self.loss = {key: torch.tensor(0.0).cuda() for key in keys}
        self.grammar = grammar

    def add(self, value, action_node, prev_actions):
        key = self._get_key(action_node, prev_actions)
        assert key in self.loss.keys(), "key:{}".format(key)
        self.loss[key] += value
        if "col" in key or "agg" in key or "tab" in key:
            self.loss["detail"] += value
        else:
            self.loss["sketch"] += value
        self.loss["total"] += value

    def get_loss_dic(self):
        return self.loss

    def get_loss(self, key):
        return self.loss[key]

    def get_total_loss(self):
        return self.get_loss("total")

    def get_keys(self):
        return self.loss.keys()


class QGMLoss(Loss):
    def __init__(self, grammar):
        super(QGMLoss, self).__init__(self, grammar)


    def _get_key(self, action_node, prev_actions):
        if action_node == self.grammar.symbol_to_symbol_id["B"]:
            key = "predicate_num"
        elif action_node == self.grammar.symbol_to_symbol_id["Q"]:
            key = "quantifier_num"
        elif action_node == self.grammar.symbol_to_symbol_id["H"]:
            key = "head_num"
        elif action_node == self.grammar.symbol_to_symbol_id["P"]:
            key = "predicate_num"
        elif action_node == self.grammar.symbol_to_symbol_id["A"]:
            # 끝까지 봐서 H 면 head_agg 바로 전이 O 면 predicate_agg
            if prev_actions[-1][0] == "O":
                key = "predicate_agg"
            else:
                for action in reversed(prev_actions):
                    if action[0] in ["A", "C"]:
                        continue
                    elif action[0] == "H":
                        key = "head_agg"
                        break
                if not key:
                    raise RuntimeError("Should not be here {}:{}".format(prev_actions, action_node))
        elif action_node == self.grammar.symbol_to_symbol_id["O"]:
            key = "predicate_op"
        elif action_node == self.grammar.symbol_to_symbol_id["T"]:
            key = "quantifier_tab"
        elif action_node == self.grammar.symbol_to_symbol_id["C"]:
            # 끝까지 봐서 H 면 head_col바로 전이 O 면 predicate_col
            if prev_actions[-2][0] == "O":
                key = "predicate_col"
            else:
                for action in reversed(prev_actions[:-1]):
                    if action[0] in ["A", "C"]:
                        continue
                    elif action[0] == "H":
                        key = "head_col"
                if not key:
                    raise RuntimeError("Should not be here {}:{}".format(prev_actions, action_node))
        else:
            raise RuntimeError("Should not be here {}:{}".format(prev_actions, action_node))
        return key

class SemQLLoss(Loss):
    def __init__(self, grammar):
        super(SemQLLoss, self).__init__(grammar)

    def _get_key(self, action_node, prev_action):
        if action_node == self.grammar.symbol_to_symbol_id["Root1"]:
            key = "predicate_num"
        elif action_node == self.grammar.symbol_to_symbol_id["Root"]:
            key = "predicate_num"
        elif action_node == self.grammar.symbol_to_symbol_id["N"]:
            key = "head_num"
        elif action_node == self.grammar.symbol_to_symbol_id["Filter"]:
            key = "predicate_num"
        elif action_node == self.grammar.symbol_to_symbol_id["A"]:
            if prev_action[-1][0] == "N":
                key = "head_agg"
            else:
                key = "predicate_agg"
        elif action_node == self.grammar.symbol_to_symbol_id["C"]:
            if prev_action[-2][0] == "N":
                key = "head_col"
            else:
                key = "predicate_col"
        elif action_node == self.grammar.symbol_to_symbol_id["T"]:
            key = "quantifier_tab"
        else:
            raise RuntimeError("Should not be here")
        return key
