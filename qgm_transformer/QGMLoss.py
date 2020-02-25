import torch


class QGMLoss:
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
            "aux",
            "total",
        ]
        self.loss = {key: torch.tensor(0.0).cuda() for key in keys}
        self.grammar = grammar

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
            if prev_actions[-1][0] == "O":
                key = "predicate_agg"
            elif prev_actions[-1][0] == "H":
                key = "head_agg"
            else:
                RuntimeError("Should not be here")
        elif action_node == self.grammar.symbol_to_symbol_id["O"]:
            key = "predicate_op"
        elif action_node == self.grammar.symbol_to_symbol_id["T"]:
            key = "quantifier_tab"
        elif action_node == self.grammar.symbol_to_symbol_id["C"]:
            if prev_actions[-2][0] == "H":
                key = "head_col"
            elif prev_actions[-2][0] == "O":
                key = "predicate_col"
            else:
                RuntimeError("Should not be here")
        else:
            RuntimeError("Should not be here")
        return key

    def add(self, value, action_node, prev_actions):
        key = self._get_key(action_node, prev_actions)
        assert key in self.loss.keys(), "key:{}".format(key)
        self.loss[key] += value
        if "col" in key or "agg" in key or "tab" in key:
            self.loss["detail"] += value
        else:
            self.loss["sketch"] += value
        self.loss["total"] += value

    def add_aux(self, value):
        self.loss["aux"] += value

    def get_loss_dic(self):
        return self.loss

    def get_loss(self, key):
        return self.loss[key]

    def get_total_loss(self):
        return self.get_loss("total")

    def get_keys(self):
        return self.loss.keys()
