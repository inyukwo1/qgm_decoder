from rule.loss import Loss


class SemQL_Loss(Loss):
    def __init__(self, symbol_to_sid):
        super(SemQL_Loss, self).__init__()


        # Symbols
        self.root = symbol_to_sid["Root"]
        self.sel = symbol_to_sid["Sel"]
        self.n = symbol_to_sid["N"]
        self.filter = symbol_to_sid["Filter"]
        self.a = symbol_to_sid["A"]
        self.c = symbol_to_sid["C"]
        self.t = symbol_to_sid["T"]

    def _get_key(self, action_node, prev_actions):
        if action_node == self.root:
            key = "predicate_num"
        elif action_node == self.sel:
            key = "head_num"
        elif action_node == self.a:
            prev_symbols = [prev_action[0] for prev_action in prev_actions if prev_action[0] in ["Sel", "Filter"]]
            assert len(prev_symbols) > 0, "Wrong prev_actions: {}".format(prev_actions)
            key = "head_agg" if prev_symbols[-1] == "Sel" else "predicate_agg"
        elif action_node == self.filter:
            key = "predicate_num"
        elif action_node == self.c:
            prev_symbols = [prev_action[0] for prev_action in prev_actions if prev_action[0] in ["Sel", "Filter"]]
            assert len(prev_symbols) > 0, "Wrong prev_actions: {}".format(prev_actions)
            key = "head_col" if prev_symbols[-1] == "Sel" else "predicate_col"
        elif action_node == self.t:
            key = "quantifier_tab"
        else:
            raise RuntimeError("Should not be here")
        return key