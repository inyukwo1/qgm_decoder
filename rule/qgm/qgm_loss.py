from rule.loss import Loss


class QGM_Loss(Loss):
    def __init__(self, symbol_to_sid):
        super(QGM_Loss, self).__init__()

        # Symbols
        self.b = symbol_to_sid("B")
        self.q = symbol_to_sid("Q")
        self.h = symbol_to_sid("H")
        self.p = symbol_to_sid("P")
        self.a = symbol_to_sid("A")
        self.o = symbol_to_sid("O")
        self.t = symbol_to_sid("T")
        self.c = symbol_to_sid("C")

    def _get_key(self, action_node, prev_actions):
        if action_node == self.b:
            key = "predicate_num"
        elif action_node == self.q:
            key = "quantifier_num"
        elif action_node == self.h:
            key = "head_num"
        elif action_node == self.p:
            key = "predicate_num"
        elif action_node == self.a:
            prev_symbols = [
                prev_action[0]
                for prev_action in prev_actions
                if prev_action[0] in ["H", "P"]
            ]
            assert len(prev_symbols) > 0, "Wrong prev_actions: {}".format(prev_actions)
            key = "head_agg" if prev_symbols[-1] == "H" else "predicate_agg"
        elif action_node == self.o:
            key = "predicate_op"
        elif action_node == self.t:
            key = "quantifier_tab"
        elif action_node == self.c:
            prev_symbols = [
                prev_action[0]
                for prev_action in prev_actions
                if prev_action[0] in ["H", "P"]
            ]
            assert len(prev_symbols) > 0, "Wrong prev_actions: {}".format(prev_actions)
            key = "head_col" if prev_symbols[-1] == "H" else "predicate_col"
        else:
            RuntimeError("Should not be here")
        return key
