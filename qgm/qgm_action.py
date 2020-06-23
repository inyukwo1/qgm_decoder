from qgm.qgm import PROBLEM_ACTIONS


class QGM_ACTION:
    instance = None

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = QGM_ACTION()
        return cls.instance

    def __init__(self):
        self.symbol_action_id_mapping = dict()
        self.action_id_to_prob_action = dict()
        action_id = 0
        for symbol, actions in PROBLEM_ACTIONS:
            self.symbol_action_id_mapping[symbol] = dict()
            for action in actions:
                self.symbol_action_id_mapping[symbol][action] = action_id
                self.action_id_to_prob_action[action_id] = (symbol, action)
                action_id += 1

    def total_action_len(self):
        return sum([len(actions) for prob, actions in PROBLEM_ACTIONS])

    def total_symbol_len(self):
        return len(PROBLEM_ACTIONS)

    def symbol_action_to_action_id(self, symbol: str, action: str):
        return self.symbol_action_id_mapping[symbol][action]

    def symbol_to_symbol_id(self, symbol: str):
        return PROBLEM_ACTIONS.index(symbol)

    def action_id_to_action(self, action_id: int):
        return self.action_id_to_prob_action[action_id][1]

    def possible_action_ids(self, symbol: str):
        action_to_id_dict = self.symbol_action_id_mapping[symbol]
        return [action_to_id_dict[action] for action in action_to_id_dict]
