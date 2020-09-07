from typing import NewType
import json

Symbol = NewType("Symbol", str)
Action = NewType("Action", str)


class QGM_ACTION:
    instance = None

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = QGM_ACTION()
        return cls.instance

    def __init__(self):
        with open("data/state_actions.json") as f:
            self.state_action_dict = json.load(f)

        self.state_action_dict["C"] = []
        self.state_action_dict["T"] = []

        self.symbol_action_id_mapping = dict()
        self.action_id_to_prob_action = dict()
        action_id = 0
        for symbol, actions in self.state_action_dict.items():
            self.symbol_action_id_mapping[symbol] = dict()
            for action in actions:
                self.symbol_action_id_mapping[symbol][action] = action_id
                self.action_id_to_prob_action[action_id] = (symbol, action)
                action_id += 1

    @classmethod
    def total_action_len(cls):
        return sum(
            [
                len(actions)
                for prob, actions in cls.get_instance().state_action_dict.items()
            ]
        )

    @classmethod
    def total_symbol_len(cls):
        return len(cls.get_instance().state_action_dict)

    @classmethod
    def symbol_action_to_action_id(cls, symbol: str, action: str):
        return cls.get_instance().symbol_action_id_mapping[symbol][action]

    @classmethod
    def symbol_to_symbol_id(cls, symbol: str):
        for index, (origin_symbol, _) in enumerate(
            cls.get_instance().state_action_dict.items()
        ):
            if symbol == origin_symbol:
                return index
        raise Exception

    @classmethod
    def action_id_to_action(cls, action_id: int):
        return cls.get_instance().action_id_to_prob_action[action_id][1]

    @classmethod
    def possible_action_ids(cls, symbol: str):
        action_to_id_dict = cls.get_instance().symbol_action_id_mapping[symbol]
        return [action_to_id_dict[action] for action in action_to_id_dict]
