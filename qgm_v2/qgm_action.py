from typing import NewType

STATE_ACTIONS = [
    ("select_col_num", ["1", "2", "3", "4", "5"]),
    ("projection_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("predicate_col_done", ["done", "do"]),
    ("predicate_conj", ["and", "or"]),
    (
        "predicate_op",
        [
            "between",
            "=",
            ">",
            "<",
            ">=",
            "<=",
            "!=",
            "like",
            "is",
            "exists",
            "in",
            "not in",
        ],
    ),
    ("value_or_subquery", ["value", "subquery"]),
    ("op_done", ["done", "do"]),
    ("subquery_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("subquery_op_done", ["done", "do"]),
    ("subquery_predicate_and_or", ["and", "or"]),
    (
        "subquery_predicate_op",
        [
            "between",
            "=",
            ">",
            "<",
            ">=",
            "<=",
            "!=",
            "like",
            "is",
            "exists",
            "in",
            "not in",
        ],
    ),
    ("orderby_exists", ["no", "yes"]),
    ("orderby_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("orderby_direction", ["asc", "desc"]),
    ("groupby_exists", ["no", "yes"]),
    ("having_exists", ["no", "yes"]),
    ("having_agg", ["none", "max", "min", "count", "sum", "avg"]),
    ("C", []),
    ("T", []),
]

State = NewType("Symbol", str)
Action = NewType("Action", str)


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
        for symbol, actions in STATE_ACTIONS:
            self.symbol_action_id_mapping[symbol] = dict()
            for action in actions:
                self.symbol_action_id_mapping[symbol][action] = action_id
                self.action_id_to_prob_action[action_id] = (symbol, action)
                action_id += 1

    @classmethod
    def total_action_len(cls):
        return sum([len(actions) for prob, actions in STATE_ACTIONS])

    @classmethod
    def total_symbol_len(cls):
        return len(STATE_ACTIONS)

    @classmethod
    def symbol_action_to_action_id(cls, symbol: str, action: str):
        return cls.get_instance().symbol_action_id_mapping[symbol][action]

    @classmethod
    def symbol_to_symbol_id(cls, symbol: str):
        for index, (origin_symbol, _) in enumerate(STATE_ACTIONS):
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
