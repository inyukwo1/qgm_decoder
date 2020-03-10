import torch.nn as nn
from typing import NewType, Tuple, Dict, List

ACTION_SIGN = " ::= "

SYMBOL_STATE = "symbol"
GRAMMAR_STATE = "grammar"

SYMBOL_START = "<symbol_start>"
SYMBOL_END = "<symbol_end>"
GRAMMAR_START = "<grammar_start>"
GRAMMAR_END = "<grammar_end>"

# Types
Symbol = NewType("Symbol", str)
SymbolId = NewType("SymbolId", int)
LocalActionId = NewType("LocalActionId", int)
GlobalActionId = NewType("GlobalActionId", int)
Action = NewType("Action", Tuple[Symbol, LocalActionId])


class Grammar(nn.Module):
    def __init__(self, grammar_path, emb_dim=300):
        super(Grammar, self).__init__()
        # Symbol
        self.symbols = []
        self.terminals = []
        self.start_symbol = None
        self.symbol_to_sid = {}
        self.sid_to_symbol = {}

        # Action
        self.actions = {}
        self.action_to_aid: Dict[Action, GlobalActionId] = {}
        self.aid_to_action = {}
        self._create_grammar(grammar_path)
        self.terminals = [
            symbol for symbol in self.symbols if symbol not in self.actions.keys()
        ]

        # Embeddings
        self.key_emb = nn.Embedding(1, emb_dim)
        self.symbol_emb = nn.Embedding(len(self.symbols), emb_dim)
        self.action_emb = nn.Embedding(len(self.action_to_aid), emb_dim)

    def _create_grammar(self, manifesto_path):
        cur_line = 0

        def parse_current_state(line, prev_state):
            if SYMBOL_START in line.lower():
                prev_state = SYMBOL_STATE
            elif SYMBOL_END in line.lower():
                prev_state = None
            elif GRAMMAR_START in line.lower():
                prev_state = GRAMMAR_STATE
            elif GRAMMAR_END in line.lower():
                prev_state = None
            return prev_state

        def parse_symbol(line):
            assert (
                len(line.split(" ")) == 1
            ), "Symbol format error in line {}: {}".format(cur_line, line)
            if line in self.symbols:
                print("Repeated symbol in line {}: {}".format(cur_line, line))
            else:
                self.symbols += [line]

        def parse_action(line):
            assert ACTION_SIGN in line, "Action Format error in line {}: {}".format(
                cur_line, line
            )
            left_symbol, right_symbols = line.split(ACTION_SIGN)
            actions = right_symbols.strip(" ").split(" | ")
            # Check left symbol
            assert (
                left_symbol in self.symbols
            ), "Undeclared symbol '{}' in line {}".format(left_symbol, cur_line)
            assert (
                left_symbol not in self.actions.keys()
            ), "Overwriting previous action rule in line {}".format(cur_line)

            # Check Right symbols are declared
            for symbols in actions:
                for symbol in symbols.split(" "):
                    assert (
                        symbol in self.symbols
                    ), "Undeclared symbol '{}' in line {}".format(symbol, cur_line)
            self.actions[left_symbol] = actions

            # Set Start Symbol
            if not self.start_symbol:
                self.start_symbol = left_symbol

        # Parse Manifesto File
        with open(manifesto_path, "r") as f:
            lines = f.readlines()
            cur_state = None
            for line in lines:
                cur_line += 1
                line = line.split("#")[0].strip()
                # Pass empty lines
                if not line:
                    continue
                # Parse state
                state = parse_current_state(line, cur_state)
                if cur_state != state:
                    cur_state = state
                else:
                    # Parse definition
                    if cur_state == SYMBOL_STATE:
                        parse_symbol(line)
                    elif cur_state == GRAMMAR_STATE:
                        parse_action(line)

        # Action List
        action_list = []
        for key in self.actions.keys():
            for idx in range(len(self.actions[key])):
                action_list += [(key, idx)]

        # action-to-id
        self.action_to_aid = {action_list[idx]: idx for idx in range(len(action_list))}

        # id-to-action
        self.aid_to_action = {idx: action_list[idx] for idx in range(len(action_list))}

        # symbol_id_to_symbol
        self.sid_to_symbol = {idx: symbol for idx, symbol in enumerate(self.symbols)}

        # symbol_to_symbol_id
        self.symbol_to_sid = {symbol: idx for idx, symbol in enumerate(self.symbols)}

    # Symbols to symbol ids
    def symbols_to_sids(self, symbols):
        return [self.symbol_to_sid[symbol] for symbol in symbols]

    # action - symbol Translation
    def str_to_action(self, action_str):
        symbol, id = action_str.split("(")
        id = int(id.split(")")[0])
        return (symbol, id)

    # Get next action
    def get_possible_aids(self, symbol):
        if isinstance(symbol, int):
            symbol = self.sid_to_symbol[symbol]
        return [
            self.action_to_aid[(symbol, idx)]
            for idx in range(len(self.actions[symbol]))
        ]

    # Parse and get nonterminals
    def parse_nonterminal_symbols(self, actions: List[Action]):
        nonterminals = []
        for action in actions:
            nonterminals += [self.parse_nonterminal_symbol(action)]
        return nonterminals

    def parse_nonterminal_symbol(self, action: Action):
        symbol, id = action
        if symbol in ["T", "C"]:
            id = 0
        return [
            item
            for item in self.actions[symbol][id].split(" ")
            if item not in self.terminals
        ]

    # Get ALL
    def get_all_symbols(self):
        return self.symbols

    def get_all_actions(self):
        return self.actions

    # ETC
    def get_start_symbol(self):
        return self.start_symbol

    def get_action_len(self):
        return len(self.action_to_aid)

    def get_key_emb(self):
        return self.key_emb.weight.data.squeeze(0)


if __name__ == "__main__":
    grammar = Grammar(
        "/home/hkkang/debugging/irnet_qgm_transformer/rule/preprocess/preprocess.manifesto"
    )
    stop = None
    pass
