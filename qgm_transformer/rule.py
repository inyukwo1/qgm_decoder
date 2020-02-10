class Grammar:
    def __init__(self, manifesto_path):
        self.symbols = {}
        self.actions = {}
        self.terminals = []
        self.action_to_id = {}
        self.id_to_action = {}
        self.start_symbol = None
        self._create_grammar(manifesto_path)
        self.terminals = [key for key in self.symbols.keys() if key not in self.actions.keys()]

    def _create_grammar(self, manifesto_path):
        cur_line = 0
        symbol_flag = "symbol"
        grammar_flag = "grammar"
        symbol_sign = " = "
        action_sign = " ::= "

        def parse_current_state(line, prev_state):
            if "Symbol_start" in line:
                prev_state = symbol_flag
            elif "Symbol_end" in line:
                prev_state = None
            elif "Grammar_start" in line:
                prev_state = grammar_flag
            elif "Grammar_end" in line:
                prev_state = None
            return prev_state

        def parse_symbol(line):
            assert symbol_sign in line, "Symbol format error in line {}".format(cur_line)
            symbol, symbol_name = line.split(symbol_sign)
            self.symbols[symbol] = symbol_name

        def parse_action(line):
            assert action_sign in line, "Action Format error in line {}".format(cur_line)
            left_symbol, right_symbols = line.split(action_sign)
            actions = right_symbols.strip(" ").split(" | ")
            assert left_symbol not in self.actions.keys(), "Overwriting previous action rule in line {}".format(cur_line)
            self.actions[left_symbol] = actions
            # Set Start Symbol
            if not self.start_symbol:
                self.start_symbol = left_symbol

        # Parse Manifesto File
        with open(manifesto_path, 'r') as f:
            lines = f.readlines()
            cur_state = None
            for line in lines:
                cur_line += 1
                line = line.strip("\n")
                # Pass empty lines
                if not line:
                    continue
                # Parse state
                state = parse_current_state(line, cur_state)
                if cur_state != state:
                    cur_state = state
                    continue
                # Parse definition
                if cur_state == symbol_flag:
                    parse_symbol(line)
                elif cur_state == grammar_flag:
                    parse_action(line)

        # Action List
        action_list = []
        for key in self.actions.keys():
            for idx in range(len(self.actions[key])):
                action_list += [(key, idx)]

        # action-to-id
        self.action_to_id = {action_list[idx]: idx for idx in range(len(action_list))}

        # id-to-action
        self.id_to_action = {idx: action_list[idx] for idx in range(len(action_list))}

    def action_to_string(self, symbol, idx):
        return "{} ::= {}".format(symbol, self.actions[symbol][idx])

    def get_action_id(self, symbol, idx):
        self.action_to_id[(symbol, idx)]

    def get_next_action(self, symbol, idx):
        return self.actions[symbol][idx]

    def get_next_actions(self, symbol):
        return self.actions[symbol]

    def get_all_symbols(self):
        return self.symbols

    def get_all_actions(self):
        return self.actions


if __name__ == "__main__":
    mani_path = "/home/hkkang/debugging/irnet_qgm_transformer/qgm_transformer/qgm.manifesto"
    grammar = Grammar(mani_path)
    stop = 1
