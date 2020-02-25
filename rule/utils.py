def get_symbol_indices(qgm_action, symbol):
    symbol_indices = [item[1] for item in qgm_action if item[0] == symbol]
    return symbol_indices

def count_symbol(qgm_action, symbol):
    symbols = [item[0] for item in qgm_action]
    return symbols.count(symbol)


def filter_action(qgm_action, target_symbol, prev_symbols):
    SkETCH_SYMBOLS = ["A", "C", "T"]
    prev_symbols = list(reversed(prev_symbols))

    actions = []
    for idx, action in enumerate(qgm_action):
        if action[0] == target_symbol:
            past_actions = [ item for item in list(reversed(qgm_action[:idx])) if item[0] not in SkETCH_SYMBOLS]
            if len(past_actions) >= len(prev_symbols):
                if past_actions[:len(prev_symbols)] == prev_symbols:
                    actions += [action]
    return actions
