import torch


def array_to_tensor(batch_array, dtype=None):
    b_size = len(batch_array)
    max_len = max([len(item) for item in batch_array])

    dtype = dtype if dtype else torch.float
    tensor = torch.zeros((b_size, max_len), dtype=dtype)

    for idx in range(b_size):
        length = len(batch_array[idx])
        tensor[idx][:length] = torch.tensor(batch_array[idx])

    return tensor.cuda()


def count_symbol(qgm_action, symbol):
    symbols = [item[0] for item in qgm_action]
    return symbols.count(symbol)


def filter_action(qgm_action, target_symbol, prev_symbols):
    prev_symbols = list(reversed(prev_symbols))
    # Get actions
    actions = []
    for idx, action in enumerate(qgm_action):
        # Is target symbol
        if action[0] == target_symbol:
            # is possible to satisfy condition
            if prev_symbols and idx >= len(prev_symbols):
                # Check if satisfy prev symbol condition
                if False not in [
                    qgm_action[idx - (s_idx + 1)][0] == prev_symbol
                    for s_idx, prev_symbol in enumerate(prev_symbols)
                ]:
                    actions += [action]
    return actions
