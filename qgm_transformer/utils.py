import torch


def array_to_tensor(batch_array):
    b_size = len(batch_array)
    max_len = max([len(item) for item in batch_array])

    tensor = torch.zeros((b_size, max_len), dtype=torch.long)

    for idx in range(b_size):
        length = len(batch_array[idx])
        tensor[idx][:length] = torch.tensor(batch_array[idx])

    return tensor.cuda()


def to_long_tensor(batch_list):
    return torch.tensor(batch_list, dtype=torch.long).cuda()


def calculate_attention_weights(source, query, source_mask=None, affine_layer=None, log_softmax=True):
    if affine_layer:
        source = affine_layer(source)
    weight_scores = torch.bmm(source, query.transpose(1, 2)).squeeze(-1)

    # Masking
    if source_mask is not None:
        weight_scores.data.masked_fill_(source_mask.bool(), -float("inf"))

    if log_softmax:
        weight_probs = torch.log_softmax(weight_scores, dim=-1)
    else:
        weight_probs = torch.softmax(weight_scores, dim=-1)

    return weight_probs


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
