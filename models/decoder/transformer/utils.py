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


def calculate_attention_weights(source, query, source_mask=None, affine_layer=None):
    if affine_layer:
        source = affine_layer(source)
    weight_scores = torch.bmm(source, query.transpose(1, 2)).squeeze(-1)

    # Masking
    if source_mask is not None:
        weight_scores.data.masked_fill_(source_mask.bool(), -float("inf"))

    weight_probs = torch.log_softmax(weight_scores, dim=-1)

    return weight_probs
