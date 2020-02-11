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
