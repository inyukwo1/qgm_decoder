import torch
from typing import List, Tuple


def assert_dim(dim, tensor: torch.Tensor) -> None:
    tensor_dim = list(tensor.size())
    assert len(dim) == len(tensor_dim), "expected: {} real: {}".format(dim, tensor_dim)
    for expected, real in zip(dim, tensor_dim):
        if expected is not None:
            assert expected == real, "expected: {} real: {}".format(dim, tensor_dim)


def stack_sequential_tensor_with_mask(
    sequential_tensor_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    length_list = [len(tensor) for tensor in sequential_tensor_list]
    max_length = max(length_list)
    first_tensor = sequential_tensor_list[0]
    padded_tensor_list: List[torch.Tensor] = []
    mask = torch.ones(
        (len(sequential_tensor_list), max_length),
        dtype=torch.bool,
        device=first_tensor.device,
    )
    if len(first_tensor.size()) == 1:
        for idx, tensor in enumerate(sequential_tensor_list):
            padding = torch.zeros(
                (max_length - len(tensor)), dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor = torch.cat((tensor, padding), dim=0)
            padded_tensor_list.append(padded_tensor)
            mask[idx, : len(tensor)] = 0
        stacked_tensor = torch.stack(padded_tensor_list, dim=0)
    elif len(first_tensor.size()) == 2:
        _, embed_dim = list(first_tensor.size())
        for idx, tensor in enumerate(sequential_tensor_list):
            padding = torch.zeros(
                (max_length - len(tensor), embed_dim),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded_tensor = torch.cat((tensor, padding), dim=0)
            padded_tensor_list.append(padded_tensor)
            mask[idx, : len(tensor)] = 0
        stacked_tensor = torch.stack(padded_tensor_list, dim=0)
    else:
        raise NotImplementedError

    return stacked_tensor, mask
