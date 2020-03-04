from typing import List, Dict, Union, NewType
import torch


class TensorPromise:
    def __init__(self, lazy_module: "LazyModule", index: int):
        self.lazy_module = lazy_module
        self.index = index
        self.result = None

    @classmethod
    def wait_list_promisedict(
        cls, list_promise_dict: List[Dict[str, Union["TensorPromise", torch.Tensor]]]
    ):
        for promise_dict in list_promise_dict:
            for promise_or_tensor in promise_dict.values():
                if isinstance(promise_or_tensor, TensorPromise):
                    promise_or_tensor.lazy_module.wait_if_not_done()

    @classmethod
    def promisedict_to_tensordict(
        cls, promise_dict: Dict[str, Union["TensorPromise", torch.Tensor]]
    ):
        tensor_dict = dict()
        for key, promise_or_tensor in promise_dict.items():
            if isinstance(promise_or_tensor, TensorPromise):
                tensor_dict[key] = promise_or_tensor.fetch()
            else:
                tensor_dict[key] = promise_or_tensor
        return tensor_dict

    def fetch(self) -> torch.Tensor:
        assert self.result is not None
        return self.result
