from typing import List, Dict
import torch


class TensorPromise:
    def __init__(self, lazy_module: "LazyModule", index: int):
        self.lazy_module = lazy_module
        self.index = index

    @classmethod
    def wait_list_promisedict(cls, list_promise_dict: List[Dict[str, "TensorPromise"]]):
        for promise_dict in list_promise_dict:
            for promise in promise_dict.values():
                promise.lazy_module.wait_if_not_done()

    @classmethod
    def promisedict_to_tensordict(cls, promise_dict: Dict[str, "TensorPromise"]):
        tensor_dict = dict()
        for key, promise in promise_dict.items():
            tensor_dict[key] = promise.fetch()

    def fetch(self) -> torch.Tensor:
        return self.lazy_module.fetch(self.index)
