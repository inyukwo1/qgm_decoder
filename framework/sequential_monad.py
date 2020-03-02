from typing import Callable, Type, Dict, Any, Iterable, Tuple, Union
from abc import ABC, abstractmethod
import torch
from torch import nn


class StateUpdater:
    @classmethod
    def __call__(cls, any: Any) -> None:
        pass


class TensorChain:
    @classmethod
    def __call__(cls, state: State, tensordict: Dict[str, torch.Tensor]=None) -> Dict[str, TensorPromise]:
        pass

class StateChecker:
    @classmethod
    def __call__(cls, state: State) -> bool:


class LogicUnit:
    def __init__(self, state_checker: Type[StateChecker]):
        self.state_checker = state_checker
        self.tensor_chain = []
        self.state_updater = None

    def __call__(self, tensorchain_or_stateupdater: Union[Type[TensorChain], Type[StateUpdater]]) -> 'LogicUnit':
        if issubclass(tensorchain_or_stateupdater, TensorChain):
            tensorchain = tensorchain_or_stateupdater
            self.tensor_chain.append(tensorchain)
        else:
            stateupdater = tensorchain_or_stateupdater
            self.state_updater = stateupdater

    def bind(self, state_iter):





class State(ABC):
    pass


class StateTransitioner(ABC):
    @classmethod
    @abstractmethod
    def check_state(cls, state) -> bool:
        pass

    @classmethod
    @abstractmethod
    def compute_state(cls, state: State) -> Dict[str, TensorPromise]:
        pass

    @classmethod
    @abstractmethod
    def update_state(cls, result: Dict[str, Any]) -> State:
        pass


class SequentialMonad:
    def __init__(self, state_iter):
        self.states = [state for state in state_iter]

    def bind(self, state_func: Callable) -> "SequentialMonad":
        updated_state_index_iter = state_func(self.states)
        for updated_state, state_idx in updated_state_index_iter:
            self.states[state_idx] = updated_state
        return self

    def __call__(
        self, state_transitioner: Type[StateTransitioner]
    ) -> "SequentialMonad":
        return self.bind(state_transitioner_binder(state_transitioner))


def state_transitioner_binder(state_transitioner: Type[StateTransitioner]) -> Callable:
    def state_transition(states) -> Iterable[Tuple[State, int]]:
        checked_indices = [
            b_idx
            for b_idx, state in enumerate(states)
            if state_transitioner.check_state(state)
        ]
        computed_promise_dicts = [
            state_transitioner.compute_state(states[b_idx]) for b_idx in checked_indices
        ]
        used_lazy_modules = [
            promise.lazy_module
            for promise_dict in computed_promise_dicts
            for promise in promise_dict.values()
        ]
        for lazy_module in used_lazy_modules:
            lazy_module.compute_if_not_done()

        def promise_dict_to_fetched_dict(promise_dict: Dict[str, TensorPromise]):
            fetched_dict = dict()
            for key, promise in promise_dict.items():
                fetched_dict[key] = promise.fetch()
            return fetched_dict

        fetched_dicts = map(promise_dict_to_fetched_dict, computed_promise_dicts)
        updated_states = list(map(state_transitioner.update_state, fetched_dicts))
        for lazy_module in used_lazy_modules:
            lazy_module.reset()
        return zip(updated_states, checked_indices)

    return state_transition
