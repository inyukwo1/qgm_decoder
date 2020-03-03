from typing import Callable, Type, Dict, Any, Iterable, Tuple, Union, NewType, List
from abc import ABC
from framework.promise import TensorPromise
import torch
from torch import nn


class State(ABC):
    pass


StateChecker = NewType("StateChecker", Callable[[State], bool])
TensorChain = NewType(
    "TensorChain", Callable[[State, Dict[str, torch.Tensor]], Dict[str, TensorPromise]]
)
StateUpdater = NewType("StateUpdater", Callable[[State, Dict[str, torch.Tensor]], None])


class LogicUnit:
    def __init__(self, state_checker: Type[StateChecker]):
        self.state_checker = state_checker
        self.tensor_chains = []
        self.state_updater = None

    def __call__(
        self, tensorchain_or_stateupdater: Union[TensorChain, StateUpdater]
    ) -> "LogicUnit":
        if issubclass(tensorchain_or_stateupdater, TensorChain):
            tensorchain = tensorchain_or_stateupdater
            self.tensor_chains.append(tensorchain)
        else:
            stateupdater = tensorchain_or_stateupdater
            self.state_updater = stateupdater
        return self

    def bind(self, state_iter):
        checked_states = [state for state in state_iter if self.state_checker(state)]
        list_tensor_dict = None
        for tensor_chain in self.tensor_chains:
            list_tensor_dict = self._invoke_tensorchain(
                tensor_chain, checked_states, list_tensor_dict
            )
        assert self.state_updater is not None
        for state, tensor_dict in zip(checked_states, list_tensor_dict):
            self.state_updater(state, tensor_dict)

    def _invoke_tensorchain(
        self,
        callback: TensorChain,
        states: List[State],
        list_tensor_dict: List[Dict[str, torch.Tensor]] = None,
    ):
        if list_tensor_dict is None:
            list_promise_dict = [callback(state) for state in states]
        else:
            list_promise_dict = [
                callback(state, tensor_dict)
                for state, tensor_dict in zip(states, list_tensor_dict)
            ]
        TensorPromise.wait_list_promisedict(list_promise_dict)
        new_list_tensor_dict = [
            TensorPromise.promisedict_to_tensordict(promise_dict)
            for promise_dict in list_promise_dict
        ]
        return new_list_tensor_dict


class WhileLogic:
    def __init__(self, stop_if_all_false: StateChecker):
        self.stop_if_all_false = stop_if_all_false
        self.logic_units = []

    def __call__(self, logic_unit: LogicUnit):
        self.logic_units.append(logic_unit)
        return self

    def bind(self, state_iter):
        state_list = [state for state in state_iter]
        check_results = [self.stop_if_all_false(state) for state in state_list]
        while any(check_results):
            for logic_unit in self.logic_units:
                logic_unit.bind(state_list)


class SequentialMonad:
    def __init__(self, state_iter):
        self.states = [state for state in state_iter]

    def __call__(self, logic: Union[WhileLogic, LogicUnit]) -> "SequentialMonad":
        logic.bind(self.states)
        return self
