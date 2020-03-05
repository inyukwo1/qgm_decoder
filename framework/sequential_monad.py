from typing import Callable, Type, Dict, Any, Iterable, Tuple, Union, NewType, List
from abc import ABC
from framework.promise import TensorPromise
from framework.lazy_modules import LazyModule
import torch
import time
from torch import nn


class State(ABC):
    pass


TensorPromiseOrTensor = NewType(
    "TensorPromiseOrTensor", Union[TensorPromise, torch.Tensor]
)

StateChecker = NewType("StateChecker", Callable[[State], bool])
TensorChain = NewType(
    "TensorChain",
    Callable[[State, Dict[str, torch.Tensor]], Dict[str, TensorPromiseOrTensor]],
)


class LogicUnit:
    @classmethod
    def If(cls, state_checker: Type[StateChecker]) -> "LogicUnit":
        return LogicUnit(state_checker)

    def Then(self, tensorchain: TensorChain) -> "LogicUnit":
        return self.__call__(tensorchain)

    def __init__(self, state_checker: Type[StateChecker]):
        self.state_checker = state_checker
        self.tensor_chains = []

    def __call__(self, tensorchain: TensorChain) -> "LogicUnit":
        self.tensor_chains.append(tensorchain)
        return self

    def bind(self, state_iter):
        checked_states = [state for state in state_iter if self.state_checker(state)]
        list_prev_return = None
        for tensor_chain in self.tensor_chains:
            list_prev_return = self._invoke_tensorchain(
                tensor_chain, checked_states, list_prev_return
            )

    def _invoke_tensorchain(
        self, callback: TensorChain, states: List[State], list_prev_return: List = None,
    ):
        if list_prev_return is None:
            list_new_return = [callback(state) for state in states]
        else:
            list_new_return = [
                callback(state, tensor_dict)
                for state, tensor_dict in zip(states, list_prev_return)
            ]
        LazyModule.wait_all()
        return list_new_return


class WhileLogic:
    @classmethod
    def While(cls, stop_if_all_false: StateChecker) -> "WhileLogic":
        return WhileLogic(stop_if_all_false)

    def Do(self, logic_unit: LogicUnit) -> "WhileLogic":
        return self.__call__(logic_unit)

    def __init__(self, stop_if_all_false: StateChecker):
        self.stop_if_all_false = stop_if_all_false
        self.logic_units = []

    def __call__(self, logic_unit: LogicUnit):
        self.logic_units.append(logic_unit)
        return self

    def bind(self, state_iter):
        state_list = [state for state in state_iter]

        def is_not_done():
            check_results = [self.stop_if_all_false(state) for state in state_list]
            return any(check_results)

        while is_not_done():
            for logic_unit in self.logic_units:
                logic_unit.bind(state_list)


class SequentialMonad:
    def __init__(self, state_iter):
        self.states = [state for state in state_iter]

    def __call__(self, logic: Union[WhileLogic, LogicUnit]) -> "SequentialMonad":
        logic.bind(self.states)
        return self
