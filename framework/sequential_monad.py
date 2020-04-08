from typing import Callable, Type, Dict, Any, Iterable, Tuple, Union, NewType, List
from abc import ABC
from framework.promise import TensorPromise
from framework.lazy_modules import LazyModule
import torch


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

    def bind(self, state_iter, list_prev_return):
        true_indices = [
            idx for idx, state in enumerate(state_iter) if self.state_checker(state)
        ]
        checked_states = [
            state for idx, state in enumerate(state_iter) if idx in true_indices
        ]
        checked_prev_return = [
            prev_return
            for idx, prev_return in enumerate(list_prev_return)
            if idx in true_indices
        ]

        for tensor_chain in self.tensor_chains:
            tmp = self._invoke_tensorchain(
                tensor_chain, checked_states, checked_prev_return
            )
            checked_prev_return = tmp

        for idx, new_return_value in zip(true_indices, checked_prev_return):
            list_prev_return[idx] = new_return_value

        return list_prev_return

    def _invoke_tensorchain(
        self, callback: TensorChain, states: List[State], list_prev_return: List,
    ):
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
        prev_return = [dict() for _ in state_iter]

        def is_not_done():
            check_results = [self.stop_if_all_false(state) for state in state_list]
            return any(check_results)

        while is_not_done():
            for idx, logic_unit in enumerate(self.logic_units):
                prev_return = logic_unit.bind(state_list, prev_return)


class SequentialMonad:
    def __init__(self, state_iter):
        self.states = [state for state in state_iter]

    def __call__(self, logic: Union[WhileLogic, LogicUnit]) -> "SequentialMonad":
        if isinstance(logic, WhileLogic):
            logic.bind(self.states)
        else:
            logic.bind(self.states, [None] * len(self.states))
        return self
