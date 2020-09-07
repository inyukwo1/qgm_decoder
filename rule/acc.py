from qgm.qgm_action import QGM_ACTION


class Acc:
    def __init__(self):
        self.correct_dict = {
            symbol: 0
            for symbol, _ in QGM_ACTION.get_instance().state_action_dict.items()
        }
        self.correct_dict.update({"total": 0})
        self.cnt_dict = {
            symbol: 0
            for symbol, _ in QGM_ACTION.get_instance().state_action_dict.items()
        }
        self.cnt_dict.update({"total": 0})

    def correct(self, symbol):
        self.cnt_dict[symbol] += 1
        self.correct_dict[symbol] += 1

    def wrong(self, symbol):
        self.cnt_dict[symbol] += 1

    def get_acc_dict(self):
        acc_dict = dict()
        for key in self.correct_dict:
            if self.cnt_dict[key] == 0:
                acc_dict[key] = 0
            else:
                acc_dict[key] = self.correct_dict[key] / self.cnt_dict[key]
        return acc_dict

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if other == 0:
            return self
        new_acc = Acc()
        for key in self.correct_dict:
            new_acc.correct_dict[key] = self.correct_dict[key] + other.correct_dict[key]
            new_acc.cnt_dict[key] = self.cnt_dict[key] + other.cnt_dict[key]
        return new_acc
