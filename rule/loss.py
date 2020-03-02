import torch


class Loss:
    def __init__(self):
        keys = [
            "head_agg",
            "head_col",
            "head_num",
            "predicate_agg",
            "predicate_col",
            "predicate_op",
            "predicate_num",
            "quantifier_num",
            "quantifier_tab",
            "sketch",
            "detail",
            "total",
        ]
        self.loss_dic = {key: torch.tensor(0.0).cuda() for key in keys}

    def add(self, value, action_node, prev_actions):
        key = self._get_key(action_node, prev_actions)
        assert key in self.loss_dic.keys(), "key:{}".format(key)
        self.loss_dic[key] += value
        if "col" in key or "agg" in key or "tab" in key:
            self.loss_dic["detail"] += value
        else:
            self.loss_dic["sketch"] += value
        self.loss_dic["total"] += value

    def get_dic(self):
        return self.loss_dic

    def get_keys(self):
        return self.loss_dic.keys()

    def get_loss_of(self, key):
        return self.loss_dic[key]

    def get_total_loss(self):
        return self.get_loss("total")
