import torch


class TreePosEncoder:
    def __init__(self, max_degree, max_height, max_dim, grammar):
        self.grammar = grammar
        self.max_degree = max_degree
        self.max_height = max_height
        self.max_dim = max_dim

    def action_seq_to_pos_encoding(self, action_seq):
        action_seq = action_seq[:]
        node_list = self._action_seq_to_node_list(action_seq)
        encodings = []
        for node in node_list:
            encodings.append(
                node.get_pos_encoding(self.max_degree, self.max_height, self.max_dim)
            )
        return torch.stack(encodings)

    def _action_seq_to_node_list(self, action_seq):
        node_list = []
        self._pop_add_node(action_seq, "B", None, None, node_list)
        return node_list

    def _pop_add_node(
        self, action_seq, expected_symbol, parent_node, child_idx, node_list
    ):
        if action_seq:
            new_action = action_seq[0]
            action_seq.pop(0)
            assert new_action[0] == expected_symbol
            new_node = PosNode(parent_node, child_idx)
            node_list.append(new_node)
            if new_action[0] in {"T", "C"}:
                return
            for idx, symbol in enumerate(self.grammar.get_next_symbols(new_action)):
                if symbol not in self.grammar.actions:
                    continue
                self._pop_add_node(action_seq, symbol, new_node, idx, node_list)
        else:
            new_node = PosNode(parent_node, child_idx)
            node_list.append(new_node)


class PosNode:
    def __init__(self, parent=None, child_idx=None):
        if parent:
            self.seq_to_root = [child_idx] + parent.seq_to_root
        else:
            self.seq_to_root = []

    def get_pos_encoding(self, max_degree, max_height, max_dim):
        encoding = torch.zeros(max_dim)
        encoding_grid = torch.zeros(max_degree * max_height)
        for idx, child_idx in enumerate(self.seq_to_root):
            encoding_grid[idx * max_degree + child_idx] = 1.0

        for idx in range(max_dim // (max_degree * max_height)):
            encoding[
                idx * (max_degree * max_height) : (idx + 1) * (max_degree * max_height)
            ] = encoding_grid
        if torch.cuda.is_available():
            encoding = encoding.cuda()
        return encoding
