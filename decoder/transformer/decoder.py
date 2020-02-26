import math
import torch
import torch.nn as nn
import decoder.utils as utils
from rule.semql.semql import SemQL
from decoder.transformer.batch_state import Transformer_Batch_State
from src.transformer_decoder import (
    TransformerDecoderLayer,
    TransformerDecoder,
)

class Transformer_Decoder(nn.Module):
    def __init__(self, cfg):
        super(Transformer_Decoder, self).__init__()
        is_bert = cfg.is_bert

        nhead = cfg.nhead
        layer_num = cfg.layer_num
        hidden_size = cfg.hidden_size

        #Decode Layers
        dim = 1024 if is_bert else hidden_size
        self.dim = dim
        self.nhead = nhead
        self.layer_num = layer_num

        decoder_layer = TransformerDecoderLayer(d_model=dim, nhead=nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=layer_num)
        self._init_positional_embedding(dim)

        self.action_affine_layer = nn.Linear(dim, dim)
        self.symbol_affine_layer = nn.Linear(dim, dim)
        self.col_affine_layer = nn.Linear(dim, dim)
        self.tab_affine_layer = nn.Linear(dim, dim)
        self.tgt_linear_layer = nn.Linear(dim*2, dim)
        self.out_linear_layer = nn.Linear(dim, dim)

        self.grammar = SemQL(dim)
        self.col_symbol_id = self.grammar.symbol_to_sid["C"]
        self.tab_symbol_id = self.grammar.symbol_to_sid["T"]
        self.start_symbol_id = self.grammar.symbol_to_sid[self.grammar.start_symbol]


    def _init_positional_embedding(self, d_model, dropout=0.1, max_len=100):
        self.pos_dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def pos_encode(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.pos_dropout(x)

    def forward(
        self,
        encoded_src,
        encoded_col,
        encoded_tab,
        src_mask,
        col_mask,
        tab_mask,
        col_tab_dic,
        tab_col_dic,
        golds,
    ):
        init_b_size = encoded_src.shape[0]
        start_symbol_ids = [[self.start_symbol_id] for _ in range(init_b_size)]

        # Mask Batch State
        state = Transformer_Batch_State(
                    torch.arange(init_b_size).long(),
                    encoded_src,
                    encoded_col,
                    encoded_tab,
                    src_mask,
                    col_mask,
                    tab_mask,
                    col_tab_dic,
                    tab_col_dic,
                    golds,
                    start_symbol_ids,
                )

        # Set Starting conditions
        losses = [self.grammar.create_loss_object() for _ in range(init_b_size)]
        pred_histories = [[] for  _ in range(init_b_size)]
        state.set_state(0, losses, pred_histories)

        # Decode
        while not state.is_done():
            # Get prev actions embedding
            prev_actions = state.get_prev_actions()
            prev_action_emb = []
            for idx, actions in enumerate(prev_actions):
                embs = [torch.zeros(self.grammar.action_emb.embedding_dim).cuda()]
                for action in actions:
                    if action[0] == "C":
                        col_idx = action[1]
                        embs += [state.get_encoded_col()[idx][col_idx]]
                    elif action[0] == "T":
                        tab_idx = action[1]
                        embs += [state.get_encoded_tab()[idx][tab_idx]]
                    else:
                        aid = torch.tensor(self.grammar.action_to_aid[action]).long().cuda()
                        embs += [self.grammar.action_emb(aid)]
                prev_action_emb += [torch.stack(embs, dim=0)]
            prev_action_emb = torch.stack(prev_action_emb, dim=0)
            tgt_action_emb = self.action_affine_layer(prev_action_emb)

            # Get prev node embeddings
            tgt_symbol_ids = []
            for idx, actions in enumerate(prev_actions):
                sids = [self.grammar.symbol_to_sid[action[0]] for action in actions]
                tgt_symbol_ids += [sids]

            # Append nonterminals
            for idx, value in enumerate(state.get_current_action_node()):
                tgt_symbol_ids[idx] += [int(value)]

            symbol_ids = torch.tensor(tgt_symbol_ids).long().cuda()
            tgt_symbol_emb = self.grammar.symbol_emb(symbol_ids)
            tgt_symbol_emb = self.symbol_affine_layer(tgt_symbol_emb)

            # Create tgt
            tgt = torch.cat([tgt_action_emb, tgt_symbol_emb], dim=-1)
            tgt = self.tgt_linear_layer(tgt).transpose(0, 1)

            # Get attention
            memory = state.get_encoded_src().transpose(0, 1)
            memory_key_padding_mask = state.get_src_mask()

            # Decode
            tgt = self.pos_encode(tgt)
            out = self.transformer_decoder(
                tgt, memory, memory_key_padding_mask=memory_key_padding_mask
            ).transpose(0, 1)
            out = self.out_linear_layer(out[:, -1:, :])

            # Get views
            action_view_indices, column_view_indices, table_view_indices = \
                state.get_view_indices(self.col_symbol_id, self.tab_symbol_id)

            if action_view_indices:
                action_view = state.create_view(action_view_indices)
                action_out = out[action_view_indices]

                # Get last action
                cur_nodes = action_view.get_current_action_node()
                next_aids = [self.grammar.get_possible_aids(int(symbol)) for symbol in cur_nodes]

                action_mask = torch.ones(action_view.get_b_size(), self.grammar.get_action_len()).long().cuda()

                for idx, item in enumerate(next_aids):
                    action_mask[idx][item] = 0

                action_emb = self.grammar.action_emb.weight.unsqueeze(0)
                action_emb = action_emb.repeat(action_view.get_b_size(), 1, 1)
                pred_aid = self.predict(action_view, action_out, action_emb, action_mask, self.action_affine_layer)

                # Append pred_history
                actions = [self.grammar.aid_to_action[aid] for aid in pred_aid]
                action_view.insert_pred_history(actions)

                # Get next nonterminals
                nonterminal_symbols = self.grammar.parse_nonterminal_symbols(actions)
                nonterminal_symbol_ids = [self.grammar.symbols_to_sids(symbols) for symbols in nonterminal_symbols]
                action_view.insert_nonterminals(nonterminal_symbol_ids)


            if column_view_indices:
                column_view = state.create_view(column_view_indices)
                column_out = out[column_view_indices]

                encoded_col = column_view.get_encoded_col()
                col_mask = column_view.get_col_mask()

                pred_col_id = self.predict(column_view, column_out, encoded_col, col_mask, self.col_affine_layer)
                actions = [("C", col_id) for col_id in pred_col_id]
                column_view.insert_pred_history(actions)

                # Get next nonterminals
                nonterminal_symbol_ids = [[self.tab_symbol_id] for _ in range(column_view.get_b_size())]
                column_view.insert_nonterminals(nonterminal_symbol_ids)

            if table_view_indices:
                table_view = state.create_view(table_view_indices)
                table_out = out[table_view_indices]

                encoded_tab = table_view.get_encoded_tab()

                # Get Mask from prev col
                prev_col_id = [item[1] for item in table_view.get_prev_action()]
                tab_mask = torch.ones(table_view.get_b_size(), len(table_view.tab_mask[0])).cuda()
                col_tab_dic = table_view.get_col_tab_dic()

                for idx, col_id in enumerate(prev_col_id):
                    tab_ids = col_tab_dic[idx][col_id]
                    tab_mask[idx][tab_ids] = 0

                pred_tab_id = self.predict(table_view, table_out, encoded_tab, tab_mask, self.tab_affine_layer)
                actions = [("T", table_id) for table_id in pred_tab_id]
                table_view.insert_pred_history(actions)

                # Get next nonterminals
                table_view.insert_nonterminals([[] for _ in range(table_view.get_b_size())])

            # State Transition
            state = state.get_next_state()

        # get losses, preds
        return (
            state.loss,
            state.pred_history,
        )


    def attention(self, src, src_mask, hidden_state):
        hidden_state = hidden_state.unsqueeze(1)
        weights = utils.calculate_similarity(src, hidden_state, source_mask=src_mask, affine_layer=self.tgt_affine_layer, log_softmax=False)

        src = src * weights.unsqueeze(-1)
        src = torch.sum(src, dim=1)

        return src


    def predict(self, view, out, src, src_mask, affine_layer=None):
        # Calculate similarity
        probs = utils.calculate_similarity(
            src, out, source_mask=src_mask, affine_layer=affine_layer
        )

        if self.training:
            golds = view.get_gold()
            if view.nonterminal_stack[0][0] in [self.col_symbol_id, self.tab_symbol_id]:
                pred_indices = [item[1] for item in golds]
            else:
                pred_indices = [self.grammar.action_to_aid[action] for action in golds]
            pred_probs = [probs[idx][item] for idx, item in enumerate(pred_indices)]
        else:
            pred_probs, pred_indices = torch.topk(probs, 1)
            pred_probs = pred_probs.squeeze(1)
            pred_indices = [int(item) for item in pred_indices.squeeze(1)]

        # Append loss
        for idx, value in enumerate(pred_probs):
            cur_symbol = view.nonterminal_stack[idx][0]
            b_idx = view.b_indices[idx]
            prev_actions = view.pred_history[b_idx]
            view.loss[idx].add(-value, cur_symbol, prev_actions)

        return pred_indices
