from typing import List, Dict
import torch
import torch.nn as nn
from rule.semql.semql import SemQL
from rule.grammar import Action, Symbol, SymbolId
from decoder.ra_transformer_framework.state import (
    RATransformerState,
    RATransformerStateGold,
    RATransformerStatePred,
)
from framework.sequential_monad import (
    SequentialMonad,
    WhileLogic,
    LogicUnit,
    TensorPromiseOrTensor,
    TensorPromise,
)
from framework.lazy_modules import (
    LazyLinear,
    LazyRATransformerDecoder,
    LazyCalculateSimilarity,
)


class RATransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super(RATransformerDecoder, self).__init__()
        nhead = cfg.nhead
        layer_num = cfg.layer_num
        hidden_size = cfg.hidden_size
        dim = 1024 if cfg.encoder_name == "bert" else hidden_size

        self.dim = dim
        self.nhead = nhead
        self.layer_num = layer_num

        # Relations
        relation_keys = [
            "padding",
            "identical",
            "same_iden",
            "same_ss",
            "same_dd",
            "same_ds",
            "same_sd",
            "diff_iden",
            "diff_ss",
            "diff_dd",
            "diff_ds",
            "diff_sd",
        ]
        self.relation_dic = {item: idx for idx, item in enumerate(relation_keys)}

        # Inference
        self.ra_transformer = LazyRATransformerDecoder(
            dim, nhead, layer_num, len(relation_keys)
        )

        # Decoding forwards
        self.action_affine_layer = LazyLinear(dim, dim)
        self.symbol_affine_layer = LazyLinear(dim, dim)
        self.tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.decoder_out_linear_layer = LazyLinear(dim, dim)

        # Similarity
        self.action_similarity = LazyCalculateSimilarity(dim, dim)
        self.column_similarity = LazyCalculateSimilarity(dim, dim)
        self.table_similarity = LazyCalculateSimilarity(dim, dim)

        # ETC
        self.grammar = SemQL(dim)
        self.col_symbol_id = self.grammar.symbol_to_sid["C"]
        self.tab_symbol_id = self.grammar.symbol_to_sid["T"]
        self.detail_symbols = ["A", "C", "T"]

    def action_to_embedding(
        self, state: RATransformerState, action: Action
    ) -> torch.Tensor:
        if action[0] == "C":
            col_idx = action[1]
            return state.encoded_col[col_idx]
        elif action[0] == "T":
            tab_idx = action[1]
            return state.encoded_tab[tab_idx]
        else:
            action_idx = torch.tensor(self.grammar.action_to_aid[action]).long().cuda()
            return self.grammar.action_emb(action_idx)

    def symbol_list_to_embedding(self, symbol_list: List[Symbol]) -> torch.Tensor:
        symbol_ids_list: List[SymbolId] = [
            self.grammar.symbol_to_sid[symbol] for symbol in symbol_list
        ]
        symbol_ids_tensor = torch.tensor(symbol_ids_list).long().cuda()
        symbol_embeddings = self.grammar.symbol_emb(symbol_ids_tensor)
        return symbol_embeddings

    def parse_relation(self, actions: List[Action]) -> List[List[int]]:
        # def parse_dependency(actions: List[Action]):
        #     nonterminals = []
        #     actions_with_dependency = []
        #     for idx, action in enumerate(actions):
        #
        #
        #     # While(
        # actions_with_dependency = parse_dependency(actions)
        return None
        # Parse Relation
        relations: List[List[int]] = []
        for idx1, action1 in enumerate(actions_with_dependency):
            tmp: List[int] = []
            for idx2, action2 in enumerate(actions_with_dependency):
                # Compare parent
                if action1[2] == action2[2]:
                    parent = "same"
                else:
                    parent = "diff"

                # Compare Identity
                if idx1 == idx2:
                    key = "identical"
                elif action1[0] == action2[0]:
                    key = "{}_identical".format(parent)
                else:
                    # Left
                    if action1[0] in self.detail_symbols:
                        left = "d"
                    else:
                        left = "s"
                    # Right
                    if action2[0] in self.detail_symbols:
                        right = "d"
                    else:
                        right = "s"
                    key = "{}_{}{}".format(parent, left, right)
                tmp.append(self.relation_dic[key])
            relations += [tmp]

        return relations

    def onedim_zero_tensor(self):
        return torch.zeros(self.dim).cuda()

    def forward(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, golds=None):
        b_size = len(encoded_src)

        if golds:
            state_class = RATransformerStateGold
            states = [
                RATransformerStateGold(
                    encoded_src[b_idx],
                    encoded_col[b_idx],
                    encoded_tab[b_idx],
                    col_tab_dic[b_idx],
                    golds[b_idx],
                )
                for b_idx in range(b_size)
            ]
            pass
        else:
            state_class = RATransformerStatePred
            states = [
                RATransformerStatePred(
                    encoded_src[b_idx],
                    encoded_col[b_idx],
                    encoded_tab[b_idx],
                    col_tab_dic[b_idx],
                    self.grammar.start_symbol,
                )
                for b_idx in range(b_size)
            ]

        def embed_history_actions(
            state: RATransformerState, prev_tensor_dic: Dict[str, TensorPromiseOrTensor]
        ) -> Dict[str, TensorPromise]:
            history_actions: List[Action] = state.get_history_actions()
            history_action_embeddings: List[torch.Tensor] = [
                self.action_to_embedding(state, action) for action in history_actions
            ]
            current_action_embedding = self.onedim_zero_tensor()
            history_action_embeddings += [current_action_embedding]
            action_embeddings = torch.stack(history_action_embeddings, dim=0)
            action_embeddings_promise: TensorPromise = self.action_affine_layer.forward_later(
                action_embeddings
            )
            return {"action_embedding": action_embeddings_promise}

        def embed_history_symbols(
            state: RATransformerState, prev_tensor_dic: Dict[str, TensorPromiseOrTensor]
        ) -> Dict[str, TensorPromise]:
            history_symbols: List[Symbol] = state.get_history_symbols()
            current_symbol: Symbol = state.get_current_symbol()
            history_symbols += [current_symbol]
            symbol_embeddings = self.symbol_list_to_embedding(history_symbols)
            symbol_embeddings_promise: TensorPromise = self.symbol_affine_layer.forward_later(
                symbol_embeddings
            )
            new_tensor_dict: Dict[str, TensorPromiseOrTensor] = dict()
            new_tensor_dict.update(prev_tensor_dic)
            new_tensor_dict.update({"symbol_embedding": symbol_embeddings_promise})
            return new_tensor_dict

        def combine_embeddings(
            state: RATransformerState, prev_tensor_dic: Dict[str, TensorPromiseOrTensor]
        ):
            action_embedding: torch.Tensor = prev_tensor_dic["action_embedding"].result
            symbol_embedding: torch.Tensor = prev_tensor_dic["symbol_embedding"].result
            combined_embedding: torch.Tensor = torch.cat(
                (action_embedding, symbol_embedding), dim=-1
            )
            combined_embedding_promise: TensorPromise = self.tgt_linear_layer.forward_later(
                combined_embedding
            )
            prev_tensor_dic.update({"combined_embedding": combined_embedding_promise})
            return prev_tensor_dic

        def embed_relations(
            state: RATransformerState, prev_tensor_dic: Dict[str, TensorPromiseOrTensor]
        ):
            actions = state.get_history_actions()
            relation_matrix = self.parse_relation(actions)
            prev_tensor_dic.update({"relation": relation_matrix})
            return prev_tensor_dic

        def pass_ratransformer(
            state: RATransformerState, prev_tensor_dic: Dict[str, TensorPromiseOrTensor]
        ):
            combined_embedding: torch.Tensor = prev_tensor_dic[
                "combined_embedding"
            ].result
            relation: List[List[int]] = prev_tensor_dic["relation"]

            src_embedding: torch.Tensor = state.encoded_src
            decoder_out_promise: TensorPromise = self.ra_transformer.forward_layer(
                combined_embedding, src_embedding, relation
            )
            prev_tensor_dic.update({"decoder_out": decoder_out_promise})
            return prev_tensor_dic

        def calc_prod(
            state: RATransformerState, prev_tensor_dic: Dict[str, TensorPromiseOrTensor]
        ):
            def calc_prod_with_idx_and_symbol(idx, symbol):
                if symbol == "C":
                    prod = self.column_similarity.forward_later(
                        decoder_out[idx], state.encoded_col, None
                    )
                elif symbol == "T":
                    prod = self.table_similarity.forward_later(
                        decoder_out[idx],
                        state.encoded_tab,
                        state.impossible_table_indices(idx),
                    )
                else:
                    # Get possible actions from nonterminal stack
                    possible_action_ids = SemQL.semql.get_possible_aids(symbol)
                    impossible_indices = [
                        idx
                        for idx in range(SemQL.semql.get_action_len())
                        if idx not in possible_action_ids
                    ]

                    prod = self.action_similarity.forward_later(
                        decoder_out[idx],
                        self.grammar.action_emb.weight,
                        impossible_indices,
                    )
                return prod

            decoder_out: torch.Tensor = prev_tensor_dic["decoder_out"].result
            promise_prods: List[TensorPromise] = []

            history_symbols: List[Symbol] = state.get_history_symbols()
            current_symbol: Symbol = state.get_current_symbol()
            if current_symbol:
                history_symbols += [current_symbol]
            for idx, symbol in enumerate(history_symbols):
                prod = calc_prod_with_idx_and_symbol(idx, symbol)
                promise_prods.append(prod)
            prev_tensor_dic.update({"infer_prods": promise_prods})
            return prev_tensor_dic

        def apply_prod(
            state: RATransformerState, prev_tensor_dic: Dict[str, TensorPromiseOrTensor]
        ):
            prev_tensor_list = prev_tensor_dic["infer_prods"]
            if state.is_gold():
                if prev_tensor_list:
                    idx = len(prev_tensor_list) - 1
                    prod = prev_tensor_list[-1].result
                    state.apply_loss(idx, prod)
            else:
                prod = prev_tensor_list[-1].result
                state.apply_pred(prod)
            state.step()
            return prev_tensor_dic

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done).Do(
                LogicUnit.If(state_class.is_not_done)
                .Then(embed_history_actions)
                .Then(embed_history_symbols)
                .Then(combine_embeddings)
                .Then(embed_relations)
                .Then(pass_ratransformer)
                .Then(calc_prod)
                .Then(apply_prod)
            )
        ).states

        if golds:
            return RATransformerStateGold.combine_loss(states)
        else:
            return RATransformerStatePred.get_preds(states)
