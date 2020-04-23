from typing import List, Dict, Union
import copy
import torch
import torch.nn as nn
from rule.semql.semql import SemQL
from rule.grammar import Action, Symbol, SymbolId
from decoder.transformer_framework.state import (
    TransformerStateGold,
    TransformerStatePred,
    TransformerState,
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
    LazyTransformerDecoder,
    LazyRATransformerDecoder,
    LazyCalculateSimilarity,
)
import random


class TransformerDecoderFramework(nn.Module):
    def __init__(self, cfg, perturb=False):
        super(TransformerDecoderFramework, self).__init__()
        is_bert = cfg.is_bert

        nhead = cfg.nhead
        layer_num = cfg.layer_num
        hidden_size = cfg.hidden_size

        # Decode Layers
        dim = 1024 if is_bert else hidden_size
        self.dim = dim
        self.nhead = nhead
        self.layer_num = layer_num
        self.use_ct_loss = cfg.use_ct_loss
        self.use_arbitrator = cfg.use_arbitrator
        self.refine_all = cfg.refine_all
        self.perturb = perturb

        if self.refine_all:
            assert self.use_ct_loss == False, "Should be false"

        # For inference
        self.infer_transformer = (
            LazyRATransformerDecoder(dim, nhead, layer_num, 10)
            if perturb
            else LazyTransformerDecoder(dim, nhead, layer_num)
        )
        self.infer_action_affine_layer = LazyLinear(dim, dim)
        self.infer_symbol_affine_layer = LazyLinear(dim, dim)
        self.infer_tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.infer_out_linear_layer = LazyLinear(dim, dim)
        self.infer_action_similarity = LazyCalculateSimilarity(dim, dim)
        self.infer_column_similarity = LazyCalculateSimilarity(dim, dim)
        self.infer_table_similarity = LazyCalculateSimilarity(dim, dim)
        self.grammar = SemQL(dim)

        self.col_symbol_id = self.grammar.symbol_to_sid["C"]
        self.tab_symbol_id = self.grammar.symbol_to_sid["T"]

    def action_to_embedding(
        self, state: TransformerState, action: Action
    ) -> torch.Tensor:
        if action[0] == "C":
            col_idx = action[1]
            return state.get_encoded_col()[col_idx]
        elif action[0] == "T":
            tab_idx = action[1]
            return state.get_encoded_tab()[tab_idx]
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

    def onedim_zero_tensor(self):
        return torch.zeros(self.dim).cuda()

    def create_relation_matrix(self, actions, cur_idx=None):
        relation = torch.ones(len(actions), len(actions)).cuda().long()
        relation[:cur_idx, :cur_idx] = 1
        relation[:cur_idx, cur_idx] = 2
        relation[:cur_idx, cur_idx + 1 :] = 3

        relation[cur_idx, :cur_idx] = 4
        relation[cur_idx, cur_idx] = 5
        relation[cur_idx, cur_idx + 1 :] = 6

        relation[cur_idx + 1 :, :cur_idx] = 7
        relation[cur_idx + 1 :, cur_idx] = 8
        relation[cur_idx + 1 :, cur_idx + 1 :] = 9
        return relation

    def perturbe(self, len_col, len_tab, actions):
        perturbed_actions = []
        for action in actions:
            if random.randrange(0, 100) > 15:
                perturbed_actions.append(action)
            else:
                symbol = action[0]
                if symbol == "C":
                    new_action = ("C", random.randrange(len_col))
                elif symbol == "T":
                    new_action = ("T", random.randrange(len_tab))
                else:
                    new_action = (
                        symbol,
                        random.randrange(len(SemQL.semql.actions[symbol])),
                    )
                perturbed_actions.append(new_action)
        return perturbed_actions

    def forward(
        self,
        b_size,
        encoded_src,
        encoded_col,
        encoded_tab,
        src_lens,
        col_lens,
        tab_lens,
        col_tab_dic,
        golds=None,
        pred_guide=None,
        is_ensemble=False,
        further_pred=None,
        wrapper=False,
    ):
        # Create states
        if golds:
            state_class = TransformerStateGold
            states = [
                TransformerStateGold(
                    encoded_src[b_idx][: src_lens[b_idx]],
                    encoded_col[b_idx][: col_lens[b_idx]],
                    encoded_tab[b_idx][: tab_lens[b_idx]],
                    col_tab_dic[b_idx],
                    golds[b_idx],
                    self.perturbe(col_lens[b_idx], tab_lens[b_idx], golds[b_idx])
                    if self.perturb
                    else None,
                )
                for b_idx in range(b_size)
            ]
        else:
            state_class = TransformerStatePred
            states = [
                TransformerStatePred(
                    encoded_src[b_idx][: src_lens[b_idx]],
                    encoded_col[b_idx][: col_lens[b_idx]],
                    encoded_tab[b_idx][: tab_lens[b_idx]],
                    col_tab_dic[b_idx],
                    SemQL.semql.start_symbol,
                    pred_guide,
                    self.perturbe(col_lens[b_idx], tab_lens[b_idx], further_pred[b_idx])
                    if wrapper
                    else further_pred[b_idx],
                )
                for b_idx in range(b_size)
            ]

        def embed_history_actions(
            state: TransformerState, _
        ) -> Dict[str, TensorPromise]:
            history_actions: List[Action] = state.get_history_actions()
            history_action_embeddings: List[torch.Tensor] = [
                self.action_to_embedding(state, action) for action in history_actions
            ]
            if self.perturb:
                history_action_embeddings[state.step_cnt] = self.onedim_zero_tensor()
            else:
                current_action_embedding = self.onedim_zero_tensor()
                history_action_embeddings += [current_action_embedding]
            action_embeddings = torch.stack(history_action_embeddings, dim=0)
            action_embeddings_promise: TensorPromise = self.infer_action_affine_layer.forward_later(
                action_embeddings
            )
            return {"action_embedding": action_embeddings_promise}

        def embed_history_symbols(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            history_symbols: List[Symbol] = state.get_history_symbols()
            if not self.perturb:
                current_symbol: Symbol = state.get_current_symbol()
                history_symbols += [current_symbol]
            symbol_embeddings = self.symbol_list_to_embedding(history_symbols)
            symbol_embeddings_promise: TensorPromise = self.infer_symbol_affine_layer.forward_later(
                symbol_embeddings
            )
            new_tensor_dict: Dict[str, TensorPromiseOrTensor] = dict()
            new_tensor_dict.update(prev_tensor_dict)
            new_tensor_dict.update({"symbol_embedding": symbol_embeddings_promise})
            return new_tensor_dict

        def combine_embeddings(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            action_embedding: torch.Tensor = prev_tensor_dict["action_embedding"].result
            symbol_embedding: torch.Tensor = prev_tensor_dict["symbol_embedding"].result
            combined_embedding: torch.Tensor = torch.cat(
                (action_embedding, symbol_embedding), dim=-1
            )
            combined_embedding_promise: TensorPromise = self.infer_tgt_linear_layer.forward_later(
                combined_embedding
            )
            prev_tensor_dict.update({"combined_embedding": combined_embedding_promise})
            return prev_tensor_dict

        def pass_infer_transformer(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            combined_embedding: torch.Tensor = prev_tensor_dict[
                "combined_embedding"
            ].result
            src_embedding: torch.Tensor = state.get_encoded_src()

            if self.perturb:
                relation = self.create_relation_matrix(
                    state.get_history_actions(), cur_idx=state.step_cnt,
                )
                decoder_out_promise: TensorPromise = self.infer_transformer.forward_later(
                    combined_embedding, src_embedding, relation
                )
            else:
                decoder_out_promise: TensorPromise = self.infer_transformer.forward_later(
                    combined_embedding, src_embedding
                )
            prev_tensor_dict.update({"decoder_out": decoder_out_promise})
            return prev_tensor_dict

        def pass_infer_out_linear(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result
            decoder_out_promise: TensorPromise = self.infer_out_linear_layer.forward_later(
                decoder_out
            )
            prev_tensor_dict.update({"decoder_out": decoder_out_promise})
            return prev_tensor_dict

        def calc_prod(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> List[TensorPromise]:
            def calc_prod_with_idx_and_symbol(idx, symbol):
                if symbol == "C":
                    prod = self.infer_column_similarity.forward_later(
                        decoder_out[idx], state.get_encoded_col(), None
                    )
                elif symbol == "T":
                    prod = self.infer_table_similarity.forward_later(
                        decoder_out[idx],
                        state.get_encoded_tab(),
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

                    prod = self.infer_action_similarity.forward_later(
                        decoder_out[idx],
                        self.grammar.action_emb.weight,
                        impossible_indices,
                    )
                return prod

            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result

            current_symbol: Symbol = state.get_current_symbol()
            assert current_symbol != None
            promise_prod: TensorPromise = calc_prod_with_idx_and_symbol(
                state.step_cnt, current_symbol
            )
            prev_tensor_dict.update({"infer_prod": promise_prod})
            return prev_tensor_dict

        def apply_prod(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            prod = prev_tensor_dict["infer_prod"].result
            if state.is_gold():
                state.apply_loss(state.step_cnt, prod)
            else:
                assert isinstance(state, TransformerStatePred)
                state.save_probs(prod)
                state.apply_pred(prod)
            state.step()
            return prev_tensor_dict

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done).Do(
                LogicUnit.If(state_class.is_to_infer)
                .Then(embed_history_actions)
                .Then(embed_history_symbols)
                .Then(combine_embeddings)
                .Then(pass_infer_transformer)
                .Then(pass_infer_out_linear)
                .Then(calc_prod)
                .Then(apply_prod)
            )
        ).states

        if golds:
            return TransformerStateGold.combine_loss(states)
        elif is_ensemble:
            return states
        else:
            return TransformerStatePred.get_preds(states)
