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
    TensorPromise,
)
from framework.lazy_modules import (
    LazyLinear,
    LazyTransformerDecoder,
    LazyRATransformerDecoder,
    LazyCalculateSimilarity,
)
import random


class TransformerDecoderModule(nn.Module):
    def __init__(self, dim, nhead, layer_num, use_relation):
        super(TransformerDecoderModule, self).__init__()
        relation_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.use_relation = use_relation
        self.transformer = (
            LazyRATransformerDecoder(dim, nhead, layer_num, len(relation_keys))
            if use_relation
            else LazyTransformerDecoder(dim, nhead, layer_num)
        )
        self.action_affine_layer = LazyLinear(dim, dim)
        self.symbol_affine_layer = LazyLinear(dim, dim)
        self.tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.out_linear_layer = LazyLinear(dim, dim)
        self.action_similarity = LazyCalculateSimilarity(dim, dim)
        self.column_similarity = LazyCalculateSimilarity(dim, dim)
        self.table_similarity = LazyCalculateSimilarity(dim, dim)

        self.symbol_emb = nn.Embedding(SemQL.semql.get_symbol_len(), dim)
        self.action_emb = nn.Embedding(SemQL.semql.get_action_len(), dim)


class TransformerDecoderFramework(nn.Module):
    def __init__(self, cfg):
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
        self.use_relation = cfg.use_relation
        self.look_left_only = cfg.look_left_only
        self.mode = cfg.mode

        if self.refine_all:
            assert self.use_ct_loss is False, "Should be false"

        self.decoders = nn.ModuleDict(
            {
                "infer": TransformerDecoderModule(
                    dim, nhead, layer_num, cfg.mode != "infer"
                )
            }
        )

        self.col_symbol_id = SemQL.semql.symbol_to_sid["C"]
        self.tab_symbol_id = SemQL.semql.symbol_to_sid["T"]

    def action_to_embedding(
        self, state: TransformerState, action: Action, mode,
    ) -> torch.Tensor:
        assert mode in ["infer", "refine", "arbitrate"]
        if action[0] == "C":
            col_idx = action[1]
            return state.get_encoded_col(mode)[col_idx]
        elif action[0] == "T":
            tab_idx = action[1]
            return state.get_encoded_tab(mode)[tab_idx]
        else:
            action_idx = torch.tensor(SemQL.semql.action_to_aid[action]).long().cuda()
            return self.decoders[mode].action_emb(action_idx)

    def symbol_list_to_embedding(self, symbol_list: List[Symbol], mode) -> torch.Tensor:
        assert mode in ["infer", "refine", "arbitrate"]

        symbol_ids_list: List[SymbolId] = [
            SemQL.semql.symbol_to_sid[symbol] for symbol in symbol_list
        ]
        symbol_ids_tensor = torch.tensor(symbol_ids_list).long().cuda()
        symbol_embeddings = self.decoders[mode].symbol_emb(symbol_ids_tensor)
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
        target_step=100,
        states=None,
        preds_init=None,
    ):
        # Create states
        if states:
            state_class = type(states[0])
            for state in states:
                state.target_step = target_step
        elif golds:
            state_class = TransformerStateGold
            states = [
                TransformerStateGold(
                    {"infer": encoded_src[b_idx][: src_lens[b_idx]]},
                    {"infer": encoded_col[b_idx][: col_lens[b_idx]]},
                    {"infer": encoded_tab[b_idx][: tab_lens[b_idx]]},
                    col_tab_dic[b_idx],
                    golds[b_idx],
                    self.perturbe(col_lens[b_idx], tab_lens[b_idx], golds[b_idx]),
                )
                for b_idx in range(b_size)
            ]
        else:
            state_class = TransformerStatePred
            states = [
                TransformerStatePred(
                    {"infer": encoded_src[b_idx][: src_lens[b_idx]]},
                    {"infer": encoded_col[b_idx][: col_lens[b_idx]]},
                    {"infer": encoded_tab[b_idx][: tab_lens[b_idx]]},
                    col_tab_dic[b_idx],
                    SemQL.semql.start_symbol,
                    target_step,
                    self.perturbe(col_lens[b_idx], tab_lens[b_idx], preds_init[b_idx])
                    if self.mode != "infer"
                    else None,
                )
                for b_idx in range(b_size)
            ]

        def embed_history_actions_per_mode(mode):
            def embed_history_actions(
                state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
            ) -> Dict[str, TensorPromise]:
                history_actions: List[Action] = state.get_history_actions(self.mode)
                history_action_embeddings: List[torch.Tensor] = [
                    self.action_to_embedding(state, action, mode)
                    for action in history_actions
                ]
                if self.mode == "infer":
                    history_action_embeddings += [self.onedim_zero_tensor()]
                action_embeddings = torch.stack(history_action_embeddings, dim=0)
                action_embeddings_promise: TensorPromise = self.decoders[
                    mode
                ].action_affine_layer.forward_later(action_embeddings)
                prev_tensor_dict.update({"action_embedding": action_embeddings_promise})
                return prev_tensor_dict

            return embed_history_actions

        def embed_history_symbols_per_mode(mode):
            def embed_history_symbols(
                state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
            ) -> Dict[str, TensorPromise]:
                history_symbols: List[Symbol] = state.get_history_symbols(self.mode)
                current_symbol: Symbol = state.get_current_symbol(self.mode)
                if self.mode == "infer":
                    history_symbols += [current_symbol]
                symbol_embeddings = self.symbol_list_to_embedding(history_symbols, mode)
                symbol_embeddings_promise: TensorPromise = self.decoders[
                    mode
                ].symbol_affine_layer.forward_later(symbol_embeddings)
                prev_tensor_dict.update({"symbol_embedding": symbol_embeddings_promise})
                return prev_tensor_dict

            return embed_history_symbols

        def combine_embeddings_per_mode(mode):
            def combine_embeddings(
                state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
            ) -> Dict[str, TensorPromise]:
                action_embedding: torch.Tensor = prev_tensor_dict[
                    "action_embedding"
                ].result
                symbol_embedding: torch.Tensor = prev_tensor_dict[
                    "symbol_embedding"
                ].result
                combined_embedding: torch.Tensor = torch.cat(
                    (action_embedding, symbol_embedding), dim=-1
                )
                combined_embedding_promise: TensorPromise = self.decoders[
                    mode
                ].tgt_linear_layer.forward_later(combined_embedding)
                prev_tensor_dict.update(
                    {"combined_embedding": combined_embedding_promise}
                )
                return prev_tensor_dict

            return combine_embeddings

        def pass_transformer_per_mode(mode):
            def pass_transformer(
                state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
            ) -> Dict[str, TensorPromise]:
                combined_embedding: torch.Tensor = prev_tensor_dict[
                    "combined_embedding"
                ].result
                src_embedding: torch.Tensor = state.get_encoded_src(mode)
                if self.decoders[mode].use_relation:
                    relation = self.create_relation_matrix(
                        state.get_history_actions(self.mode), cur_idx=state.step_cnt,
                    )
                    decoder_out_promise: TensorPromise = self.decoders[
                        mode
                    ].transformer.forward_later(
                        combined_embedding, src_embedding, relation
                    )
                else:
                    decoder_out_promise: TensorPromise = self.decoders[
                        mode
                    ].transformer.forward_later(combined_embedding, src_embedding)
                prev_tensor_dict.update({"decoder_out": decoder_out_promise})
                return prev_tensor_dict

            return pass_transformer

        def pass_out_linear_per_mode(mode):
            def pass_out_linear(
                state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
            ) -> Dict[str, TensorPromise]:
                decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result
                decoder_out_promise: TensorPromise = self.decoders[
                    mode
                ].out_linear_layer.forward_later(decoder_out)
                prev_tensor_dict.update({"decoder_out": decoder_out_promise})
                return prev_tensor_dict

            return pass_out_linear

        def calc_prod_per_mode(mode):
            def calc_prod(
                state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
            ) -> Dict[str, TensorPromise]:
                def calc_prod_with_idx_and_symbol(idx, symbol):
                    if symbol == "C":
                        prod = self.decoders[mode].column_similarity.forward_later(
                            decoder_out[idx], state.get_encoded_col(mode), None
                        )
                    elif symbol == "T":
                        prod = self.decoders[mode].table_similarity.forward_later(
                            decoder_out[idx],
                            state.get_encoded_tab(mode),
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

                        prod = self.decoders[mode].action_similarity.forward_later(
                            decoder_out[idx],
                            self.decoders[mode].action_emb.weight,
                            impossible_indices,
                        )
                    return prod

                decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result
                if mode == "infer":
                    target_symbol: Symbol = state.get_current_symbol(self.mode)
                    target_cnt = state.step_cnt
                    assert target_symbol is not None
                else:
                    assert False

                promise_prod: TensorPromise = calc_prod_with_idx_and_symbol(
                    target_cnt, target_symbol
                )
                prev_tensor_dict.update({"prod_" + mode: promise_prod})
                return prev_tensor_dict

            return calc_prod

        def apply_prod_infer(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            prod = prev_tensor_dict["prod_infer"].result
            if state.is_gold():
                state.apply_loss(state.step_cnt, prod)
            else:
                assert isinstance(state, TransformerStatePred)
                state.save_probs(prod)
                state.apply_pred(prod, self.mode)
            state.step()
            return prev_tensor_dict

        def save_initial_pred(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            if isinstance(state, TransformerStatePred):
                actions = copy.deepcopy(state.get_history_actions(self.mode))
                assert len(actions) == len(state.preds), "Diff: {} {}".format(
                    len(actions), len(state.preds)
                )
                state.save_init_preds(actions)
            return prev_tensor_dict

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done(self.mode)).Do(
                LogicUnit.If(state_class.is_to_infer(self.mode))
                .Then(embed_history_actions_per_mode("infer"))
                .Then(embed_history_symbols_per_mode("infer"))
                .Then(combine_embeddings_per_mode("infer"))
                .Then(pass_transformer_per_mode("infer"))
                .Then(pass_out_linear_per_mode("infer"))
                .Then(calc_prod_per_mode("infer"))
                .Then(apply_prod_infer)
                .Then(save_initial_pred)
            )
        ).states

        if golds:
            return TransformerStateGold.combine_loss(states)
        else:
            return TransformerStatePred.get_preds(states), states
