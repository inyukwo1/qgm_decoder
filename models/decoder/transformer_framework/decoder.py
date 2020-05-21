from typing import List, Dict, Union
import torch
import torch.nn as nn
from rule.semql.semql import SemQL
from rule.noqgm.noqgm import NOQGM
from rule.grammar import Action, Symbol, SymbolId
from models.decoder.transformer_framework.state import (
    TransformerStateGold,
    TransformerStatePred,
    TransformerState,
)
from models.framework.sequential_monad import (
    SequentialMonad,
    WhileLogic,
    LogicUnit,
    TensorPromiseOrTensor,
    TensorPromise,
)
from models.framework.lazy_modules import (
    LazyLinear,
    LazyTransformerDecoder,
    LazyCalculateSimilarity,
)


class TransformerDecoderFramework(nn.Module):
    def __init__(self, cfg):
        super(TransformerDecoderFramework, self).__init__()
        nhead = cfg.nhead
        layer_num = cfg.layer_num
        hidden_size = cfg.hidden_size

        # Decode Layers
        dim = 1024 if cfg.encoder_name == "bert" else hidden_size
        nhead = 8 if cfg.encoder_name == "bert" else nhead
        self.dim = dim
        self.nhead = nhead
        self.layer_num = layer_num
        self.use_relation = cfg.use_relation

        # For inference
        self.infer_transformer_sel = LazyTransformerDecoder(dim, nhead, layer_num)
        self.infer_action_affine_layer_sel = LazyLinear(dim, dim)
        self.infer_symbol_affine_layer_sel = LazyLinear(dim, dim)
        self.infer_tgt_linear_layer_sel = LazyLinear(dim * 2, dim)
        self.infer_out_linear_layer_sel = LazyLinear(dim, dim)
        self.infer_action_similarity_sel = LazyCalculateSimilarity(dim, dim)
        self.infer_column_similarity_sel = LazyCalculateSimilarity(dim, dim)
        self.infer_table_similarity_sel = LazyCalculateSimilarity(dim, dim)

        self.infer_transformer_filter = LazyTransformerDecoder(dim, nhead, layer_num)
        self.infer_action_affine_layer_filter = LazyLinear(dim, dim)
        self.infer_symbol_affine_layer_filter = LazyLinear(dim, dim)
        self.infer_tgt_linear_layer_filter = LazyLinear(dim * 2, dim)
        self.infer_out_linear_layer_filter = LazyLinear(dim, dim)
        self.infer_action_similarity_filter = LazyCalculateSimilarity(dim, dim)
        self.infer_column_similarity_filter = LazyCalculateSimilarity(dim, dim)
        self.infer_table_similarity_filter = LazyCalculateSimilarity(dim, dim)

        if cfg.rule == "noqgm":
            self.grammar_sel = NOQGM(dim)
            self.grammar_filter = NOQGM(dim)
        else:
            self.grammar_sel = SemQL(dim)
            self.grammar_filter = SemQL(dim)

        self.col_symbol_id = self.grammar_sel.symbol_to_sid["C"]
        self.tab_symbol_id = self.grammar_sel.symbol_to_sid["T"]

    def action_to_embedding_sel(self, state: TransformerState, action: Action):
        if action[0] == "C":
            col_idx = action[1]
            return state.get_encoded_col()[col_idx]
        elif action[0] == "T":
            tab_idx = action[1]
            return state.get_encoded_tab()[tab_idx]
        else:
            action_idx = (
                torch.tensor(self.grammar_sel.action_to_aid[action]).long().cuda()
            )
            return self.grammar_sel.action_emb(action_idx)

    def action_to_embedding_filter(self, state: TransformerState, action: Action):
        if action[0] == "C":
            col_idx = action[1]
            return state.get_encoded_col()[col_idx]
        elif action[0] == "T":
            tab_idx = action[1]
            return state.get_encoded_tab()[tab_idx]
        else:
            action_idx = (
                torch.tensor(self.grammar_sel.action_to_aid[action]).long().cuda()
            )
            return self.grammar_filter.action_emb(action_idx)

    def symbol_list_to_embedding_sel(self, symbol_list: List[Symbol]):
        symbol_ids_list: List[SymbolId] = [
            self.grammar_sel.symbol_to_sid[symbol] for symbol in symbol_list
        ]
        symbol_ids_tensor = torch.tensor(symbol_ids_list).long().cuda()
        return self.grammar_sel.symbol_emb(symbol_ids_tensor)

    def symbol_list_to_embedding_filter(self, symbol_list: List[Symbol]):
        symbol_ids_list: List[SymbolId] = [
            self.grammar_filter.symbol_to_sid[symbol] for symbol in symbol_list
        ]
        symbol_ids_tensor = torch.tensor(symbol_ids_list).long().cuda()
        return self.grammar_filter.symbol_emb(symbol_ids_tensor)

    def onedim_zero_tensor(self):
        return torch.zeros(self.dim).cuda()

    def create_relation_matrix(self, actions, cur_idx=None):
        relation = torch.ones(len(actions), len(actions)).cuda().long()
        for idx in range(cur_idx + 1):
            relation[idx][: cur_idx + 1] = 2
        return relation

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
        return_details=False,
    ):
        # Create states
        if golds:
            state_class = TransformerStateGold
            states = [
                TransformerStateGold(
                    self.grammar_sel,
                    encoded_src[b_idx][: src_lens[b_idx]],
                    encoded_col[b_idx][: col_lens[b_idx]],
                    encoded_tab[b_idx][: tab_lens[b_idx]],
                    col_tab_dic[b_idx],
                    golds[b_idx],
                )
                for b_idx in range(b_size)
            ]
        else:
            state_class = TransformerStatePred
            states = [
                TransformerStatePred(
                    self.grammar_sel,
                    encoded_src[b_idx][: src_lens[b_idx]],
                    encoded_col[b_idx][: col_lens[b_idx]],
                    encoded_tab[b_idx][: tab_lens[b_idx]],
                    col_tab_dic[b_idx],
                )
                for b_idx in range(b_size)
            ]

        def embed_history_actions(
            state: TransformerState, _
        ) -> Dict[str, TensorPromise]:
            history_actions: List[Action] = state.get_history_actions()
            history_action_embeddings_sel: List[torch.Tensor] = [
                self.action_to_embedding_sel(state, action)
                for action in history_actions
            ]
            current_action_embedding_sel = self.onedim_zero_tensor()
            history_action_embeddings_sel += [current_action_embedding_sel]
            action_embeddings_sel = torch.stack(history_action_embeddings_sel, dim=0)
            action_embeddings_sel_promise: TensorPromise = self.infer_action_affine_layer_sel.forward_later(
                action_embeddings_sel
            )

            history_action_embeddings_filter: List[torch.Tensor] = [
                self.action_to_embedding_filter(state, action)
                for action in history_actions
            ]
            current_action_embedding_filter = self.onedim_zero_tensor()
            history_action_embeddings_filter += [current_action_embedding_filter]
            action_embeddings_filter = torch.stack(
                history_action_embeddings_filter, dim=0
            )
            action_embeddings_filter_promise: TensorPromise = self.infer_action_affine_layer_filter.forward_later(
                action_embeddings_filter
            )
            return {
                "action_embedding_sel": action_embeddings_sel_promise,
                "action_embedding_filter": action_embeddings_filter_promise,
            }

        def embed_history_symbols(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            history_symbols: List[Symbol] = state.get_history_symbols()
            current_symbol: Symbol = state.get_current_symbol()
            history_symbols += [current_symbol]
            symbol_embeddings_sel = self.symbol_list_to_embedding_sel(history_symbols)
            symbol_embeddings_sel_promise: TensorPromise = self.infer_symbol_affine_layer_sel.forward_later(
                symbol_embeddings_sel
            )
            symbol_embeddings_filter = self.symbol_list_to_embedding_filter(
                history_symbols
            )
            symbol_embeddings_filter_promise: TensorPromise = self.infer_symbol_affine_layer_filter.forward_later(
                symbol_embeddings_filter
            )
            new_tensor_dict: Dict[str, TensorPromiseOrTensor] = dict()
            new_tensor_dict.update(prev_tensor_dict)
            new_tensor_dict.update(
                {
                    "symbol_embedding_sel": symbol_embeddings_sel_promise,
                    "symbol_embedding_filter": symbol_embeddings_filter_promise,
                }
            )
            return new_tensor_dict

        def combine_embeddings(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            action_embedding_sel: torch.Tensor = prev_tensor_dict[
                "action_embedding_sel"
            ].result
            symbol_embedding_sel: torch.Tensor = prev_tensor_dict[
                "symbol_embedding_sel"
            ].result
            combined_embedding_sel: torch.Tensor = torch.cat(
                (action_embedding_sel, symbol_embedding_sel), dim=-1
            )
            combined_embedding_sel_promise: TensorPromise = self.infer_tgt_linear_layer_sel.forward_later(
                combined_embedding_sel
            )
            action_embedding_filter: torch.Tensor = prev_tensor_dict[
                "action_embedding_filter"
            ].result
            symbol_embedding_filter: torch.Tensor = prev_tensor_dict[
                "symbol_embedding_filter"
            ].result
            combined_embedding_filter: torch.Tensor = torch.cat(
                (action_embedding_filter, symbol_embedding_filter), dim=-1
            )
            combined_embedding_filter_promise: TensorPromise = self.infer_tgt_linear_layer_filter.forward_later(
                combined_embedding_filter
            )
            prev_tensor_dict.update(
                {
                    "combined_embedding_sel": combined_embedding_sel_promise,
                    "combined_embedding_filter": combined_embedding_filter_promise,
                }
            )
            return prev_tensor_dict

        def pass_infer_transformer(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            combined_embedding_sel: torch.Tensor = prev_tensor_dict[
                "combined_embedding_sel"
            ].result
            src_embedding: torch.Tensor = state.get_encoded_src()

            decoder_out_sel_promise: TensorPromise = self.infer_transformer_sel.forward_later(
                combined_embedding_sel, src_embedding
            )
            combined_embedding_filter: torch.Tensor = prev_tensor_dict[
                "combined_embedding_filter"
            ].result

            decoder_out_filter_promise: TensorPromise = self.infer_transformer_filter.forward_later(
                combined_embedding_filter, src_embedding
            )
            prev_tensor_dict.update(
                {
                    "decoder_out_sel": decoder_out_sel_promise,
                    "decoder_out_filter": decoder_out_filter_promise,
                }
            )
            return prev_tensor_dict

        def pass_infer_out_linear(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            decoder_out_sel: torch.Tensor = prev_tensor_dict["decoder_out_sel"].result
            decoder_out_sel_promise: TensorPromise = self.infer_out_linear_layer_sel.forward_later(
                decoder_out_sel
            )
            decoder_out_filter: torch.Tensor = prev_tensor_dict[
                "decoder_out_filter"
            ].result
            decoder_out_filter_promise: TensorPromise = self.infer_out_linear_layer_filter.forward_later(
                decoder_out_filter
            )
            prev_tensor_dict.update(
                {
                    "decoder_out_sel": decoder_out_sel_promise,
                    "decoder_out_filter": decoder_out_filter_promise,
                }
            )
            return prev_tensor_dict

        def calc_prod(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> List[TensorPromise]:
            def calc_prod_with_idx_and_symbol(idx, symbol, is_sel_mode):
                if symbol == "C":
                    if is_sel_mode:
                        prod = self.infer_column_similarity_sel.forward_later(
                            decoder_out[idx], state.get_encoded_col(), None
                        )
                    else:

                        prod = self.infer_column_similarity_filter.forward_later(
                            decoder_out[idx], state.get_encoded_col(), None
                        )
                elif symbol == "T":
                    if is_sel_mode:
                        prod = self.infer_table_similarity_sel.forward_later(
                            decoder_out[idx],
                            state.get_encoded_tab(),
                            state.invalid_table_indices(idx),
                        )
                    else:
                        prod = self.infer_table_similarity_filter.forward_later(
                            decoder_out[idx],
                            state.get_encoded_tab(),
                            state.invalid_table_indices(idx),
                        )

                else:
                    # Get possible actions from nonterminal stack
                    possible_action_ids = self.grammar_sel.get_possible_aids(symbol)
                    impossible_indices = [
                        idx
                        for idx in range(self.grammar_sel.get_action_len())
                        if idx not in possible_action_ids
                    ]
                    if is_sel_mode:
                        prod = self.infer_action_similarity_sel.forward_later(
                            decoder_out[idx],
                            self.grammar_sel.action_emb.weight,
                            impossible_indices,
                        )
                    else:
                        prod = self.infer_action_similarity_filter.forward_later(
                            decoder_out[idx],
                            self.grammar_filter.action_emb.weight,
                            impossible_indices,
                        )

                return prod

            is_sel_mode = state.is_sel_mode()
            if is_sel_mode:
                decoder_out: torch.Tensor = prev_tensor_dict["decoder_out_sel"].result
            else:
                decoder_out: torch.Tensor = prev_tensor_dict[
                    "decoder_out_filter"
                ].result

            current_symbol: Symbol = state.get_current_symbol()
            assert current_symbol != None
            promise_prod: TensorPromise = calc_prod_with_idx_and_symbol(
                state.step_cnt, current_symbol, is_sel_mode
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
        else:
            if return_details:
                return (
                    TransformerStatePred.get_preds(states),
                    [state.probs for state in states],
                )
            else:
                return TransformerStatePred.get_preds(states)
