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
    LazyCalculateSimilarity,
)


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
        if self.refine_all:
            assert self.use_ct_loss == False, "Should be false"

        # For inference
        self.infer_transformer = LazyTransformerDecoder(dim, nhead, layer_num)
        self.infer_action_affine_layer = LazyLinear(dim, dim)
        self.infer_symbol_affine_layer = LazyLinear(dim, dim)
        self.infer_tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.infer_out_linear_layer = LazyLinear(dim, dim)
        self.infer_action_similarity = LazyCalculateSimilarity(dim, dim)
        self.infer_column_similarity = LazyCalculateSimilarity(dim, dim)
        self.infer_table_similarity = LazyCalculateSimilarity(dim, dim)
        self.grammar = SemQL(dim)

        # For refinement
        self.refine_transformer = LazyTransformerDecoder(dim, nhead, layer_num)
        self.refine_action_affine_layer = LazyLinear(dim, dim)
        self.refine_symbol_affine_layer = LazyLinear(dim, dim)
        self.refine_tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.refine_out_linear_layer = LazyLinear(dim, dim)
        self.refine_action_similarity = LazyCalculateSimilarity(dim, dim)
        self.refine_column_similarity = LazyCalculateSimilarity(dim, dim)
        self.refine_table_similarity = LazyCalculateSimilarity(dim, dim)
        self.grammar2 = SemQL(dim)

        # Arbitrator
        self.arbitray_transformer = LazyTransformerDecoder(dim, nhead, layer_num)
        self.arbitray_action_affine_layer = LazyLinear(dim, dim)
        self.arbitray_symbol_affine_layer = LazyLinear(dim, dim)
        self.arbitray_tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.arbitray_out_linear_layer = LazyLinear(dim, dim)
        self.arbitray_action_similarity = LazyCalculateSimilarity(dim, dim)
        self.arbitray_column_similarity = LazyCalculateSimilarity(dim, dim)
        self.arbitray_table_similarity = LazyCalculateSimilarity(dim, dim)
        self.grammar3 = SemQL(dim)

        self.col_symbol_id = self.grammar.symbol_to_sid["C"]
        self.tab_symbol_id = self.grammar.symbol_to_sid["T"]

    def action_to_embedding(
        self, state: TransformerState, action: Action, grammar_idx=0,
    ) -> torch.Tensor:
        if grammar_idx == 0:
            grammar = self.grammar
        elif grammar_idx == 1:
            grammar = self.grammar2
        else:
            grammar = self.grammar3
        if action[0] == "C":
            col_idx = action[1]
            return state.get_encoded_col(grammar_idx)[col_idx]
        elif action[0] == "T":
            tab_idx = action[1]
            return state.get_encoded_tab(grammar_idx)[tab_idx]
        else:
            action_idx = torch.tensor(grammar.action_to_aid[action]).long().cuda()
            return grammar.action_emb(action_idx)

    def symbol_list_to_embedding(
        self, symbol_list: List[Symbol], grammar_idx=0
    ) -> torch.Tensor:
        if grammar_idx == 0:
            grammar = self.grammar
        elif grammar_idx == 1:
            grammar = self.grammar2
        else:
            grammar = self.grammar3

        symbol_ids_list: List[SymbolId] = [
            grammar.symbol_to_sid[symbol] for symbol in symbol_list
        ]
        symbol_ids_tensor = torch.tensor(symbol_ids_list).long().cuda()
        symbol_embeddings = grammar.symbol_emb(symbol_ids_tensor)
        return symbol_embeddings

    def onedim_zero_tensor(self):
        return torch.zeros(self.dim).cuda()

    def forward(
        self,
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
    ):
        b_size = len(encoded_src[0])

        # Create states
        if states:
            state_class = type(states[0])
            for state in states:
                state.target_step = target_step
        elif golds:
            state_class = TransformerStateGold
            states = [
                TransformerStateGold(
                    [item[b_idx, : src_lens[b_idx]] for item in encoded_src],
                    [item[b_idx, : col_lens[b_idx]] for item in encoded_col],
                    [item[b_idx, : tab_lens[b_idx]] for item in encoded_tab],
                    col_tab_dic[b_idx],
                    golds[b_idx],
                )
                for b_idx in range(b_size)
            ]
        else:
            state_class = TransformerStatePred
            states = [
                TransformerStatePred(
                    [item[b_idx, : src_lens[b_idx]] for item in encoded_src],
                    [item[b_idx, : col_lens[b_idx]] for item in encoded_col],
                    [item[b_idx, : tab_lens[b_idx]] for item in encoded_tab],
                    col_tab_dic[b_idx],
                    self.grammar.start_symbol,
                    target_step,
                )
                for b_idx in range(b_size)
            ]

        def embed_history_actions(
            state: TransformerState, _
        ) -> Dict[str, TensorPromise]:
            history_actions: List[Action] = state.get_history_actions()
            history_action_embeddings: List[torch.Tensor] = [
                self.action_to_embedding(state, action, grammar_idx=0)
                for action in history_actions
            ]
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
            current_symbol: Symbol = state.get_current_symbol()
            history_symbols += [current_symbol]
            symbol_embeddings = self.symbol_list_to_embedding(
                history_symbols, grammar_idx=0
            )
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
            src_embedding: torch.Tensor = state.get_encoded_src(0)
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
                        decoder_out[idx], state.get_encoded_col(0), None
                    )
                elif symbol == "T":
                    prod = self.infer_table_similarity.forward_later(
                        decoder_out[idx],
                        state.get_encoded_tab(0),
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

        def save_initial_pred(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            if isinstance(state, TransformerStatePred):
                actions = copy.deepcopy(state.get_history_actions())
                assert len(actions) == len(state.preds), "Diff: {} {}".format(
                    len(actions), len(state.preds)
                )
                state.save_init_preds(actions)

        # Functions for refinement stage
        def embed_history_actions_for_refine(
            state: TransformerState, prev_tensor_dict: Dict
        ) -> Dict[str, TensorPromise]:
            history_actions: List[Action] = state.get_history_actions()
            history_action_embeddings: List[torch.Tensor] = [
                self.onedim_zero_tensor()
                if idx == state.refine_step_cnt
                else self.action_to_embedding(state, action, grammar_idx=1)
                for idx, action in enumerate(history_actions)
            ]
            action_embeddings = torch.stack(history_action_embeddings, dim=0)
            action_embeddings_promise: TensorPromise = self.refine_action_affine_layer.forward_later(
                action_embeddings
            )
            new_tensor_dict = {"action_embedding": action_embeddings_promise}
            return new_tensor_dict

        def embed_history_symbols_for_refine(
            state: TransformerState, prev_tensor_dict: Dict
        ):
            history_symbols: List[Symbol] = state.get_history_symbols()
            symbol_embeddings = self.symbol_list_to_embedding(
                history_symbols, grammar_idx=1
            )
            symbol_embeddings_promise: TensorPromise = self.refine_symbol_affine_layer.forward_later(
                symbol_embeddings
            )
            prev_tensor_dict.update({"symbol_embedding": symbol_embeddings_promise})
            return prev_tensor_dict

        def combine_embeddings_for_refine(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            action_embedding: torch.Tensor = prev_tensor_dict["action_embedding"].result
            symbol_embedding: torch.Tensor = prev_tensor_dict["symbol_embedding"].result
            combined_embedding = torch.cat((action_embedding, symbol_embedding), dim=-1)
            combined_embedding_promise: TensorPromise = self.refine_tgt_linear_layer.forward_later(
                combined_embedding
            )
            prev_tensor_dict.update({"combined_embedding": combined_embedding_promise})
            return prev_tensor_dict

        def pass_refine_transformer(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            combined_embedding: torch.Tensor = prev_tensor_dict[
                "combined_embedding"
            ].result
            src_embedding: torch.Tensor = state.get_encoded_src(1)
            refine_out_promise: TensorPromise = self.refine_transformer.forward_later(
                combined_embedding, src_embedding
            )
            prev_tensor_dict.update({"refine_out": refine_out_promise})
            return prev_tensor_dict

        def pass_refine_out_linear(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            refine_out: torch.Tensor = prev_tensor_dict["refine_out"].result
            refine_out_promise: TensorPromise = self.refine_out_linear_layer.forward_later(
                refine_out
            )
            prev_tensor_dict.update({"refine_out": refine_out_promise})
            return prev_tensor_dict

        def calc_prod_for_refine(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> List[TensorPromise]:
            def calc_prod_with_idx_and_symbol(idx, symbol):
                if symbol == "C":
                    prod = self.refine_column_similarity.forward_later(
                        decoder_out[idx], state.get_encoded_col(1), None
                    )
                elif symbol == "T":
                    prod = self.refine_table_similarity.forward_later(
                        decoder_out[idx],
                        state.get_encoded_tab(1),
                        state.impossible_table_indices(idx),
                    )
                else:
                    possible_action_ids = SemQL.semql.get_possible_aids(symbol)
                    impossible_indices = [
                        idx
                        for idx in range(SemQL.semql.get_action_len())
                        if idx not in possible_action_ids
                    ]

                    prod = self.refine_action_similarity.forward_later(
                        decoder_out[idx],
                        self.grammar.action_emb.weight,
                        impossible_indices,
                    )
                return prod

            decoder_out: torch.Tensor = prev_tensor_dict["refine_out"].result

            assert state.get_current_symbol() == None

            history_symbols: List[Symbol] = state.get_history_symbols()
            target_symbol = history_symbols[state.refine_step_cnt]
            promise_prod: TensorPromise = calc_prod_with_idx_and_symbol(
                state.refine_step_cnt, target_symbol
            )
            prev_tensor_dict.update({"refine_prods": promise_prod})
            return prev_tensor_dict

        def embed_history_actions_for_arbitrate(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            history_actions: List[Action] = state.get_history_actions()
            history_action_embeddings: List[torch.Tensor] = [
                self.onedim_zero_tensor()
                if idx == state.refine_step_cnt
                else self.action_to_embedding(state, action, grammar_idx=2)
                for idx, action in enumerate(history_actions)
            ]
            action_embeddings = torch.stack(history_action_embeddings, dim=0)
            action_embeddings_promise: TensorPromise = self.arbitray_action_affine_layer.forward_later(
                action_embeddings
            )
            prev_tensor_dict.update({"action_embedding": action_embeddings_promise})
            return prev_tensor_dict

        def embed_history_symbols_for_arbitrate(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            history_symbols: List[Symbol] = state.get_history_symbols()
            symbol_embeddings = self.symbol_list_to_embedding(
                history_symbols, grammar_idx=2
            )
            symbol_embeddings_promise: TensorPromise = self.arbitray_symbol_affine_layer.forward_later(
                symbol_embeddings
            )
            prev_tensor_dict.update({"symbol_embedding": symbol_embeddings_promise})
            return prev_tensor_dict

        def combine_embeddings_for_arbitrate(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            action_embedding: torch.Tensor = prev_tensor_dict["action_embedding"].result
            symbol_embedding: torch.Tensor = prev_tensor_dict["symbol_embedding"].result
            combined_embedding = torch.cat((action_embedding, symbol_embedding), dim=-1)
            combined_embedding_promise: TensorPromise = self.arbitray_tgt_linear_layer.forward_later(
                combined_embedding
            )
            prev_tensor_dict.update({"combined_embedding": combined_embedding_promise})
            return prev_tensor_dict

        def pass_arbitrate_transformer(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            combined_embedding: torch.Tensor = prev_tensor_dict[
                "combined_embedding"
            ].result
            src_embedding: torch.Tensor = state.get_encoded_src(2)
            arbitrate_out_promise: TensorPromise = self.arbitray_transformer.forward_later(
                combined_embedding, src_embedding
            )
            prev_tensor_dict.update({"arbitrate_out": arbitrate_out_promise})
            return prev_tensor_dict

        def pass_arbitrate_out_linear(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            arbitrate_out: torch.Tensor = prev_tensor_dict["arbitrate_out"].result
            arbitrate_out_promise: TensorPromise = self.arbitray_out_linear_layer.forward_later(
                arbitrate_out
            )
            prev_tensor_dict.update({"arbitrate_out": arbitrate_out_promise})
            return prev_tensor_dict

        def calc_prod_for_arbitrate(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            def calc_prod_with_idx_and_symbol(idx, symbol):
                if symbol == "C":
                    prod = self.arbitray_column_similarity.forward_later(
                        decoder_out[idx], state.get_encoded_col(2), None
                    )
                elif symbol == "T":
                    prod = self.arbitray_table_similarity.forward_later(
                        decoder_out[idx],
                        state.get_encoded_tab(2),
                        state.impossible_table_indices(idx),
                    )
                else:
                    possible_action_ids = SemQL.semql.get_possible_aids(symbol)
                    impossible_indices = [
                        idx
                        for idx in range(SemQL.semql.get_action_len())
                        if idx not in possible_action_ids
                    ]

                    prod = self.arbitray_action_similarity.forward_later(
                        decoder_out[idx],
                        self.grammar.action_emb.weight,
                        impossible_indices,
                    )
                return prod

            decoder_out: torch.Tensor = prev_tensor_dict["arbitrate_out"].result

            assert state.get_current_symbol() == None

            history_symbols: List[Symbol] = state.get_history_symbols()
            target_symbol = history_symbols[state.refine_step_cnt]
            promise_prod: TensorPromise = calc_prod_with_idx_and_symbol(
                state.refine_step_cnt, target_symbol
            )
            prev_tensor_dict.update({"arbitrate_prods": promise_prod})
            return prev_tensor_dict

        def apply_prod_for_refine(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            def roll_back_nonterminal_stack(actions):
                stack = []
                for idx, action in enumerate(actions):
                    # Get next action symbols
                    symbols = SemQL.semql.parse_nonterminal_symbol(action)

                    # Pop first item
                    if idx + 1 != len(actions):
                        symbols = symbols[1:]

                    # Append
                    stack = symbols + stack
                return stack

            history_actions: List[Action] = state.get_history_actions()
            symbol = history_actions[state.refine_step_cnt]
            refine_prod = prev_tensor_dict["refine_prods"].result
            arbitrate_prod = prev_tensor_dict["arbitrate_prods"].result
            cur_refine_step = state.refine_step_cnt
            if state.is_gold():
                # Loss for C T (?)
                if not self.use_ct_loss or symbol in ["C", "T"]:
                    state.apply_loss(cur_refine_step, refine_prod)
                    state.apply_loss(cur_refine_step, arbitrate_prod)
            else:
                assert isinstance(state, TransformerStatePred)
                ori_action = history_actions[cur_refine_step]
                ori_pred_idx = (
                    ori_action[1]
                    if ori_action[0] in ["T", "C"]
                    else SemQL.semql.action_to_aid[ori_action]
                )

                # Refine pred
                refine_pred_idx = torch.argmax(refine_prod).item()
                refine_action = (
                    (ori_action[0], refine_pred_idx)
                    if ori_action[0] in ["T", "C"]
                    else SemQL.semql.aid_to_action[refine_pred_idx]
                )

                # get pred_action
                if ori_action != refine_action:
                    if self.refine_all or ori_action[0] in ["C", "T"]:
                        # Compare (change C T only)
                        if ori_pred_idx != refine_pred_idx:
                            if self.use_arbitrator:
                                if (
                                    arbitrate_prod[refine_pred_idx]
                                    > arbitrate_prod[ori_pred_idx]
                                ):
                                    final_pred_idx = refine_pred_idx
                                else:
                                    final_pred_idx = ori_pred_idx
                            else:
                                final_pred_idx = refine_pred_idx

                            final_action: Action = (
                                ori_action[0],
                                final_pred_idx,
                            ) if ori_action[0] in [
                                "T",
                                "C",
                            ] else SemQL.semql.aid_to_action[
                                final_pred_idx
                            ]

                            # Save refiner and arbitrator's inference
                            state.refine_pred(refine_action, cur_refine_step)
                            state.arbitrate_pred(final_action, cur_refine_step)

                            if ori_action != final_action:
                                # alter pred history, step cnt
                                state.step_cnt = cur_refine_step + 1
                                state.infer_pred(final_action, cur_refine_step)

                                # roll back nonterminal
                                state.nonterminal_symbol_stack = roll_back_nonterminal_stack(
                                    state.arbitrated_preds[: cur_refine_step + 1]
                                )
                else:
                    state.refine_pred(refine_action, cur_refine_step)
                    state.arbitrate_pred(refine_action, cur_refine_step)
            state.refine_step_cnt = cur_refine_step + 1

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done)
            .Do(
                LogicUnit.If(state_class.is_to_infer)
                .Then(embed_history_actions)
                .Then(embed_history_symbols)
                .Then(combine_embeddings)
                .Then(pass_infer_transformer)
                .Then(pass_infer_out_linear)
                .Then(calc_prod)
                .Then(apply_prod)
            )
            .Do(LogicUnit.If(state_class.is_initial_pred).Then(save_initial_pred))
            .Do(
                LogicUnit.If(state_class.is_to_refine)
                .Then(embed_history_actions_for_refine)
                .Then(embed_history_symbols_for_refine)
                .Then(combine_embeddings_for_refine)
                .Then(pass_refine_transformer)
                .Then(pass_refine_out_linear)
                .Then(calc_prod_for_refine)
                .Then(embed_history_actions_for_arbitrate)
                .Then(embed_history_symbols_for_arbitrate)
                .Then(combine_embeddings_for_arbitrate)
                .Then(pass_arbitrate_transformer)
                .Then(pass_arbitrate_out_linear)
                .Then(calc_prod_for_arbitrate)
                .Then(apply_prod_for_refine)
            )
        ).states

        if golds:
            return TransformerStateGold.combine_loss(states)
        else:
            return TransformerStatePred.get_preds(states), states
