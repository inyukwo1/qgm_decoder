from typing import List, Dict, Union
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

        self.decoder_transformer = LazyTransformerDecoder(dim, nhead, layer_num)
        self.refine_transformer = LazyTransformerDecoder(dim, nhead, layer_num)

        self.refine_action_affine_layer = LazyLinear(dim, dim)
        self.refine_symbol_affine_layer = LazyLinear(dim, dim)
        self.refine_tgt_linear_layer = LazyLinear(dim * 3, dim)

        self.action_affine_layer = LazyLinear(dim, dim)
        self.symbol_affine_layer = LazyLinear(dim, dim)
        self.tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.decoder_out_linear_layer = LazyLinear(dim, dim)
        self.refine_out_linear_layer = LazyLinear(dim, dim)

        self.action_similarity = LazyCalculateSimilarity(dim, dim)
        self.column_similarity = LazyCalculateSimilarity(dim, dim)
        self.table_similarity = LazyCalculateSimilarity(dim, dim)

        self.grammar = SemQL(dim)
        self.col_symbol_id = self.grammar.symbol_to_sid["C"]
        self.tab_symbol_id = self.grammar.symbol_to_sid["T"]

    def action_to_embedding(
        self, state: TransformerState, action: Action
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
        golds,
    ):
        b_size = len(encoded_src)
        if golds:
            state_class = TransformerStateGold
            states = [
                TransformerStateGold(
                    encoded_src[b_idx, : src_lens[b_idx]],
                    encoded_col[b_idx, : col_lens[b_idx]],
                    encoded_tab[b_idx, : tab_lens[b_idx]],
                    col_tab_dic[b_idx],
                    golds[b_idx],
                )
                for b_idx in range(b_size)
            ]
        else:
            state_class = TransformerStatePred
            states = [
                TransformerStatePred(
                    encoded_src[b_idx, : src_lens[b_idx]],
                    encoded_col[b_idx, : col_lens[b_idx]],
                    encoded_tab[b_idx, : tab_lens[b_idx]],
                    col_tab_dic[b_idx],
                    self.grammar.start_symbol,
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
            current_action_embedding = self.grammar.get_key_emb()
            history_action_embeddings += [current_action_embedding]
            action_embeddings = torch.stack(history_action_embeddings, dim=0)
            action_embeddings_promise: TensorPromise = self.action_affine_layer.forward_later(
                action_embeddings
            )
            return {"action_embedding": action_embeddings_promise}

        def embed_history_symbols(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            history_symbols: List[Symbol] = state.get_history_symbols()
            current_symbol: Symbol = state.get_current_symbol()
            history_symbols += [current_symbol]
            symbol_embeddings = self.symbol_list_to_embedding(history_symbols)
            symbol_embeddings_promise: TensorPromise = self.symbol_affine_layer.forward_later(
                symbol_embeddings
            )
            new_tensor_dict: Dict[str, TensorPromiseOrTensor] = dict()
            new_tensor_dict.update(prev_tensor_dict)
            new_tensor_dict.update({"symbol_embedding": symbol_embeddings_promise})
            return new_tensor_dict

        def combine_symbol_action_embeddings(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            action_embedding: torch.Tensor = prev_tensor_dict["action_embedding"].result
            symbol_embedding: torch.Tensor = prev_tensor_dict["symbol_embedding"].result
            combined_embedding: torch.Tensor = torch.cat(
                (action_embedding, symbol_embedding), dim=-1
            )
            combined_embedding_promise: TensorPromise = self.tgt_linear_layer.forward_later(
                combined_embedding
            )
            prev_tensor_dict.update({"combined_embedding": combined_embedding_promise})
            return prev_tensor_dict

        def pass_decoder_transformer(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            combined_embedding: torch.Tensor = prev_tensor_dict[
                "combined_embedding"
            ].result
            src_embedding: torch.Tensor = state.encoded_src
            decoder_out_promise: TensorPromise = self.decoder_transformer.forward_later(
                combined_embedding, src_embedding
            )
            prev_tensor_dict.update({"decoder_out": decoder_out_promise})
            return prev_tensor_dict

        def pass_decoder_out_linear(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result
            decoder_out_promise: TensorPromise = self.decoder_out_linear_layer.forward_later(
                decoder_out
            )
            prev_tensor_dict.update({"decoder_out": decoder_out_promise})
            return prev_tensor_dict

        def calc_prod(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> List[TensorPromise]:
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
                    symbol = state.get_current_symbol()
                    if not symbol:
                        symbol = state.get_history_symbols()[idx]
                    possible_action_ids = SemQL.semql.get_possible_aids(symbol)

                    impossible_indices = [idx for idx in range(SemQL.semql.get_action_len()) if idx not in possible_action_ids]

                    prod = self.action_similarity.forward_later(
                        decoder_out[idx], self.grammar.action_emb.weight, impossible_indices
                    )
                return prod

            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result
            promise_prods: List[TensorPromise] = []

            history_symbols: List[Symbol] = state.get_history_symbols()
            current_symbol: Symbol = state.get_current_symbol()
            if current_symbol:
                history_symbols += [current_symbol]
            for idx, symbol in enumerate(history_symbols):
                prod = calc_prod_with_idx_and_symbol(idx, symbol)
                promise_prods.append(prod)
            prev_tensor_dict.update({"infer_prods": promise_prods})
            return prev_tensor_dict

        def apply_prod(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            prev_tensor_list = prev_tensor_dict["infer_prods"]
            if isinstance(state, TransformerStateGold):
                for idx, prod_promise in enumerate(prev_tensor_list):
                    prod = prod_promise.result
                    state.apply_loss(idx, prod)
            else:
                assert isinstance(state, TransformerStatePred)
                prod = prev_tensor_list[-1].result
                state.apply_pred(prod)
            state.step()
            return prev_tensor_dict

        # Functions for refinement stage
        def embed_history_actions_for_refine(state: TransformerState, prev_tensor_dict: Dict) -> Dict[str, TensorPromise]:
            history_actions: List[Action] = state.get_history_actions()
            history_action_embeddings: List[torch.Tensor] = [
                self.action_to_embedding(state, action) for action in history_actions
            ]
            action_embeddings = torch.stack(history_action_embeddings, dim=0)
            action_embeddings_promise: TensorPromise = self.refine_action_affine_layer.forward_later(
                action_embeddings
            )
            prev_tensor_dict.update({"action_embedding": action_embeddings_promise})
            return prev_tensor_dict

        def embed_history_symbols_for_refine(state: TransformerState, prev_tensor_dict: Dict):
            history_symbols: List[Symbol] = state.get_history_symbols()
            symbol_embeddings = self.symbol_list_to_embedding(history_symbols)
            symbol_embeddings_promise: TensorPromise = self.refine_symbol_affine_layer.forward_later(
                symbol_embeddings
            )
            new_tensor_dict: Dict[str, TensorPromiseOrTensor] = dict()
            new_tensor_dict.update(prev_tensor_dict)
            new_tensor_dict.update({"symbol_embedding": symbol_embeddings_promise})
            return new_tensor_dict

        def combine_symbol_action_embeddings_for_refine(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            action_embedding: torch.Tensor = prev_tensor_dict["action_embedding"].result
            symbol_embedding: torch.Tensor = prev_tensor_dict["symbol_embedding"].result
            hidden_vectors: torch.Tensor = prev_tensor_dict["decoder_out"].result
            combined_embedding: torch.Tensor = torch.cat(
                (action_embedding, symbol_embedding, hidden_vectors), dim=-1
            )
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
            src_embedding: torch.Tensor = state.encoded_src
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
            refine_out_promise: TensorPromise = self.decoder_out_linear_layer.forward_later(
                refine_out
            )
            prev_tensor_dict.update({"refine_out": refine_out_promise})
            return prev_tensor_dict

        def calc_prod_for_refine(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> List[TensorPromise]:
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
                    symbol = state.get_history_symbols()[idx]
                    possible_action_ids = SemQL.semql.get_possible_aids(symbol)

                    impossible_indices = [idx for idx in range(SemQL.semql.get_action_len()) if idx not in possible_action_ids]

                    prod = self.action_similarity.forward_later(
                        decoder_out[idx], self.grammar.action_emb.weight, impossible_indices
                    )
                return prod

            decoder_out: torch.Tensor = prev_tensor_dict["refine_out"].result
            promise_prods: List[TensorPromise] = []

            history_symbols: List[Symbol] = state.get_history_symbols()
            current_symbol: Symbol = state.get_current_symbol()
            if current_symbol:
                history_symbols += [current_symbol]
            for idx, symbol in enumerate(history_symbols):
                prod = calc_prod_with_idx_and_symbol(idx, symbol)
                promise_prods.append(prod)
            prev_tensor_dict.update({"refine_prods": promise_prods})
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
            prev_tensor_list = prev_tensor_dict["refine_prods"]

            # Find argmax for all prods
            pred_indices = [torch.argmax(item.result).item() for item in prev_tensor_list]

            if isinstance(state, TransformerStateGold):
                for idx, pred_idx in enumerate(pred_indices):
                    prod = prev_tensor_list[idx].result
                    state.apply_loss(idx, prod)
            else:
                for idx in range(state.refine_step_cnt, len(prev_tensor_list)):
                    ori_action = history_actions[idx]
                    prod = prev_tensor_list[idx].result
                    pred_idx = torch.argmax(prod).item()
                    if ori_action[0] in ["T", "C"]:
                        new_action = (ori_action[0], pred_idx)
                    else:
                        new_action = SemQL.semql.aid_to_action[pred_idx]

                    # Compare
                    if ori_action != new_action:
                        # alter pred history, step cnt
                        state.step_cnt = idx+1
                        state.preds = state.preds[:idx] + [new_action]

                        # roll back nonterminal
                        state.nonterminal_symbol_stack = roll_back_nonterminal_stack(state.preds)
                        break
                state.refine_step_cnt = idx+1

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done)
            .Do(
                LogicUnit.If(state_class.is_not_done)
                .Then(embed_history_actions)
                .Then(embed_history_symbols)
                .Then(combine_symbol_action_embeddings)
                .Then(pass_decoder_transformer)
                .Then(pass_decoder_out_linear)
                .Then(calc_prod)
                .Then(apply_prod)
            )
            .Do(
                LogicUnit.If(state_class.is_to_refine)
                .Then(embed_history_actions_for_refine)
                .Then(embed_history_symbols_for_refine)
                .Then(combine_symbol_action_embeddings_for_refine)
                .Then(pass_refine_transformer)
                .Then(pass_refine_out_linear)
                .Then(calc_prod_for_refine)
                .Then(apply_prod_for_refine)
            )
        ).states

        if golds:
            return TransformerStateGold.combine_loss(states)
        else:
            return TransformerStatePred.get_preds(states)
