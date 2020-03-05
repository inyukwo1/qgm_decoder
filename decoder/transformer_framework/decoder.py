from typing import List, Dict
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

        self.transformer_decoder = LazyTransformerDecoder(dim, nhead, layer_num)

        self.action_affine_layer = LazyLinear(dim, dim)
        self.symbol_affine_layer = LazyLinear(dim, dim)
        self.tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.out_linear_layer = LazyLinear(dim, dim)

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
        tab_col_dic,
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
                    tab_col_dic[b_idx],
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
                    tab_col_dic[b_idx],
                    self.grammar.start_symbol,
                )
                for b_idx in range(b_size)
            ]

        def embed_history_actions(state: TransformerState,) -> Dict[str, TensorPromise]:
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
            return {"combined_embedding": combined_embedding_promise}

        def pass_transformer(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            combined_embedding: torch.Tensor = prev_tensor_dict[
                "combined_embedding"
            ].result
            src_embedding: torch.Tensor = state.encoded_src
            decoder_out_promise: TensorPromise = self.transformer_decoder.forward_later(
                combined_embedding, src_embedding
            )
            return {"decoder_out": decoder_out_promise}

        def pass_out_linear(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result
            decoder_out_promise: TensorPromise = self.out_linear_layer.forward_later(
                decoder_out
            )
            return {"decoder_out": decoder_out_promise}

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
                    prod = self.action_similarity.forward_later(
                        decoder_out[idx], self.grammar.action_emb.weight, None
                    )
                return prod

            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result
            current_symbol: Symbol = state.get_current_symbol()
            promise_prods: List[TensorPromise] = []
            if isinstance(state, TransformerStateGold):
                history_symbols: List[Symbol] = state.get_history_symbols()
                history_symbols += [current_symbol]
                for idx, symbol in enumerate(history_symbols):
                    prod = calc_prod_with_idx_and_symbol(idx, symbol)
                    promise_prods.append(prod)
            else:
                assert isinstance(state, TransformerStatePred)
                prod = calc_prod_with_idx_and_symbol(-1, current_symbol)
                promise_prods.append(prod)
            return promise_prods

        def apply_prod(state: TransformerState, prev_tensor_list: List[TensorPromise]):
            if isinstance(state, TransformerStateGold):
                for idx, prod_promise in enumerate(prev_tensor_list):
                    prod = prod_promise.result
                    state.apply_loss(idx, prod)
            else:
                assert isinstance(state, TransformerStatePred)
                prod = prev_tensor_list[0].result
                state.apply_pred(prod)
            state.step()

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done).Do(
                LogicUnit.If(state_class.is_not_done)
                .Then(embed_history_actions)
                .Then(embed_history_symbols)
                .Then(combine_symbol_action_embeddings)
                .Then(pass_transformer)
                .Then(pass_out_linear)(calc_prod)
                .Then(apply_prod)
            )
        ).states

        if golds:
            return TransformerStateGold.combine_loss(states)
        else:
            return TransformerStatePred.get_preds(states)
