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
from framework.lazy_modules import LazyLinear, LazyTransformerDecoder


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

        self.action_decode_affine_layer = nn.Linear(dim, dim)
        self.col_decode_affine_layer = nn.Linear(dim, dim)
        self.tab_decode_affine_layer = nn.Linear(dim, dim)

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

        def embed_history_actions(
            state: TransformerState,
        ) -> Dict[str, TensorPromiseOrTensor]:
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
            state: TransformerState, prev_tensor_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, TensorPromiseOrTensor]:
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
            state: TransformerState, prev_tensor_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, TensorPromiseOrTensor]:
            action_embedding: torch.Tensor = prev_tensor_dict["action_embedding"]
            symbol_embedding: torch.Tensor = prev_tensor_dict["symbol_embedding"]
            combined_embedding: torch.Tensor = torch.cat(
                (action_embedding, symbol_embedding), dim=-1
            )
            combined_embedding_promise: TensorPromise = self.tgt_linear_layer.forward_later(
                combined_embedding
            )
            return {"combined_embedding": combined_embedding_promise}

        def pass_transformer(
            state: TransformerState, prev_tensor_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, TensorPromiseOrTensor]:
            combined_embedding: torch.Tensor = prev_tensor_dict["combined_embedding"]
            src_embedding: torch.Tensor = state.encoded_src
            decoder_out_promise: TensorPromise = self.transformer_decoder.forward_later(
                combined_embedding, src_embedding
            )
            return {"decoder_out": decoder_out_promise}

        def pass_out_linear(
            state: TransformerState, prev_tensor_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, TensorPromiseOrTensor]:
            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"]
            decoder_out_promise: TensorPromise = self.out_linear_layer.forward_later(
                decoder_out
            )
            return {"decoder_out": decoder_out_promise}

        def calc_prod(
            state: TransformerState, prev_tensor_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, TensorPromiseOrTensor]:
            def calculate_similarity(source, query, affine_layer, mask=None):
                source = affine_layer(source)
                weight = torch.mm(source.unsqueeze(0), query.transpose(0, 1)).squeeze()
                if mask is not None:
                    weight.data.masked_fill_(mask.bool(), -float("inf"))
                prob = torch.log_softmax(weight, dim=-1)
                return prob

            def calculate_similarity_action(source):
                return calculate_similarity(
                    source,
                    self.grammar.action_emb.weight,
                    self.action_decode_affine_layer,
                )

            def calculate_similarity_column(source):
                return calculate_similarity(
                    source, state.encoded_col, self.col_decode_affine_layer
                )

            def calculate_similarity_table(source, possible_table_indices: List[int]):
                mask = torch.ones(len(state.encoded_tab)).cuda()
                for index in possible_table_indices:
                    mask[index] = 0
                return calculate_similarity(
                    source, state.encoded_tab, self.tab_decode_affine_layer, mask
                )

            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"]
            current_symbol: Symbol = state.get_current_symbol()
            if isinstance(state, TransformerStateGold):
                history_symbols: List[Symbol] = state.get_history_symbols()
                history_symbols += [current_symbol]
                for idx, symbol in enumerate(history_symbols):
                    if symbol == "C":
                        prod = calculate_similarity_column(decoder_out[idx])
                    elif symbol == "T":
                        prod = calculate_similarity_table(
                            decoder_out[idx], state.possible_table_indices(idx)
                        )
                    else:
                        prod = calculate_similarity_action(decoder_out[idx])
                    state.apply_loss(idx, prod)
            else:
                assert isinstance(state, TransformerStatePred)
                if current_symbol == "C":
                    prod = calculate_similarity_column(decoder_out[-1])
                elif current_symbol == "T":
                    prod = calculate_similarity_table(
                        decoder_out[-1], state.possible_table_indices(-1)
                    )
                else:
                    prod = calculate_similarity_action(decoder_out[-1])

                state.apply_pred(prod)
            state.step()
            return dict()

        states = SequentialMonad(states)(
            WhileLogic(state_class.is_not_done)(
                LogicUnit(state_class.is_not_done)(embed_history_actions)(
                    embed_history_symbols
                )(combine_symbol_action_embeddings)(pass_transformer)(pass_out_linear)(
                    calc_prod
                )
            )
        ).states

        if golds:
            return TransformerStateGold.combine_loss(states)
        else:
            return TransformerStatePred.get_preds(states)
