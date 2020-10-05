from typing import List, Dict, Union, Tuple
import torch
import torch.nn as nn
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
from qgm.qgm_action import QGM_ACTION, Action, Symbol


class TransformerDecoderFramework(nn.Module):
    def __init__(self, cfg):
        super(TransformerDecoderFramework, self).__init__()
        nhead = cfg.nhead
        layer_num = cfg.layer_num
        hidden_size = cfg.hidden_size

        # Decode Layers
        dim = 1024 if cfg.encoder_name == "bert" else hidden_size
        nhead = 8 if cfg.encoder_name == "bert" else nhead
        self.cfg = cfg
        self.dim = dim
        self.nhead = nhead
        self.layer_num = layer_num
        self.use_relation = cfg.use_relation
        self.padding_action_tensor = nn.Parameter(torch.rand(self.dim))
        self.padding_symbol_tensor = nn.Parameter(torch.rand(self.dim))
        self.padding_num = 2

        # For inference
        self.infer_transformer = LazyTransformerDecoder(dim, nhead, layer_num)
        self.infer_action_affine_layer = LazyLinear(dim, dim)
        self.infer_symbol_affine_layer = LazyLinear(dim, dim)
        self.infer_tgt_linear_layer = LazyLinear(dim * 2, dim)
        self.infer_out_linear_layer = LazyLinear(dim, dim)
        self.infer_out_linear_layer_pointer_dict = LazyLinear(dim * 2, dim)
        self.infer_action_similarity = LazyCalculateSimilarity(dim, dim)
        self.infer_column_similarity = LazyCalculateSimilarity(dim, dim)
        self.infer_table_similarity = LazyCalculateSimilarity(dim, dim)
        self.padding_action_tensor = nn.Parameter(torch.rand(self.dim))

        self.col_end_tensor = nn.Parameter(torch.rand(self.dim))

        self.symbol_emb = nn.Embedding(QGM_ACTION.total_symbol_len(), dim)
        self.action_emb = nn.Embedding(QGM_ACTION.total_action_len(), dim)

        self.wrong_trace = []

    def action_to_embedding(
        self, state: TransformerState, symbol: Symbol, action: Action,
    ) -> torch.Tensor:
        if symbol == "C":
            col_idx = action
            return state.get_encoded_col()[col_idx]
        elif symbol == "T":
            tab_idx = action
            return state.get_encoded_tab()[tab_idx]
        else:
            action_idx = (
                torch.tensor(QGM_ACTION.symbol_action_to_action_id(symbol, action))
                .long()
                .cuda()
            )
            return self.action_emb(action_idx)

    def symbol_to_embedding(self, symbol: Symbol) -> torch.Tensor:
        symbol_id = torch.tensor(QGM_ACTION.symbol_to_symbol_id(symbol)).long().cuda()
        return self.symbol_emb(symbol_id)

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
        dbs,
        nlqs,
        golds=None,
        is_train=False,
    ):
        # Create states
        if is_train:
            state_class = TransformerStateGold
            states = [
                TransformerStateGold(
                    self.cfg,
                    encoded_src[b_idx][: src_lens[b_idx]],
                    encoded_col[b_idx][: col_lens[b_idx]],
                    encoded_tab[b_idx][: tab_lens[b_idx]],
                    col_tab_dic[b_idx],
                    golds[b_idx][:],
                    nlqs[b_idx],
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
                    golds[b_idx][:],
                    dbs[b_idx],
                    nlqs[b_idx],
                    golds is not None,
                )
                for b_idx in range(b_size)
            ]
        for state in states:
            state.ready()

        def embed_history_symbol_actions(
            state: TransformerState, _
        ) -> Dict[str, TensorPromise]:
            history_symbol_actions: List[
                Tuple[Symbol, Action]
            ] = state.get_history_symbol_actions()
            history_action_embeddings: List[torch.Tensor] = []
            history_symbol_embeddings: List[torch.Tensor] = []
            for symbol, action in history_symbol_actions:
                history_action_embeddings += [
                    self.action_to_embedding(state, symbol, action)
                ]
                history_action_embeddings += [
                    self.padding_action_tensor
                ] * self.padding_num
                history_symbol_embeddings += [self.symbol_to_embedding(symbol)]
                history_symbol_embeddings += [
                    self.padding_symbol_tensor
                ] * self.padding_num
            history_action_embeddings += [self.padding_action_tensor] * self.padding_num
            history_symbol_embeddings += [self.padding_symbol_tensor] * self.padding_num
            current_action_embedding = self.onedim_zero_tensor()
            history_action_embeddings += [current_action_embedding]
            current_symbol = state.get_current_symbol()
            history_symbol_embeddings += [self.symbol_to_embedding(current_symbol)]
            assert len(history_symbol_embeddings) == len(history_action_embeddings)
            action_embeddings = torch.stack(history_action_embeddings, dim=0)
            symbol_embeddings = torch.stack(history_symbol_embeddings, dim=0)
            action_embeddings_promise: TensorPromise = self.infer_action_affine_layer.forward_later(
                action_embeddings
            )
            symbol_embeddings_promise: TensorPromise = self.infer_symbol_affine_layer.forward_later(
                symbol_embeddings
            )
            return {
                "action_embedding": action_embeddings_promise,
                "symbol_embedding": symbol_embeddings_promise,
            }

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

            decoder_out_promise: TensorPromise = self.infer_transformer.forward_later(
                combined_embedding, src_embedding
            )
            prev_tensor_dict.update({"decoder_out": decoder_out_promise})
            return prev_tensor_dict

        def pass_infer_out_linear(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result

            current_symbol: Symbol = state.get_current_symbol()
            if current_symbol not in {"BASE_COL_EXIST", "C", "T"}:
                predicate_col_pointer = state.get_base_col_pointer()
                if predicate_col_pointer == len(state.history_indices):
                    decoder_out_promise: TensorPromise = self.infer_out_linear_layer_pointer_dict.forward_later(
                        torch.cat((self.col_end_tensor, decoder_out[-1],), dim=-1,)
                    )

                else:
                    history_idx = state.history_indices[predicate_col_pointer]
                    physical_idx = history_idx * (self.padding_num + 1)
                    decoder_out_promise: TensorPromise = self.infer_out_linear_layer_pointer_dict.forward_later(
                        torch.cat(
                            (decoder_out[physical_idx], decoder_out[-1],), dim=-1,
                        )
                    )
            else:
                decoder_out_promise: TensorPromise = self.infer_out_linear_layer.forward_later(
                    decoder_out[-1]
                )
            prev_tensor_dict.update({"decoder_out": decoder_out_promise})
            return prev_tensor_dict

        def calc_prod(
            state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
        ) -> Dict[str, TensorPromise]:
            def calc_prod_with_idx_and_symbol(org_idx, symbol):
                if symbol == "C":
                    prod = self.infer_column_similarity.forward_later(
                        decoder_out, state.get_encoded_col(), None
                    )
                elif symbol == "T":
                    prod = self.infer_table_similarity.forward_later(
                        decoder_out,
                        state.get_encoded_tab(),
                        state.invalid_table_indices(org_idx),
                    )
                else:
                    # Get possible actions from nonterminal stack
                    possible_action_ids = QGM_ACTION.possible_action_ids(symbol)
                    impossible_indices = [
                        idx
                        for idx in range(QGM_ACTION.total_action_len())
                        if idx not in possible_action_ids
                    ]
                    prod = self.infer_action_similarity.forward_later(
                        decoder_out, self.action_emb.weight, impossible_indices,
                    )
                return prod

            decoder_out: torch.Tensor = prev_tensor_dict["decoder_out"].result
            current_symbol: Symbol = state.get_current_symbol()
            assert current_symbol is not None
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
            if isinstance(state, TransformerStateGold):
                state.apply_loss(prod)
            else:
                assert isinstance(state, TransformerStatePred)
                state.apply_pred(prod)
            state.ready()
            return prev_tensor_dict

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done).Do(
                LogicUnit.If(state_class.is_to_infer)
                .Then(embed_history_symbol_actions)
                .Then(combine_embeddings)
                .Then(pass_infer_transformer)
                .Then(pass_infer_out_linear)
                .Then(calc_prod)
                .Then(apply_prod)
            )
        ).states

        # for state in states:
        #     if state.wrong:
        #         self.wrong_trace.append(
        #             {
        #                 "nlq": " ".join([" ".join(word) for word in state.nlq]),
        #                 "answer/pred": [
        #                     (ac1[0], str(ac1[1]) + " / " + str(ac2[1]))
        #                     for ac1, ac2 in zip(state.history, state.pred_history)
        #                 ],
        #                 "db": [
        #                     (i, state.db["table_names"][t_id], c_name)
        #                     for i, (t_id, c_name) in enumerate(state.db["column_names"])
        #                 ],
        #             }
        #         )

        if is_train:
            return TransformerStateGold.combine_loss(states)
        else:
            if golds is not None:
                return TransformerStatePred.get_accs(states)
            else:
                return TransformerStatePred.get_preds(states)
