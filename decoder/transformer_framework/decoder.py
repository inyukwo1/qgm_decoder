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


class TransformerDecoderModule(nn.Module):
    def __init__(self, dim, nhead, layer_num, use_relation):
        super(TransformerDecoderModule, self).__init__()
        relation_keys = [0, 1, 2]
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

        if self.refine_all:
            assert self.use_ct_loss is False, "Should be false"

        self.decoders = nn.ModuleDict(
            {
                "infer": TransformerDecoderModule(dim, nhead, layer_num, False),
                "refine": TransformerDecoderModule(
                    dim, nhead, layer_num, self.use_relation
                ),
                "arbitrate": TransformerDecoderModule(
                    dim, nhead, layer_num, self.use_relation
                ),
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
        for idx in range(cur_idx):
            relation[idx][:cur_idx] = 2
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
        target_step=100,
        states=None,
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
                    {
                        "infer": encoded_src[0][b_idx][: src_lens[b_idx]]
                        if encoded_src[0] != [None]
                        else None,
                        "refine": encoded_src[1][b_idx][: src_lens[b_idx]]
                        if encoded_src[1] != [None]
                        else None,
                        "arbitrate": encoded_src[2][b_idx][: src_lens[b_idx]]
                        if encoded_src[2] != [None]
                        else None,
                    },
                    {
                        "infer": encoded_col[0][b_idx][: col_lens[b_idx]]
                        if encoded_col[0] != [None]
                        else None,
                        "refine": encoded_col[1][b_idx][: col_lens[b_idx]]
                        if encoded_col[1] != [None]
                        else None,
                        "arbitrate": encoded_col[2][b_idx][: col_lens[b_idx]]
                        if encoded_col[2] != [None]
                        else None,
                    },
                    {
                        "infer": encoded_tab[0][b_idx][: tab_lens[b_idx]]
                        if encoded_tab[0] != [None]
                        else None,
                        "refine": encoded_tab[1][b_idx][: tab_lens[b_idx]]
                        if encoded_tab[1] != [None]
                        else None,
                        "arbitrate": encoded_tab[2][b_idx][: tab_lens[b_idx]]
                        if encoded_tab[2] != [None]
                        else None,
                    },
                    col_tab_dic[b_idx],
                    golds[b_idx],
                )
                for b_idx in range(b_size)
            ]
        else:
            state_class = TransformerStatePred
            states = [
                TransformerStatePred(
                    {
                        "infer": encoded_src[0][b_idx][: src_lens[b_idx]],
                        "refine": encoded_src[1][b_idx][: src_lens[b_idx]],
                        "arbitrate": encoded_src[2][b_idx][: src_lens[b_idx]],
                    },
                    {
                        "infer": encoded_col[0][b_idx][: col_lens[b_idx]],
                        "refine": encoded_col[1][b_idx][: col_lens[b_idx]],
                        "arbitrate": encoded_col[2][b_idx][: col_lens[b_idx]],
                    },
                    {
                        "infer": encoded_tab[0][b_idx][: tab_lens[b_idx]],
                        "refine": encoded_tab[1][b_idx][: tab_lens[b_idx]],
                        "arbitrate": encoded_tab[2][b_idx][: tab_lens[b_idx]],
                    },
                    col_tab_dic[b_idx],
                    SemQL.semql.start_symbol,
                    target_step,
                )
                for b_idx in range(b_size)
            ]

        def embed_history_actions_per_mode(mode):
            def embed_history_actions(
                state: TransformerState, prev_tensor_dict: Dict[str, TensorPromise]
            ) -> Dict[str, TensorPromise]:
                history_actions: List[Action] = state.get_history_actions(
                    mode, self.look_left_only
                )
                history_action_embeddings: List[torch.Tensor] = [
                    self.action_to_embedding(state, action, mode)
                    for action in history_actions
                ]
                if mode == "infer":
                    history_action_embeddings += [self.onedim_zero_tensor()]
                else:
                    history_action_embeddings[
                        state.refine_step_cnt
                    ] = self.onedim_zero_tensor()
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
                history_symbols: List[Symbol] = state.get_history_symbols(
                    mode, self.look_left_only
                )
                if mode == "infer":
                    current_symbol: Symbol = state.get_current_symbol()
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
                        state.get_history_actions(mode, self.look_left_only),
                        cur_idx=state.refine_step_cnt,
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
                    target_symbol: Symbol = state.get_current_symbol()
                    target_cnt = state.step_cnt
                    assert target_symbol is not None
                else:
                    assert state.get_current_symbol() is None
                    history_symbols: List[Symbol] = state.get_history_symbols(
                        mode, self.look_left_only
                    )
                    target_symbol = history_symbols[state.refine_step_cnt]
                    target_cnt = state.refine_step_cnt

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
                state.apply_pred(prod)
            state.step()
            return prev_tensor_dict

        def save_initial_pred(
            state: TransformerState,
            prev_tensor_dict: Dict[str, Union[List[TensorPromise], TensorPromise]],
        ):
            if isinstance(state, TransformerStatePred):
                actions = copy.deepcopy(
                    state.get_history_actions("infer", self.look_left_only)
                )
                assert len(actions) == len(state.preds), "Diff: {} {}".format(
                    len(actions), len(state.preds)
                )
                state.save_init_preds(actions)
            return prev_tensor_dict

        def apply_prod_final(
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

            history_actions: List[Action] = state.get_history_actions(
                "refine", self.look_left_only
            )
            cur_refine_step = state.refine_step_cnt
            symbol = history_actions[cur_refine_step]
            if isinstance(state, TransformerStateGold):
                # Loss for C T (?)
                if not self.use_ct_loss or symbol in ["C", "T"]:
                    if not state.skip_refinement:
                        refine_prod = prev_tensor_dict["prod_refine"].result
                        state.apply_loss(cur_refine_step, refine_prod)
                    if not state.skip_arbitrator:
                        arbitrate_prod = prev_tensor_dict["prod_arbitrate"].result
                        state.apply_loss(cur_refine_step, arbitrate_prod)
            else:
                refine_prod = prev_tensor_dict["prod_refine"].result
                arbitrate_prod = prev_tensor_dict["prod_arbitrate"].result
                assert isinstance(state, TransformerStatePred)
                ori_action: Action = history_actions[cur_refine_step]
                ori_symbol: Symbol = ori_action[0]
                refine_pred_idx = torch.argmax(refine_prod).item()

                if ori_symbol in ["T", "C"]:
                    ori_pred_idx = ori_action[1]
                    refine_action = (ori_symbol, refine_pred_idx)
                else:
                    ori_pred_idx = SemQL.semql.action_to_aid[ori_action]
                    refine_action = SemQL.semql.aid_to_action[refine_pred_idx]

                # get pred_action
                if ori_action != refine_action:
                    if self.refine_all or ori_symbol in ["C", "T"]:
                        # Compare
                        if self.use_arbitrator and (
                            arbitrate_prod[refine_pred_idx]
                            < arbitrate_prod[ori_pred_idx]
                        ):
                            final_action = ori_action
                        else:
                            final_action = refine_action

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
            return prev_tensor_dict

        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done)
            .Do(
                LogicUnit.If(state_class.is_to_infer)
                .Then(embed_history_actions_per_mode("infer"))
                .Then(embed_history_symbols_per_mode("infer"))
                .Then(combine_embeddings_per_mode("infer"))
                .Then(pass_transformer_per_mode("infer"))
                .Then(pass_out_linear_per_mode("infer"))
                .Then(calc_prod_per_mode("infer"))
                .Then(apply_prod_infer)
            )
            .Do(LogicUnit.If(state_class.is_initial_pred).Then(save_initial_pred))
            .Do(
                LogicUnit.If(state_class.is_to_refine)
                .Then(embed_history_actions_per_mode("refine"))
                .Then(embed_history_symbols_per_mode("refine"))
                .Then(combine_embeddings_per_mode("refine"))
                .Then(pass_transformer_per_mode("refine"))
                .Then(pass_out_linear_per_mode("refine"))
                .Then(calc_prod_per_mode("refine"))
            )
            .Do(
                LogicUnit.If(state_class.is_to_arbitrate)
                .Then(embed_history_actions_per_mode("arbitrate"))
                .Then(embed_history_symbols_per_mode("arbitrate"))
                .Then(combine_embeddings_per_mode("arbitrate"))
                .Then(pass_transformer_per_mode("arbitrate"))
                .Then(pass_out_linear_per_mode("arbitrate"))
                .Then(calc_prod_per_mode("arbitrate"))
            )
            .Do(LogicUnit.If(state_class.is_to_apply).Then(apply_prod_final))
        ).states

        if golds:
            return TransformerStateGold.combine_loss(states)
        else:
            return TransformerStatePred.get_preds(states), states
