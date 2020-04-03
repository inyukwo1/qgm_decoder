from typing import Dict
from torch import Tensor
import torch
import torch.nn as nn
from encoder.ra_transformer.encoder import RA_Transformer_Encoder
from decoder.transformer_framework.decoder import TransformerDecoderFramework
from decoder.ensemble.state import EnsembleState
from framework.sequential_monad import SequentialMonad, WhileLogic, LogicUnit
from rule.semql.semql import SemQL


class EnsembleDecoder(nn.Module):
    def __init__(self, cfg):
        super(EnsembleDecoder, self).__init__()
        self.cfg = cfg
        self.model_num = 0
        self.encoders = []
        self.decoders = []

    def load_model(self):
        key_embs = []
        for idx, model_name in enumerate(self.cfg.model_names):
            path = self.cfg.model_path.format(model_name)
            pretrained_weights = torch.load(path, map_location=lambda storage, loc: storage)
            pretrained_encoder = {".".join(key.split(".")[1:]) : item for key, item in pretrained_weights.items() if
                                  "encoder." in key}
            pretrained_decoder = {key.replace("decoder.", ""): item for key, item in pretrained_weights.items() if
                                  "decoder." in key}

            encoder = RA_Transformer_Encoder(self.cfg)
            encoder.load_state_dict(pretrained_encoder)
            self.encoders += [encoder]

            decoder = TransformerDecoderFramework(self.cfg)
            decoder.load_state_dict(pretrained_decoder)
            self.decoders += [decoder]
            key_embs += [pretrained_weights["key_emb.weight"]]

        self.model_num = len(self.decoders)

        # Setting
        for model in self.decoders:
            model.eval()
            model.cuda()
        for model in self.encoders:
            model.eval()
            model.cuda()
        return key_embs

    def forward(self,
        srcs,
        cols,
        tabs,
        src_len,
        col_len,
        tab_len,
        src_mask,
        col_mask,
        tab_mask,
        relation_matrix,
        col_tab_dic,
        gt,
        ):
        b_size = len(srcs[0])

        states = [EnsembleState(src_len[b_idx], col_len[b_idx], tab_len[b_idx], col_tab_dic[b_idx], gt[b_idx]) for b_idx in range(b_size)]

        # Encoder
        for idx, encoder in enumerate(self.encoders):
            src = srcs[idx]
            col = cols[idx]
            tab = tabs[idx]
            encoded_src, encoded_col, encoded_tab = \
                encoder(src, col, tab, src_len, col_len, tab_len, src_mask, col_mask, tab_mask, relation_matrix)
            # save into state
            for b_idx in range(b_size):
                states[b_idx].append_encoded_values(encoded_src[b_idx, : src_len[b_idx]],
                    encoded_col[b_idx, : col_len[b_idx]],
                    encoded_tab[b_idx, : tab_len[b_idx]])

        def get_individual_model_score(state: EnsembleState, _) -> Dict[str, Tensor]:
            # Pass to individual models and get
            new_child_states = []
            # Get states
            child_states = state.get_child_states()

            for idx, model in enumerate(self.decoders):
                _, new_child_state = model(
                    state.get_encoded_src(idx),
                    state.get_encoded_col(idx),
                    state.get_encoded_tab(idx),
                    state.get_src_lens(),
                    state.get_col_lens(),
                    state.get_tab_lens(),
                    state.get_col_tab_dic(),
                    target_step=state.get_pred_len(),
                    states=[child_states[idx]] if child_states else None,
                )
                new_child_states += [new_child_state[0]]

            # Save state_list
            state.save_children_state(new_child_states)

            probs = [s.get_probs() for s in new_child_states]
            prev_tensor_dict = {"probs": probs}
            return prev_tensor_dict

        def vote(state: EnsembleState, prev_tensor_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # Compute Avg
            scores = prev_tensor_dict["probs"]

            # aids = [action[1] if action[0] in ["C", "T"] else SemQL.semql.action_to_aid[action] for action in state.gt]
            # if state.step_cnt < len(state.gt):
            #     gt = state.gt[state.step_cnt]
            #     gt = gt[1] if gt[0] in ["C", "T"] else SemQL.semql.action_to_aid[gt]
            #     gt_action = state.gt[state.step_cnt]
            # else:
            #     gt = -1
            #     gt_action = "-1"

            max_indices = [torch.argmax(score).item() for score in scores]

            # Find highest prob
            # Soft voting
            soft_voting = True
            avg = sum(scores) / len(scores)
            act_only = False
            current_symbol = state.nonterminals[0]
            if act_only and current_symbol not in ["A", "C", "T"]:
                pred_idx = max_indices[0]
            else:
                if soft_voting:
                    pred_idx = torch.argmax(avg).item()
                # Hard voting
                #elif False:
                elif len(max_indices) == 3:
                    # Assuming that 3 models are given
                    # pred Same
                    if max_indices[0] == max_indices[1]:
                        pred_idx = max_indices[0]
                    else:
                        # Pred differently. Need an arbitrator.
                        if max_indices[2] in max_indices[0:2]:
                            # Arbitrator chooses a side
                            pred_idx = max_indices[2]
                        else:
                            # If arbitrator thinks differently. Soft voting
                            pred_idx = torch.argmax(avg).item()
                else:
                    # Hard voting
                    from collections import Counter
                    counter = Counter(max_indices)
                    max_num = 0
                    max_list = []
                    for key, value in counter.items():
                        if value > max_num:
                            max_num = value
                            max_list = [key]
                        elif value == max_num:
                            max_list += [key]

                    # Soft voting with max_list
                    max_value = float("-inf")
                    max_idx = -1
                    for idx in max_list:
                        value = avg[idx]
                        if value >= max_value:
                            max_idx = idx
                            max_value = value
                    pred_idx = max_idx

            # def printt(tag):
            #     with open("tmp.txt", "a") as f:
            #         f.write(tag+"\n")
            #         f.write("aids: {}\n".format(aids))
            #         f.write("ground_truth: {} {}\n".format(gt_action, gt))
            #         f.write("avg_idx: {}\n".format(pred_idx))
            #         f.write("avg_score: {}\n".format(avg))
            #         for score in scores:
            #             max_idx = torch.argmax(score).item()
            #             f.write("max_idx: {}\n".format(max_idx))
            #             f.write("score: {}\n\n".format(score))
            #
            # # If all pred same
            # with open("tmp.txt", "a") as f:
            #     f.write("cnt!\n")
            #
            # right_wrong = [item == gt for item in max_indices]
            # # All wrong
            # if gt not in max_indices:
            #     if pred_idx == gt:
            #         printt("All wrong and final right")
            #     else:
            #         printt("All wrong and final wrong")
            # elif False in right_wrong:
            #     if pred_idx == gt:
            #         printt("some right and final right")
            #     else:
            #         printt("some right and final wrong")
            # else:
            #     if pred_idx == gt:
            #         printt("All right and final right")
            #     else:
            #         printt("All right and final wrong")

            prev_tensor_dict.update({"pred_idx": pred_idx})
            return prev_tensor_dict

        def update_state(state: EnsembleState, prev_tensor_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # Find highest and create action and save it
            pred_idx = prev_tensor_dict["pred_idx"]
            symbol = state.get_first_nonterminal()
            if symbol in ["C", "T"]:
                action = (symbol, pred_idx)
            else:
                action = self.decoders[0].grammar.aid_to_action[pred_idx]
            nonterminals = self.decoders[0].grammar.parse_nonterminal_symbol(action)

            # Save
            state.save_pred(action, nonterminals)
            state.update_children_state()

            return None

        states = SequentialMonad(states)(
            WhileLogic.While(EnsembleState.is_not_done)
            .Do(
                LogicUnit.If(EnsembleState.is_not_done)
                .Then(get_individual_model_score)
                .Then(vote)
                .Then(update_state)
            )
        ).states

        return EnsembleState.get_final_pred(states)
