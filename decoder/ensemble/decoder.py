from typing import Dict
from torch import Tensor
import torch
import torch.nn as nn
from decoder.transformer_framework.decoder import TransformerDecoderFramework
from decoder.ensemble.state import EnsembleState
from framework.sequential_monad import SequentialMonad, WhileLogic, LogicUnit


class EnsembleDecoder(nn.Module):
    def __init__(self, cfg):
        super(EnsembleDecoder, self).__init__()
        self.cfg = cfg
        self.model_num = 0
        self.models = []

    def load_model(self):
        for model_name in self.cfg.model_names:
            path = self.cfg.model_path.format(model_name)
            pretrained_decoder = torch.load(path, map_location=lambda storage, loc: storage)
            pretrained_decoder = {key.replace("decoder.", ""): item for key, item in pretrained_decoder.items() if "decoder." in key}

            model = TransformerDecoderFramework(self.cfg)
            model.load_state_dict(pretrained_decoder)

            self.models += [model]
        self.model_num = len(self.models)

        # Setting
        for model in self.models:
            model.eval()
            model.cuda()

    def forward(self,
        encoded_src,
        encoded_col,
        encoded_tab,
        src_lens,
        col_lens,
        tab_lens,
        col_tab_dic,
        golds=None
        ):
        b_size = len(encoded_src)
        states = [
                EnsembleState(
                    encoded_src[b_idx, : src_lens[b_idx]],
                    encoded_col[b_idx, : col_lens[b_idx]],
                    encoded_tab[b_idx, : tab_lens[b_idx]],
                    src_lens[b_idx],
                    col_lens[b_idx],
                    tab_lens[b_idx],
                    col_tab_dic[b_idx],
                    )
                for b_idx in range(b_size)
            ]

        def get_individual_model_score(state: EnsembleState, _) -> Dict[str, Tensor]:
            # Pass to individual models and get
            probs = []
            for model in self.models:
                _, prob = model(
                    state.get_encoded_src(),
                    state.get_encoded_col(),
                    state.get_encoded_tab(),
                    state.get_src_lens(),
                    state.get_col_lens(),
                    state.get_tab_lens(),
                    state.get_col_tab_dic(),
                    target_step=state.get_pred_len(),
                    pred_guide=state.get_preds(),
                )
                probs += [prob[0]]

            prev_tensor_dict = {"probs": probs}
            return prev_tensor_dict

        def vote(state: EnsembleState, prev_tensor_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # Compute Avg
            scores = prev_tensor_dict["probs"]
            avg = sum(scores) / len(scores)

            # Find highest prob
            pred_idx = torch.argmax(avg).item()

            prev_tensor_dict.update({"pred_idx": pred_idx})
            return prev_tensor_dict

        def update_state(state: EnsembleState, prev_tensor_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # Find highest and create action and save it
            pred_idx = prev_tensor_dict["pred_idx"]
            symbol = state.get_first_nonterminal()
            if symbol in ["C", "T"]:
                action = (symbol, pred_idx)
            else:
                action = self.models[0].grammar.aid_to_action[pred_idx]
            nonterminals = self.models[0].grammar.parse_nonterminal_symbol(action)

            # Save
            state.save_pred(action, nonterminals)

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
