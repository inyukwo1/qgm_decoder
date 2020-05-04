import copy
import logging

import torch
import torch.nn as nn
import torch.nn.utils

from src import utils
from models.wrapper_model import EncoderDecoderModel
from src.dataset import Batch


log = logging.getLogger(__name__)


class EnsembleWrapper(nn.Module):
    def __init__(self, cfg):
        super(EnsembleWrapper, self).__init__()
        self.cfg = cfg
        self.sub_models = nn.ModuleList([])

    def load_models(self, model_paths):
        for model_path in model_paths:
            log.info("Model path: {}".format(model_path))
            # Create sub model
            model = EncoderDecoderModel(self.cfg)

            # Load trained weights
            log.info("Load pretrained model from {}".format(model_path))
            pretrained_model = torch.load(
                model_path, map_location=lambda storage, loc: storage
            )
            change_key = False
            if change_key:
                new_pretrained_model = {}
                for key, value in copy.deepcopy(pretrained_model).items():
                    continue
                    new_key = key.replace("decoder_", "infer_")
                    new_key = new_key.replace(
                        "action_affine_layer", "infer_action_affine"
                    )
                    new_key = new_key.replace(
                        "symbol_affine_layer", "infer_symbol_affine_layer"
                    )
                    new_key = new_key.replace(".action", ".infer_action")
                    new_key = new_key.replace(
                        ".infer_action_affine", ".infer_action_affine_layer"
                    )
                    new_key = new_key.replace(
                        ".tgt_linear_layer", ".infer_tgt_linear_layer"
                    )
                    new_key = new_key.replace("encoder.ra", "encoder.0.ra")
                    new_key = new_key.replace("encoder.tab", "encoder.0.tab")
                    new_key = new_key.replace("encoder.sen", "encoder.0.sen")
                    new_key = new_key.replace("encoder.col", "encoder.0.col")
                    if "grammar" not in new_key:
                        new_key = new_key.replace("key_emb.weight", "key_embs.0.weight")
                    new_key = new_key.replace(
                        "decoder.grammar.infer_action_emb.weight",
                        "decoder.grammar.action_emb.weight",
                    )
                    new_key = new_key.replace(
                        "decoder.column_similarity", "decoder.infer_column_similarity"
                    )
                    new_key = new_key.replace(
                        "decoder.table_similarity", "decoder.infer_table_similarity"
                    )
                    new_pretrained_model[new_key] = value

                pretrained_model_ = copy.deepcopy(new_pretrained_model)
            else:
                pretrained_model_ = copy.deepcopy(pretrained_model)

            for k in pretrained_model.keys():
                if k not in model.state_dict().keys():
                    del pretrained_model_[k]
            model.load_state_dict(pretrained_model_)

            # Load word embedding
            model.word_emb = (
                None
                if self.cfg.is_bert
                else utils.load_word_emb(self.cfg.glove_embed_path)
            )

            # Append sub model
            self.sub_models.append(model)

    def forward(self):
        pass

    def parse(self, examples):
        # pass and get table from the first model
        init_preds = self.sub_models[0].parse(examples)[0]
        init_preds2 = self.sub_models[1].parse(examples)[0]
        return init_preds, None

        # narrow down schema using 1- hop scheme
        new_data_list = []
        for pred, pred2, example in zip(
            init_preds["preds"], init_preds2["preds"], examples
        ):
            # Append neighbors
            new_data = copy.deepcopy(example.data)

            selected_table_indices = set([item[1] for item in pred if item[0] == "T"])
            # selected_table_indices.update([item[1] for item in pred2 if item[0] == "T"])

            gt_table_indices = list(
                set([item[1] for item in example.gt if item[0] == "T"])
            )

            # selected_table_indices = utils.append_one_hop_neighbors(
            #     new_data["db"]["neighbors"], selected_table_indices
            # )

            # Create sub schema
            sub_schema = utils.create_sub_schema(
                selected_table_indices,
                new_data["table_names"],
                new_data["column_names"],
                new_data["col_set"],
            )

            # Alter schema
            for key, item in sub_schema.items():
                new_data[key] = item

            # Create sub relation
            sub_relation = utils.create_sub_relation(
                new_data["relation"],
                sub_schema["column_mapping"],
                sub_schema["table_mapping"],
            )

            # Alter relation
            for key, item in sub_relation.items():
                new_data["relation"][key] = item

            # Append
            new_data_list.append(new_data)

        new_examples = utils.to_batch_seq(new_data_list)

        # pass second and third model for inference
        new_preds = self.sub_models[0].parse(new_examples)[0]

        # Alter pred back to original index
        final_preds = []
        for new_data, new_pred in zip(new_data_list, new_preds["preds"]):
            # Reverse mapping (one-to-one anyway)
            column_mapping_rev = {
                item: key for key, item in new_data["column_mapping"].items()
            }
            table_mapping_rev = {
                item: key for key, item in new_data["table_mapping"].items()
            }

            # Modify
            modified_pred = []
            for item in new_pred:
                if item[0] == "C":
                    modified_pred += [(item[0], column_mapping_rev[item[1]])]
                elif item[0] == "T":
                    modified_pred += [(item[0], table_mapping_rev[item[1]])]
                else:
                    modified_pred += [item]

            final_preds.append(modified_pred)

        new_preds["preds"] = final_preds
        return new_preds, None
