import copy
import logging

import torch
import torch.nn as nn
import torch.nn.utils

from src import utils
from src.dataset import Batch
from rule.semql.semql import SemQL
from models.wrapper_model import EncoderDecoderModel


log = logging.getLogger(__name__)


class SequentialEnsemble(nn.Module):
    def __init__(self, cfg):
        super(SequentialEnsemble, self).__init__()
        self.cfg = cfg
        self.is_cuda = cfg.cuda != -1
        self.sub_models = nn.ModuleList([])

    def load_models(self, model_paths):
        for model_path in model_paths:
            # Create sub model
            model = EncoderDecoderModel(self.cfg)

            # Load trained weights
            log.info("Load pretrained model from {}".format(model_path))
            pretrained_model = torch.load(
                model_path, map_location=lambda storage, loc: storage
            )
            new_pretrained_model = copy.deepcopy(pretrained_model)
            for key, item in new_pretrained_model.items():
                if "encoder.0" in key:
                    pretrained_model[key.replace("encoder.0", "encoder")] = item
                elif "key_embs.0" in key:
                    pretrained_model[key.replace("key_embs.0", "key_embs")] = item
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
        batch = Batch(examples, is_cuda=self.is_cuda)
        with torch.no_grad():
            # Encode
            (
                src_encodings1,
                table_embeddings1,
                schema_embeddings1,
                _,
            ) = self.sub_models[0].encode(batch)
            (
                src_encodings2,
                table_embeddings2,
                schema_embeddings2,
                _,
            ) = self.sub_models[1].encode(batch)

            b_size = len(src_encodings1)

            # Decode
            ensembled_pred = []
            not_done = True

            while not_done:
                states1 = self.sub_models[0].decoder(
                    b_size,
                    src_encodings1,
                    table_embeddings1,
                    schema_embeddings1,
                    batch.src_sents_len,
                    batch.col_num,
                    batch.table_len,
                    batch.col_tab_dic,
                    pred_guide=ensembled_pred,
                    is_ensemble=True,
                )
                states2 = self.sub_models[1].decoder(
                    b_size,
                    src_encodings2,
                    table_embeddings2,
                    schema_embeddings2,
                    batch.src_sents_len,
                    batch.col_num,
                    batch.table_len,
                    batch.col_tab_dic,
                    pred_guide=ensembled_pred,
                    is_ensemble=True,
                )
                pred1 = states1[0].get_preds(states1)["preds"][0]
                pred2 = states2[0].get_preds(states2)["preds"][0]
                for idx in range(len(ensembled_pred), min(len(pred1), len(pred2))):
                    action1 = pred1[idx]
                    action2 = pred2[idx]
                    if action1 != action2:
                        # Get all tables
                        selected_table_indices = set(
                            [item[1] for item in pred1 if item[0] == "T"]
                        )
                        selected_table_indices.update(
                            [item[1] for item in pred2 if item[0] == "T"]
                        )
                        # Narrow down schema
                        new_data = copy.deepcopy(examples[0].data)
                        sub_schema = utils.create_sub_schema(
                            selected_table_indices,
                            new_data["table_names"],
                            new_data["column_names"],
                            new_data["col_set"],
                        )
                        for key, item in sub_schema.items():
                            new_data[key] = item
                        sub_relation = utils.create_sub_relation(
                            new_data["relation"],
                            sub_schema["column_mapping"],
                            sub_schema["table_mapping"],
                        )
                        for key, item in sub_relation.items():
                            new_data["relation"][key] = item
                        # Create new example and batch
                        new_examples = utils.to_batch_seq([new_data])
                        new_batch = Batch(new_examples, is_cuda=self.is_cuda)
                        # Encode
                        (
                            src_encodings3,
                            table_embeddings3,
                            schema_embeddings3,
                            _,
                        ) = self.sub_models[2].encode(new_batch)
                        narrowed_pred = []
                        for item in ensembled_pred:
                            if item[0] == "C":
                                narrowed_pred += [
                                    (item[0], new_data["column_mapping"][item[1]])
                                ]
                            elif item[0] == "T":
                                narrowed_pred += [
                                    (item[0], new_data["table_mapping"][item[1]])
                                ]
                            else:
                                narrowed_pred += [item]

                        states3 = self.sub_models[2].decoder(
                            b_size,
                            src_encodings3,
                            table_embeddings3,
                            schema_embeddings3,
                            new_batch.src_sents_len,
                            new_batch.col_num,
                            new_batch.table_len,
                            new_batch.col_tab_dic,
                            pred_guide=narrowed_pred,
                            is_ensemble=True,
                        )
                        prod = states3[0].get_probs(idx)
                        # Choose one with higher prob from model3
                        assert action1[0] == action2[0], "{} {}".format(
                            action1, action2
                        )
                        if action1[0] == "C":
                            idx1 = new_data["column_mapping"][action1[1]]
                            idx2 = new_data["column_mapping"][action2[1]]
                        elif action2[0] == "T":
                            idx1 = new_data["table_mapping"][action1[1]]
                            idx2 = new_data["table_mapping"][action2[1]]
                        else:
                            idx1 = SemQL.semql.action_to_aid[action1]
                            idx2 = SemQL.semql.action_to_aid[action2]
                        # # First model
                        # new_action = action1
                        #
                        # # import random
                        # new_action = action1 if random.randint(0, 1) else action2

                        # Using third model
                        if prod[idx1] > prod[idx2]:
                            new_action = action1
                        else:
                            new_action = action2

                        ensembled_pred += [new_action]
                        break
                    else:
                        ensembled_pred += [action1]

                not_done = len(pred1) != len(pred2) or len(ensembled_pred) != len(pred1)

        return (
            {
                "preds": [ensembled_pred],
                "refined_preds": [[]],
                "arbitrated_preds": [[]],
                "initial_preds": [[]],
            },
            _,
        )
