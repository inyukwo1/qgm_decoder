import copy
import logging

import torch
import torch.nn as nn
import torch.nn.utils

from src import utils
from src.dataset import Batch
from models.wrapper_model import EncoderDecoderModel


log = logging.getLogger(__name__)


class SequentialEnsemble(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sub_moduels = nn.ModuleList([])

    def load_models(self, model_paths):
        for model_path in model_paths:
            # Create sub model
            model = EncoderDecoderModel(self.cfg)

            # Load trained weights
            log.info("Load pretrained model from {}".format(model_path))
            pretrained_model = torch.load(
                model_path, map_location=lambda storage, loc: storage
            )
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
            (src_encodings1, table_embeddings1, schema_embeddings1) = self.model[
                0
            ].encode(batch)
            (src_encodings2, table_embeddings2, schema_embeddings2) = self.model[
                1
            ].encode(batch)

            b_size = len(src_encodings1)

            # Decode
            ensembled_pred = []
            step_cnt = 0
            not_done = True
            states1 = states2 = None

            while not_done:
                states1 = self.model[0].decoder(
                    b_size,
                    src_encodings1,
                    table_embeddings1,
                    schema_embeddings1,
                    batch.src_sents_len,
                    batch.col_num,
                    batch.table_len,
                    batch.col_tab_dic,
                    target_step=step_cnt,
                    states=states1,
                    is_ensemble=True,
                )
                states2 = self.model[1].decoder(
                    b_size,
                    src_encodings2,
                    table_embeddings2,
                    schema_embeddings2,
                    batch.src_sents_len,
                    batch.col_num,
                    batch.table_len,
                    batch.col_tab_dic,
                    target_step=step_cnt,
                    states=states2,
                    is_ensemble=True,
                )

                # Vote next pred
                pred = None
                ensembled_pred += [pred]
                step_cnt += 1

        return ensembled_pred
