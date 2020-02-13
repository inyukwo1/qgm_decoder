from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
from src.utils import to_batch_seq
from tqdm import tqdm
from copy import copy
import torch

from src.models.model import IRNet
from src.dataset import Batch


def to_ref_examples(examples, is_bert):
    new_examples = [copy(example) for example in examples]
    # TODO
    if is_bert:
        assert False
    else:
        for example in new_examples:
            example.src_sent = ["<unk>" for _ in example.src_sent]
    return new_examples


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def view_captum(model: IRNet, sql_data, table_data, is_bert, is_qgm):
    model.eval()

    def forward_until_step(src_embeddings, examples, step):
        examples = examples * len(src_embeddings)
        _, _, gold_score, pred_score = model.forward_until_step(
            src_embeddings, examples, step
        )
        return gold_score

    lig = LayerIntegratedGradients(forward_until_step, model.captum_iden)

    perm = range(len(sql_data))
    for st in tqdm(range(len(sql_data))):
        ed = st + 1 if st + 1 < len(perm) else len(perm)
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, is_qgm=is_qgm)
        ref_examples = to_ref_examples(examples, is_bert)
        for action_idx, action in enumerate(examples[0].truth_actions):
            batch = Batch(examples, is_cuda=model.is_cuda)
            ref_batch = Batch(ref_examples, is_cuda=model.is_cuda)
            src_embeddings = model.gen_x_batch(batch.src_sents)
            ref_embeddings = model.gen_x_batch(ref_batch.src_sents)
            attributions, delta = lig.attribute(
                inputs=src_embeddings,
                baselines=ref_embeddings,
                additional_forward_args=(examples, action_idx),
                return_convergence_delta=True,
            )
            attributions_sum = summarize_attributions(attributions)
            gold_action, pred_action, gold_score, pred_score = model.forward_until_step(
                src_embeddings, examples, action_idx
            )
            sentence = [" ".join(words) for words in examples[0].src_sent]
            print("gold_score: {}".format(gold_score.squeeze(0).item()))
            if pred_action == gold_action:
                print("CORRECT")
            else:
                print("WRONG")
            vis = viz.VisualizationDataRecord(
                attributions_sum,
                pred_score.squeeze(0),
                pred_action,
                gold_action,
                gold_action,
                attributions_sum.sum(),
                sentence,
                delta,
            )
            viz.visualize_text([vis])
