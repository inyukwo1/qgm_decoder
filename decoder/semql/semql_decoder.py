import numpy as np
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable

import torch
import logging
from preprocess.rule.semql import semQL
from preprocess.rule.semql import semQL as define_rule
from encoder.irnet import nn_utils
from decoder.semql.beam import Beams, ActionInfo
from encoder.irnet.pointer_net import PointerNet
log = logging.getLogger(__name__)

SKETCH_LIST = ["Root1", "Root", "Sel", "N", "Filter", "Sup", "Order"]
DETAIL_LIST = ["A", "C", "T"]


class SemQL_Decoder(nn.Module):
    def __init__(self, cfg):
        super(SemQL_Decoder, self).__init__()
        self.cfg = cfg
        self.embed_size = 1024 if cfg.is_bert else 300
        self.is_cuda = cfg.cuda != -1
        self.grammar = semQL.Grammar()
        self.use_column_pointer = cfg.column_pointer

        if self.is_cuda:
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_tensor = torch.FloatTensor

        hidden_size = cfg.hidden_size
        att_vec_size = cfg.att_vec_size
        type_embed_size = cfg.type_embed_size
        action_embed_size = cfg.action_embed_size
        input_dim = action_embed_size + att_vec_size + type_embed_size

        self.encoder_lstm = nn.LSTM(
            self.embed_size, hidden_size // 2, bidirectional=True, batch_first=True,
        )

        self.decode_max_time_step = 40
        self.action_embed_size = action_embed_size
        self.type_embed_size = type_embed_size

        self.decoder_cell_init = nn.Linear(hidden_size, hidden_size)
        self.lf_decoder_lstm = nn.LSTMCell(input_dim, hidden_size)
        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, hidden_size)

        self.att_sketch_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(
            hidden_size + hidden_size, att_vec_size, bias=False
        )
        self.lf_att_vec_linear = nn.Linear(
            hidden_size + hidden_size, att_vec_size, bias=False
        )

        self.prob_att = nn.Linear(att_vec_size, 1)
        self.prob_len = nn.Linear(1, 1)

        self.production_embed = nn.Embedding(
            len(self.grammar.prod2id), action_embed_size
        )
        self.type_embed = nn.Embedding(len(self.grammar.type2id), type_embed_size)
        self.production_readout_b = nn.Parameter(
            torch.FloatTensor(len(self.grammar.prod2id)).zero_()
        )

        self.N_embed = nn.Embedding(
            len(define_rule.N._init_grammar()), action_embed_size
        )

        self.read_out_act = (
            torch.tanh if cfg.readout == "non_linear" else nn_utils.identity
        )

        self.query_vec_to_action_embed = nn.Linear(
            att_vec_size, action_embed_size, bias=cfg.readout == "non_linear",
        )

        self.production_readout = lambda q: torch.nn.functional.linear(
            self.read_out_act(self.query_vec_to_action_embed(q)),
            self.production_embed.weight,
            self.production_readout_b,
        )

        self.q_att = nn.Linear(hidden_size, self.embed_size)

        self.column_rnn_input = nn.Linear(self.embed_size, action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(self.embed_size, action_embed_size, bias=False)

        self.column_pointer_net = PointerNet(
            hidden_size, self.embed_size, attention_type=cfg.column_att
        )

        self.table_pointer_net = PointerNet(
            hidden_size, self.embed_size, attention_type=cfg.column_att
        )

        self.dropout = nn.Dropout(cfg.dropout)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.N_embed.weight.data)
        log.info("Use Column Pointer: {}".format(True if self.use_column_pointer else False))

    def _init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

    def forward(self, batch, step=None): # Step for captum
        b_size = batch.b_size
        sketches = batch.semql_sketch
        src_encodings = batch.sen_encoding
        table_embedding = batch.col_emb
        schema_embedding = batch.tab_emb
        dec_init_vec = batch.last_cell

        # Variables to remove examples
        tgt_actions = batch.semql
        tab_cols = batch.col
        table_names = batch.tab

        table_appear_mask = batch.table_appear_mask

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        h_tm1 = self._init_decoder_state(dec_init_vec)
        action_probs = [[] for _ in range(b_size)]

        zero_action_embed = Variable(self.new_tensor(self.action_embed_size).zero_())
        zero_type_embed = Variable(self.new_tensor(self.type_embed_size).zero_())

        for t in range(batch.max_semql_sketch_num):
            if t == 0:
                x = Variable(
                    self.new_tensor(
                        len(batch), self.sketch_decoder_lstm.input_size
                    ).zero_(),
                    requires_grad=False,
                )
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, sketch in enumerate(sketches):

                    if t < len(sketch):
                        # get the last action
                        # This is the action embedding
                        action_tm1 = sketch[t - 1]
                        if type(action_tm1) in [
                            define_rule.Root1,
                            define_rule.Root,
                            define_rule.Sel,
                            define_rule.Filter,
                            define_rule.Sup,
                            define_rule.N,
                            define_rule.Order,
                        ]:
                            a_tm1_embed = self.production_embed.weight[
                                self.grammar.prod2id[action_tm1.production]
                            ]
                        else:
                            print(action_tm1, "only for sketch")
                            quit()
                            a_tm1_embed = zero_action_embed
                            pass
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, sketch in enumerate(sketches):
                    if t < len(sketch):
                        action_tm = sketch[t - 1]
                        pre_type = self.type_embed.weight[
                            self.grammar.type2id[type(action_tm)]
                        ]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(
                x,
                h_tm1,
                src_encodings,
                utterance_encodings_sketch_linear,
                self.sketch_decoder_lstm,
                self.sketch_att_vec_linear,
                src_token_mask=src_mask,
                return_att_weight=True,
            )

            # get the Root possibility
            apply_rule_prob = torch.softmax(self.production_readout(att_t), dim=-1)

            if t == step:
                gold_action = sketches[0][t].production
                gold_action_id = self.grammar.prod2id[gold_action]
                pred_action_id = torch.argmax(apply_rule_prob[-1]).item()
                pred_action = self.grammar.id2prod[pred_action_id]
                pred_probs = apply_rule_prob[:, pred_action_id]
                gold_probs = apply_rule_prob[:, gold_action_id]
                return gold_action, pred_action, gold_probs, pred_probs

            for e_id, sketch in enumerate(sketches):
                if t < len(sketch):
                    action_t = sketch[t]
                    act_prob_t_i = apply_rule_prob[
                        e_id, self.grammar.prod2id[action_t.production]
                    ]
                    action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        sketch_prob_var = torch.stack(
            [
                torch.stack(action_probs_i, dim=0).log().sum()
                for action_probs_i in action_probs
            ],
            dim=0,
        )

        batch_table_dict = batch.col_tab_dic
        table_enable = np.zeros(shape=(b_size))
        action_probs = [[] for _ in range(b_size)]

        h_tm1 = self._init_decoder_state(dec_init_vec)

        detail_t = 0

        for t in range(batch.max_semql_action_num):
            if t == 0:
                x = Variable(
                    self.new_tensor(
                        len(batch), self.lf_decoder_lstm.input_size
                    ).zero_(),
                    requires_grad=False,
                )
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, tgt_action in enumerate(tgt_actions):
                    if t < len(tgt_action):
                        action_tm1 = tgt_action[t - 1]
                        if type(action_tm1) in [
                            define_rule.Root1,
                            define_rule.Root,
                            define_rule.Sel,
                            define_rule.Filter,
                            define_rule.Sup,
                            define_rule.N,
                            define_rule.Order,
                        ]:

                            a_tm1_embed = self.production_embed.weight[
                                self.grammar.prod2id[action_tm1.production]
                            ]

                        else:
                            if isinstance(action_tm1, define_rule.C):
                                a_tm1_embed = self.column_rnn_input(
                                    table_embedding[e_id, action_tm1.id_c]
                                )
                            elif isinstance(action_tm1, define_rule.T):
                                a_tm1_embed = self.column_rnn_input(
                                    schema_embedding[e_id, action_tm1.id_c]
                                )
                            elif isinstance(action_tm1, define_rule.A):
                                a_tm1_embed = self.production_embed.weight[
                                    self.grammar.prod2id[action_tm1.production]
                                ]
                            else:
                                print(action_tm1, "not implement")
                                quit()
                                a_tm1_embed = zero_action_embed
                                pass

                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                # tgt t-1 action type
                for e_id, tgt_action in enumerate(tgt_actions):
                    if t < len(tgt_action):
                        action_tm = tgt_action[t - 1]
                        pre_type = self.type_embed.weight[
                            self.grammar.type2id[type(action_tm)]
                        ]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)

                inputs.append(pre_types)

                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(
                x,
                h_tm1,
                src_encodings,
                utterance_encodings_lf_linear,
                self.lf_decoder_lstm,
                self.lf_att_vec_linear,
                src_token_mask=src_mask,
                return_att_weight=True,
            )

            apply_rule_prob = torch.softmax(self.production_readout(att_t), dim=-1)
            table_appear_mask_val = torch.from_numpy(table_appear_mask)
            if self.cuda:
                table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = torch.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(
                    src_encodings=table_embedding,
                    query_vec=att_t.unsqueeze(0),
                    src_token_mask=None,
                ) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=table_embedding,
                    query_vec=att_t.unsqueeze(0),
                    src_token_mask=None,
                ) * (
                    1 - table_appear_mask_val
                ) * (
                    1 - gate
                )
            else:
                weights = self.column_pointer_net(
                    src_encodings=table_embedding,
                    query_vec=att_t.unsqueeze(0),
                    src_token_mask=batch.table_token_mask,
                )

            weights.data.masked_fill_(batch.table_token_mask.bool(), -float("inf"))

            column_attention_weights = torch.softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(
                src_encodings=schema_embedding,
                query_vec=att_t.unsqueeze(0),
                src_token_mask=None,
            )

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float("inf"))
            table_dict = [
                batch_table_dict[x_id][int(x)]
                for x_id, x in enumerate(table_enable.tolist())
            ]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float("inf"))

            table_weights = torch.softmax(table_weights, dim=-1)

            if step == detail_t + batch.max_semql_sketch_num:
                gold_action_t = tgt_actions[0][t]
                if isinstance(gold_action_t, define_rule.C):
                    gold_action_id = gold_action_t.id_c
                    gold_action = "C({})".format(
                        " ".join(tab_cols[0][gold_action_id])
                    )
                    pred_action_id = torch.argmax(column_attention_weights[-1]).item()
                    pred_action = "C({})".format(
                        " ".join(tab_cols[0][pred_action_id])
                    )
                    gold_probs = column_attention_weights[:, gold_action_id]
                    pred_probs = column_attention_weights[:, pred_action_id]
                    return gold_action, pred_action, gold_probs, pred_probs
                elif isinstance(gold_action_t, define_rule.T):
                    gold_action_id = gold_action_t.id_c
                    gold_action = "T({})".format(
                        " ".join(table_names[0][gold_action_id])
                    )
                    pred_action_id = torch.argmax(table_weights[-1]).item()
                    pred_action = "T({})".format(
                        " ".join(table_names[0][pred_action_id])
                    )
                    gold_probs = table_weights[:, gold_action_id]
                    pred_probs = table_weights[:, pred_action_id]
                    return gold_action, pred_action, gold_probs, pred_probs
                elif isinstance(gold_action_t, define_rule.A):
                    gold_action_id = self.grammar.prod2id[gold_action_t.production]
                    gold_action = gold_action_t.production
                    pred_action_id = torch.argmax(apply_rule_prob[-1]).item()
                    pred_action = self.grammar.id2prod[pred_action_id]
                    pred_probs = apply_rule_prob[:, pred_action_id]
                    gold_probs = apply_rule_prob[:, gold_action_id]
                    return gold_action, pred_action, gold_probs, pred_probs
                else:
                    pass

            for e_id, tgt_action in enumerate(tgt_actions):
                if t < len(tgt_action):
                    action_t = tgt_action[t]
                    if isinstance(action_t, define_rule.C):
                        table_appear_mask[e_id, action_t.id_c] = 1
                        table_enable[e_id] = action_t.id_c
                        act_prob_t_i = column_attention_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.T):
                        act_prob_t_i = table_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.A):
                        act_prob_t_i = apply_rule_prob[
                            e_id, self.grammar.prod2id[action_t.production]
                        ]
                        action_probs[e_id].append(act_prob_t_i)
                    else:
                        pass
            try:
                if (
                    isinstance(tgt_actions[0][t], define_rule.C)
                    or isinstance(tgt_actions[0][t], define_rule.T)
                    or isinstance(tgt_actions[0][t], define_rule.A)
                ):
                    detail_t += 1
            except:
                pass

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        lf_prob_var = torch.stack(
            [
                torch.stack(action_probs_i, dim=0).log().sum()
                for action_probs_i in action_probs
            ],
            dim=0,
        )
        assert step is None

        total_loss = [-sketch_prob_var[idx] + -lf_prob_var[idx] for idx in range(len(sketch_prob_var))]

        return {"total": total_loss, "sketch": -sketch_prob_var, "detail": -lf_prob_var}, tgt_actions

    def parse(
        self,
        batch
    ):
        beam_size=5
        src_encodings = batch.sen_encoding
        table_embedding = batch.col_emb
        schema_embedding = batch.tab_emb
        dec_init_vec = batch.last_cell

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        h_tm1 = self._init_decoder_state(dec_init_vec)

        t = 0
        beams = [Beams(is_sketch=True)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.decode_max_time_step:
            hyp_num = len(beams)
            exp_src_enconding = src_encodings.expand(
                hyp_num, src_encodings.size(1), src_encodings.size(2)
            )
            exp_src_encodings_sketch_linear = utterance_encodings_sketch_linear.expand(
                hyp_num,
                utterance_encodings_sketch_linear.size(1),
                utterance_encodings_sketch_linear.size(2),
            )
            if t == 0:
                with torch.no_grad():
                    x = Variable(
                        self.new_tensor(1, self.sketch_decoder_lstm.input_size).zero_()
                    )
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [
                        define_rule.Root1,
                        define_rule.Root,
                        define_rule.Sel,
                        define_rule.Filter,
                        define_rule.Sup,
                        define_rule.N,
                        define_rule.Order,
                    ]:
                        a_tm1_embed = self.production_embed.weight[
                            self.grammar.prod2id[action_tm1.production]
                        ]
                    else:
                        raise ValueError("unknown action %s" % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[
                        self.grammar.type2id[type(action_tm)]
                    ]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(
                x,
                h_tm1,
                exp_src_enconding,
                exp_src_encodings_sketch_linear,
                self.sketch_decoder_lstm,
                self.sketch_att_vec_linear,
                src_token_mask=None,
            )

            apply_rule_log_prob = torch.log_softmax(
                self.production_readout(att_t), dim=-1
            )

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                action_class = hyp.get_availableClass()
                if action_class in [
                    define_rule.Root1,
                    define_rule.Root,
                    define_rule.Sel,
                    define_rule.Filter,
                    define_rule.Sup,
                    define_rule.N,
                    define_rule.Order,
                ]:
                    possible_productions = self.grammar.get_production(action_class)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]
                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {
                            "action_type": action_class,
                            "prod_id": prod_id,
                            "score": prod_score,
                            "new_hyp_score": new_hyp_score,
                            "prev_hyp_id": hyp_id,
                        }
                        new_hyp_meta.append(meta_entry)
                else:
                    raise RuntimeError("No right action class")

            if not new_hyp_meta:
                break

            new_hyp_scores = torch.stack(
                [x["new_hyp_score"] for x in new_hyp_meta], dim=0
            )
            top_new_hyp_scores, meta_ids = torch.topk(
                new_hyp_scores,
                k=min(new_hyp_scores.size(0), beam_size - len(completed_beams)),
            )

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(
                top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()
            ):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry["prev_hyp_id"]
                prev_hyp = beams[prev_hyp_id]
                action_type_str = hyp_meta_entry["action_type"]
                prod_id = hyp_meta_entry["prod_id"]
                if prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(
                        list(action_type_str._init_grammar()).index(production)
                    )
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry["score"]
                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                beams = new_beams
                t += 1
            else:
                break

        # now get the sketch result
        completed_beams.sort(key=lambda hyp: -hyp.score)
        if len(completed_beams) == 0:
            return [[], []]

        sketch_actions = completed_beams[0].actions
        # sketch_actions = examples.sketch

        padding_sketch = self.padding_sketch(sketch_actions)

        batch_table_dict = batch.col_table_dict

        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=False)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.decode_max_time_step:
            hyp_num = len(beams)

            # expand value
            exp_src_encodings = src_encodings.expand(
                hyp_num, src_encodings.size(1), src_encodings.size(2)
            )
            exp_utterance_encodings_lf_linear = utterance_encodings_lf_linear.expand(
                hyp_num,
                utterance_encodings_lf_linear.size(1),
                utterance_encodings_lf_linear.size(2),
            )
            exp_table_embedding = table_embedding.expand(
                hyp_num, table_embedding.size(1), table_embedding.size(2)
            )

            exp_schema_embedding = schema_embedding.expand(
                hyp_num, schema_embedding.size(1), schema_embedding.size(2)
            )

            table_appear_mask = batch.table_appear_mask
            table_appear_mask = np.zeros(
                (hyp_num, table_appear_mask.shape[1]), dtype=np.float32
            )
            table_enable = np.zeros(shape=(hyp_num))
            for e_id, hyp in enumerate(beams):
                for act in hyp.actions:
                    if type(act) == define_rule.C:
                        table_appear_mask[e_id][act.id_c] = 1
                        table_enable[e_id] = act.id_c

            if t == 0:
                with torch.no_grad():
                    x = Variable(
                        self.new_tensor(1, self.lf_decoder_lstm.input_size).zero_()
                    )
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [
                        define_rule.Root1,
                        define_rule.Root,
                        define_rule.Sel,
                        define_rule.Filter,
                        define_rule.Sup,
                        define_rule.N,
                        define_rule.Order,
                    ]:

                        a_tm1_embed = self.production_embed.weight[
                            self.grammar.prod2id[action_tm1.production]
                        ]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, define_rule.C):
                        a_tm1_embed = self.column_rnn_input(
                            table_embedding[0, action_tm1.id_c]
                        )
                    elif isinstance(action_tm1, define_rule.T):
                        a_tm1_embed = self.column_rnn_input(
                            schema_embedding[0, action_tm1.id_c]
                        )
                    elif isinstance(action_tm1, define_rule.A):
                        a_tm1_embed = self.production_embed.weight[
                            self.grammar.prod2id[action_tm1.production]
                        ]
                    else:
                        raise ValueError("unknown action %s" % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[
                        self.grammar.type2id[type(action_tm)]
                    ]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(
                x,
                h_tm1,
                exp_src_encodings,
                exp_utterance_encodings_lf_linear,
                self.lf_decoder_lstm,
                self.lf_att_vec_linear,
                src_token_mask=None,
            )

            apply_rule_log_prob = torch.log_softmax(
                self.production_readout(att_t), dim=-1
            )

            table_appear_mask_val = torch.from_numpy(table_appear_mask)

            if self.is_cuda:
                table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = torch.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(
                    src_encodings=exp_table_embedding,
                    query_vec=att_t.unsqueeze(0),
                    src_token_mask=None,
                ) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=exp_table_embedding,
                    query_vec=att_t.unsqueeze(0),
                    src_token_mask=None,
                ) * (
                    1 - table_appear_mask_val
                ) * (
                    1 - gate
                )
                # weights = weights + self.col_attention_out(exp_embedding_differ).squeeze()
            else:
                weights = self.column_pointer_net(
                    src_encodings=exp_table_embedding,
                    query_vec=att_t.unsqueeze(0),
                    src_token_mask=batch.table_token_mask,
                )
            # weights.data.masked_fill_(exp_col_pred_mask, -float('inf'))

            column_selection_log_prob = torch.log_softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(
                src_encodings=exp_schema_embedding,
                query_vec=att_t.unsqueeze(0),
                src_token_mask=None,
            )
            # table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float("inf"))

            table_dict = [
                batch_table_dict[0][int(x)]
                for x_id, x in enumerate(table_enable.tolist())
            ]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float("inf"))

            table_weights = torch.log_softmax(table_weights, dim=-1)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                # TODO: should change this
                if type(padding_sketch[t]) == define_rule.A:
                    possible_productions = self.grammar.get_production(define_rule.A)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]

                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {
                            "action_type": define_rule.A,
                            "prod_id": prod_id,
                            "score": prod_score,
                            "new_hyp_score": new_hyp_score,
                            "prev_hyp_id": hyp_id,
                        }
                        new_hyp_meta.append(meta_entry)

                elif type(padding_sketch[t]) == define_rule.C:
                    for col_id, _ in enumerate(batch.table_sents[0]):
                        col_sel_score = column_selection_log_prob[hyp_id, col_id]
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {
                            "action_type": define_rule.C,
                            "col_id": col_id,
                            "score": col_sel_score,
                            "new_hyp_score": new_hyp_score,
                            "prev_hyp_id": hyp_id,
                        }
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t]) == define_rule.T:
                    for t_id, _ in enumerate(batch.table_names[0]):
                        t_sel_score = table_weights[hyp_id, t_id]
                        new_hyp_score = hyp.score + t_sel_score.data.cpu()

                        meta_entry = {
                            "action_type": define_rule.T,
                            "t_id": t_id,
                            "score": t_sel_score,
                            "new_hyp_score": new_hyp_score,
                            "prev_hyp_id": hyp_id,
                        }
                        new_hyp_meta.append(meta_entry)
                else:
                    prod_id = self.grammar.prod2id[padding_sketch[t].production]
                    new_hyp_score = hyp.score + torch.tensor(0.0)
                    meta_entry = {
                        "action_type": type(padding_sketch[t]),
                        "prod_id": prod_id,
                        "score": torch.tensor(0.0),
                        "new_hyp_score": new_hyp_score,
                        "prev_hyp_id": hyp_id,
                    }
                    new_hyp_meta.append(meta_entry)

            if not new_hyp_meta:
                break

            new_hyp_scores = torch.stack(
                [x["new_hyp_score"] for x in new_hyp_meta], dim=0
            )
            top_new_hyp_scores, meta_ids = torch.topk(
                new_hyp_scores,
                k=min(new_hyp_scores.size(0), beam_size - len(completed_beams)),
            )

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(
                top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()
            ):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry["prev_hyp_id"]
                prev_hyp = beams[prev_hyp_id]

                action_type_str = hyp_meta_entry["action_type"]
                if "prod_id" in hyp_meta_entry:
                    prod_id = hyp_meta_entry["prod_id"]
                if action_type_str == define_rule.C:
                    col_id = hyp_meta_entry["col_id"]
                    action = define_rule.C(col_id)
                elif action_type_str == define_rule.T:
                    t_id = hyp_meta_entry["t_id"]
                    action = define_rule.T(t_id)
                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(
                        list(action_type_str._init_grammar()).index(production)
                    )
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry["score"]

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]

                beams = new_beams
                t += 1
            else:
                break

        completed_beams.sort(key=lambda hyp: -hyp.score)

        """
        total_loss = [-sketch_prob_var[idx] + -lf_prob_var[idx] for idx in range(len(sketch_prob_var))]

        return {"total": total_loss, "sketch": -sketch_prob_var, "detail": -lf_prob_var}, completed_beams
        """


        return [completed_beams, sketch_actions]

    def padding_sketch(self, sketch):
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == define_rule.N:
                for _ in range(action.id_c + 1):
                    padding_result.append(define_rule.A(0))
                    padding_result.append(define_rule.C(0))
                    padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Filter and "A" in action.production:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Order or type(action) == define_rule.Sup:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))

        return padding_result

    def step(
        self,
        x,
        h_tm1,
        src_encodings,
        src_encodings_att_linear,
        decoder,
        attention_func,
        src_token_mask=None,
        return_att_weight=False,
    ):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = decoder(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(
            h_t, src_encodings, src_encodings_att_linear, mask=src_token_mask
        )

        att_t = torch.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def get_accuracy(self, pred_semql, gold_semql):
        assert len(pred_semql) == len(gold_semql), "pred_len:{} gold_len:{}".format(
            len(pred_semql), len(gold_semql)
        )

        acc_list = []
        acc = {"total": 0.0, "sketch": 0.0, "detail": 0.0}
        for idx in range(len(gold_semql)):
            pred_sketch = []
            pred_detail = []
            for item in pred_semql[idx]:
                if item.is_sketch:
                    pred_sketch += [str(item)]
                else:
                    pred_detail += [str(item)]

            gold_sketch = []
            gold_detail = []
            for item in gold_semql[idx]:
                if item.is_sketch:
                    gold_sketch += [str(item)]
                else:
                    gold_detail += [str(item)]

            # Sketch acc
            sketch_is_correct = pred_sketch == gold_sketch
            acc["sketch"] += sketch_is_correct

            # Detail acc
            detail_is_correct = pred_detail == gold_detail
            acc["detail"] += detail_is_correct

            # Total acc
            is_correct = sketch_is_correct and detail_is_correct
            acc["total"] += is_correct
            acc_list += [is_correct]

        # To percentage
        for key in acc.keys():
            acc[key] /= len(gold_semql)

        return acc, acc_list
