from service import End2End
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from ours.preprocess.data_process import process_data_one_entry
from ours.src import args as arg, utils
from ours.src.models.model import IRNet
from ours.src.utils import (
    process,
    schema_linking,
    get_col_table_dict,
    get_table_colNames,
    to_batch_seq,
)
from ours.my_layerconductance import MyLayerConductance
from ours.src.rule.sem_utils import (
    alter_column0_one_entry,
    alter_inter_one_entry,
    alter_not_in_one_entry,
)
from value_prediction_utils import is_number_tryexcept, exchange_values, find_values

from ours.sem2SQL import transform
from ours.src.dataset import Example, Batch
from ours.src.rule import semQL
import torch
from pattern.en import lemma
import json
import re
import sqlite3
import time
import copy
import seaborn as sns
import numpy as np

from irnet.preprocess.sql2SemQL import SemQLConverter
from captum.attr import (
    LayerConductance,
    IntegratedGradients,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
from captum.attr import visualization as viz
from universal_utils import (
    captum_vis_to_html,
    tab_col_attribution_to_html,
    src_attribution_to_html,
)


class End2EndOurs(End2End):
    english_RelatedTo = None
    english_IsA = None

    def __init__(self):
        if End2EndOurs.english_RelatedTo is None:
            with open("ours/preprocess/conceptNet/english_RelatedTo.pkl", "rb") as f:
                End2EndOurs.english_RelatedTo = pickle.load(f)
        if End2EndOurs.english_IsA is None:
            with open("ours/preprocess/conceptNet/english_IsA.pkl", "rb") as f:
                End2EndOurs.english_IsA = pickle.load(f)

    def captum_mode(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def load_model(self, dataset):
        arg_parser = arg.init_arg_parser()
        args = arg.init_config(arg_parser)

        grammar = semQL.Grammar()
        model_path = "test_models/ours_{}.model".format(dataset)
        model_path = "test_models/spider_service_bert_basic.model"

        self.model = IRNet(args, grammar)

        if args.cuda:
            self.model.cuda()

        print("load pretrained model from %s" % (model_path))
        pretrained_model = torch.load(
            model_path, map_location=lambda storage, loc: storage
        )
        pretrained_modeled = copy.deepcopy(pretrained_model)
        for k in pretrained_model.keys():
            if k not in self.model.state_dict().keys():
                del pretrained_modeled[k]

        self.model.load_state_dict(pretrained_modeled, strict=False)

        self.model.eval()

    def prepare_model(self, dataset):
        _, _, val_sql_data, val_table_data = utils.load_dataset(
            "./data/{}".format(dataset), use_eval_only=True
        )
        self.val_table_data = val_table_data

        self.load_model(dataset)

        self.db_values = dict()

        schema_dict = dict()
        for one_schema in val_table_data:
            schema_dict[one_schema] = val_table_data[one_schema]
            schema_dict[one_schema]["only_cnames"] = [
                c_name.lower()
                for tid, c_name in val_table_data[one_schema]["column_names_original"]
            ]

        for db_id in val_table_data:
            if db_id not in self.db_values:
                schema_json = schema_dict[db_id]
                primary_foreigns = set()
                if not db_id == "imdb":
                    continue

                for f, p in schema_json["foreign_keys"]:
                    primary_foreigns.add(f)
                    primary_foreigns.add(p)

                conn = sqlite3.connect(
                    "data/{}/database/{}/{}.sqlite".format(dataset, db_id, db_id)
                )
                # conn.text_factory = bytes
                cursor = conn.cursor()

                schema = {}

                # fetch table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [
                    str(table[0].lower())
                    for table in cursor.fetchall()
                    if table[0].lower() not in ("sqlite_sequence")
                ]

                # fetch table info
                for table in tables:
                    cursor.execute("PRAGMA table_info({})".format(table))
                    schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
                col_value_set = dict()
                for table in tables:
                    name_list = [
                        name for _, name in val_table_data[db_id]["column_names"]
                    ]
                    for col in schema[table]:
                        print("table: {} col: {}".format(table, col))
                        # if not (col == "name" and table == "writer"):
                        #     continue
                        col_idx = schema_json["only_cnames"].index(col)
                        if (
                            col_idx in primary_foreigns
                            and schema_json["column_types"][col_idx] == "number"
                        ):
                            continue
                        sql = 'SELECT DISTINCT "{}" FROM "{}"'.format(col, table)
                        cursor.execute(sql)
                        col = name_list[col_idx]
                        print("new col: {}".format(col))
                        value_set = set()
                        while True:
                            try:
                                val = cursor.fetchone()
                            except:
                                continue
                            if val is None:
                                break
                            if isinstance(val[0], str):
                                value_set.add(str(val[0].lower()))
                                value_set.add(lemma(str(val[0].lower())))
                        if col in col_value_set:
                            col_value_set[col] |= value_set
                        else:
                            col_value_set[col] = value_set
                self.db_values[db_id] = col_value_set
        print("lang elliot" in self.db_values["imdb"]["name"])
        self.semql_parser = SemQLConverter("data/{}/tables.json".format(dataset))

    def prepare_table(self, db_id, table=None):
        if not table:
            table = self.val_table_data[db_id]
        tmp_col = []
        for cc in [x[1] for x in table["column_names"]]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table["col_set"] = tmp_col
        table["schema_content"] = [col[1] for col in table["column_names"]]
        table["col_table"] = [col[0] for col in table["column_names"]]
        return table

    def prepare_entry(self, table, db_id, nl_string):
        entry = {}
        entry["db_id"] = db_id
        entry["question"] = nl_string
        entry["question_toks"] = re.findall(
            r"[^,.:;\"`?! ]+|[,.:;\"?!]", nl_string.replace("'", " '")
        )
        entry["names"] = table["schema_content"]
        entry["table_names"] = table["table_names"]
        entry["col_set"] = table["col_set"]
        entry["col_table"] = table["col_table"]
        keys = {}
        for kv in table["foreign_keys"]:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in table["primary_keys"]:
            keys[id_k] = id_k
        entry["keys"] = keys
        return entry

    def prepare_example(self, entry, table, sql):
        process_dict = process(entry, table)

        for c_id, col_ in enumerate(process_dict["col_set_iter"]):
            for q_id, ori in enumerate(process_dict["q_iter_small"]):
                if ori in col_:
                    process_dict["col_set_type"][c_id][0] += 1

        for t_id, tab_ in enumerate(process_dict["table_names"]):
            for q_id, ori in enumerate(process_dict["q_iter_small"]):
                if ori in tab_:
                    process_dict["tab_set_type"][t_id][0] += 1

        schema_linking(
            process_dict["question_arg"],
            process_dict["question_arg_type"],
            process_dict["one_hot_type"],
            process_dict["col_set_type"],
            process_dict["col_set_iter"],
            process_dict["tab_set_type"],
            process_dict["table_names"],
            entry,
        )

        col_table_dict = get_col_table_dict(
            process_dict["tab_cols"], process_dict["tab_ids"], entry
        )
        table_col_name = get_table_colNames(
            process_dict["tab_ids"], process_dict["col_iter"]
        )

        process_dict["col_set_iter"][0] = ["count", "number", "many"]

        rule_label = None

        example = Example(
            src_sent=process_dict["question_arg"],
            col_num=len(process_dict["col_set_iter"]),
            vis_seq=(entry["question"], process_dict["col_set_iter"], None),
            tab_cols=process_dict["col_set_iter"],
            tab_iter=process_dict["tab_set_iter"],
            sql=None,
            one_hot_type=process_dict["one_hot_type"],
            col_hot_type=process_dict["col_set_type"],
            tab_hot_type=process_dict["tab_set_type"],
            table_names=process_dict["table_names"],
            table_len=len(process_dict["table_names"]),
            col_table_dict=col_table_dict,
            cols=process_dict["tab_cols"],
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            tokenized_src_sent=process_dict["col_set_type"],
            tgt_actions=rule_label,
        )
        example.sql_json = copy.deepcopy(entry)
        example.db_id = entry["db_id"]
        return example

    def run_model(self, db_id, nl_string, table=None):
        self.eval()
        table = self.prepare_table(db_id, table)
        entry = self.prepare_entry(table, db_id, nl_string)

        print("Start preprocessing.. {}".format(time.strftime("%Y%m%d-%H%M%S")))
        process_data_one_entry(
            entry,
            End2EndOurs.english_RelatedTo,
            End2EndOurs.english_IsA,
            self.db_values,
        )
        print("End preprocessing.. {}".format(time.strftime("%Y%m%d-%H%M%S")))

        print(
            "End schema linking and start model running.. {}".format(
                time.strftime("%Y%m%d-%H%M%S")
            )
        )
        example = self.prepare_example(entry, table, None)
        results_all = self.model.parse(example, beam_size=5)
        print(
            "End model running and start postprocessing.. {}".format(
                time.strftime("%Y%m%d-%H%M%S")
            )
        )
        results = results_all[0]
        list_preds = []
        # list_actions = []
        # list_attentions = []
        try:
            pred = " ".join([str(x) for x in results[0].actions])
            for x in results:
                list_preds.append(" ".join(str(x.actions)))
        except Exception as e:
            pred = ""

        simple_json = example.sql_json["pre_sql"]
        simple_json["sketch_result"] = " ".join(str(x) for x in results_all[1])
        simple_json["model_result"] = pred
        simple_json["query"] = None

        print(simple_json)

        alter_not_in_one_entry(simple_json, table)
        alter_inter_one_entry(simple_json)
        alter_column0_one_entry(simple_json)

        try:
            result = transform(simple_json, table)
        except Exception as e:
            result = transform(
                simple_json, table, origin="Root1(3) Root(5) Sel(0) N(0) A(3) C(0) T(0)"
            )
        sql = result[0]

        print(
            "question_arg: {} \n question_arg_type: {} \n original_question: {} \n mapper: {}".format(
                simple_json["question_arg"],
                simple_json["question_arg_type"],
                simple_json["origin_question_toks_for_value"],
                simple_json["mapper"],
            )
        )
        sql_values = find_values(
            simple_json["question_arg"],
            simple_json["question_arg_type"],
            simple_json["origin_question_toks_for_value"],
            simple_json["mapper"],
        )
        print(sql_values)
        sql_with_value = exchange_values(sql, sql_values, " 1")
        print("End post processing {}".format(time.strftime("%Y%m%d-%H%M%S")))
        return sql_with_value, None, entry["question_toks"]

    def run_captum(self, db_id, nl_string, gold_sql, table=None):
        self.captum_mode()
        model = self.model
        table = self.prepare_table(db_id, table)
        entry = self.prepare_entry(table, db_id, nl_string)
        process_data_one_entry(
            entry,
            End2EndOurs.english_RelatedTo,
            End2EndOurs.english_IsA,
            self.db_values,
        )
        entry["query"] = gold_sql
        entry = self.semql_parser.parse(db_id, gold_sql, entry, table)

        def to_ref_examples(examples):
            new_examples = [copy.copy(example) for example in examples]
            for example in new_examples:
                example.src_sent = [
                    ["[unk]"] * len(words) for words in example.src_sent
                ]
                example.tab_cols = [
                    ["[unk]"] * len(words) for words in example.tab_cols
                ]
                example.table_names = [
                    ["[unk]"] * len(words) for words in example.table_names
                ]
            return new_examples

        def summarize_attributions(
            attributions,
            word_start_end_batch,
            col_start_end_batch,
            tab_start_end_batch,
        ):
            attributions = attributions.sum(dim=-1).squeeze(0)
            # attributions = attributions

            src_attributions = torch.stack(
                [attributions[st:ed].sum() for st, ed in word_start_end_batch[0]]
            )
            col_attributions = torch.stack(
                [attributions[st:ed].sum() for st, ed in col_start_end_batch[0]]
            )
            tab_attributions = torch.stack(
                [attributions[st:ed].sum() for st, ed in tab_start_end_batch[0]]
            )
            src_attributions = src_attributions / torch.norm(attributions)
            col_attributions = col_attributions / torch.norm(attributions)
            tab_attributions = tab_attributions / torch.norm(attributions)

            return src_attributions, col_attributions, tab_attributions

        def summarize_conductance_attributions(attributions):
            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            return attributions

        def forward_captum(
            input_embedding,
            word_start_end_batch,
            col_start_end_batch,
            tab_start_end_batch,
            examples,
        ):
            examples = examples * len(input_embedding)
            word_start_end_batch = word_start_end_batch * len(input_embedding)
            col_start_end_batch = col_start_end_batch * len(input_embedding)
            tab_start_end_batch = tab_start_end_batch * len(input_embedding)
            sketch_score, lf_score = model.forward(
                examples,
                input_embedding,
                word_start_end_batch,
                col_start_end_batch,
                tab_start_end_batch,
            )
            return (sketch_score + lf_score) * 10

        lig = IntegratedGradients(forward_captum)
        examples = to_batch_seq([entry], self.val_table_data, [0], 0, 1, True, table)
        batch = Batch(examples, model.grammar, cuda=model.args.cuda)
        (
            input_embedding,
            word_start_end_batch,
            col_start_end_batch,
            tab_start_end_batch,
            tokenized_questions,
        ) = model.transformer_embed(batch)
        print(list(input_embedding.size()))
        attributions, delta = lig.attribute(
            inputs=input_embedding,
            baselines=torch.zeros_like(input_embedding),
            additional_forward_args=(
                word_start_end_batch,
                col_start_end_batch,
                tab_start_end_batch,
                examples,
            ),
            return_convergence_delta=True,
        )

        src_attributions, col_attributions, tab_attributions = summarize_attributions(
            attributions,
            word_start_end_batch,
            col_start_end_batch,
            tab_start_end_batch,
        )

        layer_attrs = []
        bert_config = self.model.transformer_encoder.config
        for i in range(bert_config.num_hidden_layers):
            lc = LayerConductance(
                forward_captum, self.model.transformer_encoder.encoder.layer[i]
            )
            layer_attributions, delta = lc.attribute(
                inputs=input_embedding,
                baselines=torch.zeros_like(input_embedding),
                additional_forward_args=(
                    word_start_end_batch,
                    col_start_end_batch,
                    tab_start_end_batch,
                    examples,
                ),
                return_convergence_delta=True,
            )
            layer_attrs.append(
                summarize_conductance_attributions(layer_attributions[0])
                .cpu()
                .detach()
                .tolist()
            )
        yticklabels = list(range(1, bert_config.num_hidden_layers + 1))
        xticklabels = [
            token if token.startswith("[") else token
            for token in tokenized_questions[0]
        ]
        fig, ax = plt.subplots(figsize=(15, 5))
        ax = sns.heatmap(
            np.array(layer_attrs),
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            linewidth=0.2,
        )

        layer_attrs = []
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        plt.savefig("fig1.png", dpi=300)  # TODO fix

        lc = MyLayerConductance(
            forward_captum, [self.model.sketch_decoder_lstm, self.model.lf_decoder_lstm]
        )
        layer_attributions, delta = lc.attribute(
            inputs=input_embedding,
            baselines=torch.zeros_like(input_embedding),
            additional_forward_args=(
                word_start_end_batch,
                col_start_end_batch,
                tab_start_end_batch,
                examples,
            ),
            return_convergence_delta=True,
        )
        layer_attrs.append(
            summarize_conductance_attributions(layer_attributions[0])
            .cpu()
            .detach()
            .tolist()
        )
        yticklabels = list(range(1, 2))
        skectch_rule_labels = [
            w
            for w in entry["rule_label"].split(" ")
            if not (w.startsWith("C") and w.startsWith("T") and w.startsWith("A"))
        ]
        xticklabels = skectch_rule_labels + entry["rule_label"].split(" ")
        fig, ax = plt.subplots(figsize=(15, 5))
        ax = sns.heatmap(
            np.array(layer_attrs),
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            linewidth=0.2,
        )
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        plt.savefig("fig2.png", dpi=300)  # TODO fix

        return (
            "<b> Attribution of: </b> words of the input sentence </br> <b> For the:</b> ground truth query </br> </br>"
            + src_attribution_to_html(examples[0].src_sent, src_attributions),
            "<b> Attribution of: </b> words of the schema </br> <b> For the:</b> ground truth query </br>"
            + tab_col_attribution_to_html(
                examples[0], col_attributions, tab_attributions
            ),
            # "<b> Conductance of: </b> the transformer encoder </br> <b> For the:</b> ground truth query </br>"
            # + '<img src="http://141.223.199.148:4001/image" >',
        )

    def value_predictor(self, db_id, sql, question, value_tok):
        table = self.val_table_data[db_id]
        tmp_col = []
        for cc in [x[1] for x in table["column_names"]]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table["col_set"] = tmp_col
        db_name = table["db_id"]
        table["schema_content"] = [col[1] for col in table["column_names"]]
        table["col_table"] = [col[0] for col in table["column_names"]]

        entry = {}
        entry["db_id"] = db_id
        entry["question"] = question
        entry["question_toks"] = re.findall(
            r"[^,.:;\"`?! ]+|[,.:;\"?!]", question.replace("'", " '")
        )
        entry["names"] = table["schema_content"]
        entry["table_names"] = table["table_names"]
        entry["col_set"] = table["col_set"]
        entry["col_table"] = table["col_table"]
        keys = {}
        for kv in table["foreign_keys"]:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in table["primary_keys"]:
            keys[id_k] = id_k
        entry["keys"] = keys
        print("Start preprocessing.. {}".format(time.strftime("%Y%m%d-%H%M%S")))
        process_data_one_entry(
            entry,
            End2EndOurs.english_RelatedTo,
            End2EndOurs.english_IsA,
            self.db_values,
        )
        print(entry["question_arg"])
        print(entry["question_arg_type"])
        print(entry["mapper"])
        sql_values = find_values(
            entry["question_arg"],
            entry["question_arg_type"],
            entry["origin_question_toks_for_value"],
            entry["mapper"],
        )
        return exchange_values(sql, sql_values, value_tok)


if __name__ == "__main__":
    end2end = End2EndOurs()
    end2end.prepare_model("spider")
    end2end.captum_mode()
    new_table = {
        "column_names": [
            [-1, "*"],
            [0, "mid"],
            [0, "title"],
            [0, "release year"],
            [1, "msid"],
            [1, "aid"],
            [2, "aid"],
            [2, "name"],
            [2, "birth year"],
            [3, "sid"],
            [3, "title"],
            [3, "release year"],
        ],
        "column_names_original": [
            [-1, "*"],
            [0, "mid"],
            [0, "title"],
            [0, "release_year"],
            [1, "msid"],
            [1, "aid"],
            [2, "aid"],
            [2, "name"],
            [2, "birth_year"],
            [3, "sid"],
            [3, "title"],
            [3, "release_year"],
        ],
        "column_types": [
            "text",
            "number",
            "text",
            "text",
            "number",
            "number",
            "number",
            "text",
            "text",
            "number",
            "text",
            "text",
        ],
        "db_id": "",
        "foreign_keys": [[4, 1], [5, 6], [4, 9]],
        "primary_keys": [1, 6, 9],
        "table_names": ["movie", "cast", "actor", "tv series"],
        "table_names_original": ["movie", "cast", "actor", "tv_series"],
        "only_cnames": [
            "*",
            "mid",
            "title",
            "release year",
            "msid",
            "aid",
            "aid",
            "name",
            "birth year",
            "sid",
            "title",
            "release year",
        ],
    }
    nlq = "How many performers are there?"
    sql, _, _ = end2end.run_model("imdb", nlq, new_table)
    print(sql)
    html_src, html_schema, _ = end2end.run_captum(
        "imdb", nlq, "SELECT count(*) FROM actor", new_table,
    )
    print(html_src)
    print(html_schema)
