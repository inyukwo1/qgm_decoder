from service import End2End
import pickle
from ours.preprocess.data_process import process_data_one_entry
from ours.src import args as arg, utils
from ours.src.models.model import IRNet
from ours.src.utils import (
    process,
    schema_linking,
    get_col_table_dict,
    get_table_colNames,
)
from ours.src.rule.sem_utils import (
    alter_column0_one_entry,
    alter_inter_one_entry,
    alter_not_in_one_entry,
)
from ours.sem2SQL import transform
from ours.src.dataset import Example
from ours.src.rule import semQL
import torch
from pattern.en import lemma
import json
import re
import sqlite3
import time
import copy


class End2EndOurs(End2End):
    def prepare_model(self, dataset):
        model_path = "test_models/ours_{}.model".format(dataset)
        with open("ours/preprocess/conceptNet/english_RelatedTo.pkl", "rb") as f:
            self.english_RelatedTo = pickle.load(f)

        with open("ours/preprocess/conceptNet/english_IsA.pkl", "rb") as f:
            self.english_IsA = pickle.load(f)

        arg_parser = arg.init_arg_parser()
        args = arg.init_config(arg_parser)

        grammar = semQL.Grammar()
        _, _, val_sql_data, val_table_data = utils.load_dataset(
            "./data/{}".format(dataset), use_eval_only=True
        )
        self.val_table_data = val_table_data

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

        self.model.load_state_dict(pretrained_modeled)

        self.model.eval()

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
        # print("lang elliot" in self.db_values["imdb"]["name"])

    def run_model(self, db_id, nl_string):
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
        print("Start preprocessing.. {}".format(time.strftime("%Y%m%d-%H%M%S")))
        process_data_one_entry(
            entry, self.english_RelatedTo, self.english_IsA, self.db_values
        )
        print("End preprocessing.. {}".format(time.strftime("%Y%m%d-%H%M%S")))

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
        print(
            "End schema linking and start model running.. {}".format(
                time.strftime("%Y%m%d-%H%M%S")
            )
        )

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
        sql_values = self._find_values(
            simple_json["question_arg"],
            simple_json["question_arg_type"],
            simple_json["origin_question_toks_for_value"],
            simple_json["mapper"],
        )
        print(sql_values)
        sql_with_value = self._exchange_values(sql, sql_values, " 1")
        print("End post processing {}".format(time.strftime("%Y%m%d-%H%M%S")))
        return sql_with_value, None, entry["question_toks"]

    def _is_number_tryexcept(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _find_values(self, question_arg, question_arg_type, question_origin, mapper):
        values = []
        flag_double_q = False
        flag_double_q_for_schema = False
        cur_val = []
        flag_single_q = False
        flag_single_q_for_schema = False
        flag_upper = False
        cur_upper_val = []
        for idx, (token, tag) in enumerate(zip(question_arg, question_arg_type)):
            if idx == 0:
                continue
            start_idx = mapper[idx][0]
            end_idx = mapper[idx][1]
            if len(token) == 0:
                continue
            if flag_double_q:
                if '"' not in token[0]:
                    cur_val.append(" ".join(question_origin[start_idx:end_idx]))
                    if tag[0] in ("table", "col"):
                        flag_double_q_for_schema = True
                    continue
            if flag_single_q:
                if "'" not in token[0]:
                    #                      for i, t in enumerate(token):
                    #                          idx = first_substring( question_origin[start_idx:end_idx], t )
                    #                          if idx != -1:
                    #                              token[i]=question_origin[idx]
                    cur_val.append(" ".join(question_origin[start_idx:end_idx]))
                    if tag[0] in ("table", "col"):
                        flag_single_q_for_schema = True
                    continue

            if flag_upper:
                # If Jason 'Two ... separate
                if (
                    len(question_origin[start_idx]) > 0
                    and question_origin[start_idx][0].isupper()
                    and tag[0] not in ("col", "table")
                ):
                    cur_upper_val.append(" ".join(question_origin[start_idx:end_idx]))
                    continue
                else:
                    values.append(" ".join(cur_upper_val))
                    cur_upper_val = []
                    flag_upper = False

            def is_year(tok):
                if (
                    len(str(tok)) == 4
                    and str(tok).isdigit()
                    and 15 < int(str(tok)[:2]) < 22
                ):
                    return True

            is_inserted_already = False
            if (
                len(token) == 1
                and is_year(token[0])
                and self._is_number_tryexcept(question_origin[start_idx])
            ):
                is_inserted_already = True
                values.append(question_origin[start_idx])

            if '"' in token[0]:
                if flag_double_q:
                    is_inserted_already = True
                    flag_double_q = False
                    if not flag_double_q_for_schema:
                        values.append(" ".join(cur_val))
                    cur_val = []
                    flag_double_q_for_schema = False
                elif len(token[0]) == 1:
                    is_inserted_already = True
                    flag_double_q = True
            elif "'" in token[0]:
                if flag_single_q:
                    is_inserted_already = True
                    flag_single_q = False
                    if not flag_single_q_for_schema:
                        values.append(" ".join(cur_val))
                    cur_val = []
                    flag_single_q_for_schema = False
                elif len(token[0]) == 1:
                    is_inserted_already = True
                    flag_single_q = True

            if (
                (not is_inserted_already)
                and len(question_origin[start_idx]) > 0
                and question_origin[start_idx][0].isupper()
                and start_idx != 0
            ):
                if tag[0] not in ("col", "table"):
                    is_inserted_already = True
                    flag_upper = True
                    cur_upper_val.append(" ".join(question_origin[start_idx:end_idx]))
            if (
                (not is_inserted_already)
                and tag[0] in ("value", "*", "db")
                and token[0] != "the"
            ):
                is_inserted_already = True
                values.append(" ".join(question_origin[start_idx:end_idx]))
        return values

    def _exchange_values(self, sql, sql_values, value_tok):
        sql = sql.replace(value_tok, " 1")
        cur_index = sql.find(" 1")
        sql_with_value = ""
        before_index = 0
        values_index = 0
        while cur_index != -1 and values_index < len(sql_values):
            sql_with_value = sql_with_value + sql[before_index:cur_index]
            if sql[cur_index - 1] in ("=", ">", "<"):
                cur_value = sql_values[values_index]
                values_index = values_index + 1
                if not self._is_number_tryexcept(cur_value):
                    cur_value = '"' + cur_value + '"'
                sql_with_value = sql_with_value + " " + cur_value
            elif cur_index - 3 > 0 and sql[cur_index - 4 : cur_index] in ("like"):
                cur_value = "%" + sql_values[values_index] + "%"
                values_index = values_index + 1
                if not self._is_number_tryexcept(cur_value):
                    cur_value = '"' + cur_value + '"'
                sql_with_value = sql_with_value + " " + cur_value
            elif cur_index - 6 > 0 and sql[cur_index - 7 : cur_index] in ("between"):
                if values_index + 1 < len(sql_values):
                    cur_value1 = sql_values[values_index]
                    values_index = values_index + 1
                    cur_value2 = sql_values[values_index]
                    values_index = values_index + 1
                else:
                    cur_value1 = sql_values[values_index]
                    cur_value2 = sql_values[values_index]
                    values_index = values_index + 1
                if not self._is_number_tryexcept(cur_value1):
                    cur_value1 = "1"
                if not self._is_number_tryexcept(cur_value2):
                    cur_value2 = "2"
                sql_with_value = (
                    sql_with_value + " " + cur_value1 + " AND " + cur_value2
                )
                cur_index = cur_index + 6
            else:
                sql_with_value = sql_with_value + sql[cur_index : cur_index + 2]
            before_index = cur_index + 2
            cur_index = sql.find(" 1", cur_index + 1)
        sql_with_value = sql_with_value + sql[before_index:]
        print(sql_with_value)
        return sql_with_value

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
            entry, self.english_RelatedTo, self.english_IsA, self.db_values
        )
        print(entry["question_arg"])
        print(entry["question_arg_type"])
        print(entry["mapper"])
        sql_values = self._find_values(
            entry["question_arg"],
            entry["question_arg_type"],
            entry["origin_question_toks_for_value"],
            entry["mapper"],
        )
        return self._exchange_values(sql, sql_values, value_tok)
