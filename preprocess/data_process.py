# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : data_process.py
# @Software: PyCharm
"""
import json
import argparse
import nltk
import os
import pickle
import sqlite3
from tqdm import tqdm
from utils import (
    symbol_filter,
    re_lemma,
    fully_part_header,
    group_header,
    partial_header,
    num2year,
    group_symbol,
    group_values,
    group_digital,
    group_db,
    partial_matches,
    value_match,
    db_match
)
from utils import AGG, wordnet_lemmatizer
from utils import load_dataSets
import re


SKIP_WORDS = ["he", "the", "has", "have"]


def process_datas(datas, args, dataset_name):
    """
    :param datas:
    :param args:
    :return:
    """
    with open(os.path.join(args.conceptNet, "english_RelatedTo.pkl"), "rb") as f:
        english_RelatedTo = pickle.load(f)

    with open(os.path.join(args.conceptNet, "english_IsA.pkl"), "rb") as f:
        english_IsA = pickle.load(f)

    db_values = dict()

    with open(args.table_path) as f:
        schema_tables = json.load(f)
    schema_dict = dict()
    for one_schema in schema_tables:
        schema_dict[one_schema["db_id"]] = one_schema
        schema_dict[one_schema["db_id"]]["only_cnames"] = [
            c_name.lower() for tid, c_name in one_schema["column_names_original"]
        ]
    # copy of the origin question_toks
    # for d in datas:
    #     if "origin_question_toks" not in d:
    #         d["origin_question_toks"] = d["question_toks"]

    for idx in tqdm(range(len(datas))):
        entry = datas[idx]
        db_id = entry["db_id"]
        if db_id not in db_values:
            schema_json = schema_dict[db_id]
            primary_foreigns = set()
            for f, p in schema_json["foreign_keys"]:
                primary_foreigns.add(f)
                primary_foreigns.add(p)

            conn = sqlite3.connect(
                "../data/{}/database/{}/{}.sqlite".format(dataset_name, db_id, db_id)
            )
            # conn.text_factory = bytes
            cursor = conn.cursor()

            schema = {}

            # fetch table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [str(table[0].lower()) for table in cursor.fetchall()]

            # fetch table info
            for table in tables:
                cursor.execute("PRAGMA table_info({})".format(table))
                schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
            col_value_set = dict()
            for table in tables:
                for col in schema[table]:
                    col_idx = schema_json["only_cnames"].index(col)
                    if (
                        col_idx in primary_foreigns
                        and schema_json["column_types"][col_idx] == "number"
                    ):
                        continue
                    cursor.execute('SELECT "{}" FROM "{}"'.format(col, table))
                    col = entry["names"][col_idx]
                    value_set = set()
                    try:
                        for val in cursor.fetchall():
                            if isinstance(val[0], str):
                                val_str = val[0].lower()
                                value_set.add(val_str)
                                if val_str not in SKIP_WORDS:
                                    val_str = wordnet_lemmatizer.lemmatize(val_str)
                                value_set.add(val_str)
                            elif isinstance(val[0], int) or isinstance(val[0], float):
                                continue
                            else:
                                print(
                                    "check this out: db:{} tab:{} col:{} val:{}".format(
                                        db_id, table, col, val
                                    )
                                )
                    except:
                        print("bad utf-8?")
                    if col in col_value_set:
                        col_value_set[col] |= value_set
                    else:
                        col_value_set[col] = value_set
            db_values[db_id] = col_value_set

        # entry["question_toks"] = symbol_filter(entry["question_toks"])
        origin_question_toks = [
            x.lower()
            for x in re.findall(
                r"[^,.():;\"`?! ]+|[,.():;\"?!]", entry["question"].replace("'", " ' ")
            )
        ]
        question_toks = [
            x if x in SKIP_WORDS else wordnet_lemmatizer.lemmatize(x)
            for x in origin_question_toks
        ]

        entry["question_toks"] = origin_question_toks
        entry["origin_question_toks"] = entry["question_toks"]

        table_names = []
        table_names_pattern = []

        for y in entry["table_names"]:
            x = [
                x if x in SKIP_WORDS else wordnet_lemmatizer.lemmatize(x.lower())
                for x in y.split(" ")
            ]
            table_names.append(" ".join(x))
            x = [re_lemma(x.lower()) for x in y.split(" ")]
            table_names_pattern.append(" ".join(x))

        header_toks = []
        header_toks_list = []

        header_toks_pattern = []
        header_toks_list_pattern = []

        for y in entry["col_set"]:
            x = [
                x if x in SKIP_WORDS else wordnet_lemmatizer.lemmatize(x.lower())
                for x in y.split(" ")
            ]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

            x = [re_lemma(x.lower()) for x in y.split(" ")]
            header_toks_pattern.append(" ".join(x))
            header_toks_list_pattern.append(x)

        num_toks = len(question_toks)
        idx = 0
        tok_concol = []
        type_concol = []
        nltk_result = nltk.pos_tag(question_toks)
        while idx < num_toks:
            for endIdx in reversed(range(idx + 1, num_toks + 1)):
                sub_toks = " ".join(question_toks[idx:endIdx])
                sub_toks_list = question_toks[idx:endIdx]
                original_sub_toks = " ".join(origin_question_toks[idx:endIdx])
                original_sub_toks_list = origin_question_toks[idx:endIdx]
                sub_tok_types = []
                partial_col_match = partial_matches(sub_toks_list, header_toks)
                partial_tab_match = partial_matches(sub_toks_list, table_names)

                if sub_toks in header_toks:
                    sub_tok_types.append(["col", header_toks.index(sub_toks)])
                elif partial_col_match:
                    sub_tok_types.append(["col_part", partial_col_match])

                if sub_toks in table_names:
                    sub_tok_types.append(["table", table_names.index(sub_toks)])
                elif partial_tab_match:
                    sub_tok_types.append(["table_part", partial_tab_match])

                # check for aggregation
                if sub_toks in AGG:
                    sub_tok_types.append(["agg"])

                def get_concept_result(toks, graph):
                    for begin_id in range(0, len(toks)):
                        for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                            tmp_query = "_".join(toks[begin_id:r_ind])
                            if tmp_query in graph:
                                mi = graph[tmp_query]
                                for col in entry["col_set"]:
                                    if col in mi:
                                        return col
                if idx > 0 and endIdx < len(question_toks) - 1 and \
                        question_toks[idx - 1] == "'" and question_toks[endIdx] == "'":
                    pro_result = get_concept_result(sub_toks_list, english_IsA)
                    if pro_result is None:
                        pro_result = get_concept_result(sub_toks_list, english_RelatedTo)
                    if pro_result is None:
                        pro_result = "NONE"
                    sub_tok_types.append([pro_result])

                if value_match(original_sub_toks_list) and (len(original_sub_toks_list) > 1 or question_toks[idx - 1] not in ["?", "."]):
                    tmp_toks = [
                        x if x in SKIP_WORDS else wordnet_lemmatizer.lemmatize(x)
                        for x in sub_toks_list
                    ]
                    pro_result = get_concept_result(tmp_toks, english_IsA)
                    if pro_result is None:
                        pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                    if pro_result is None:
                        pro_result = "NONE"
                    sub_tok_types.append([pro_result])

                db_cols = db_match(original_sub_toks, db_values[db_id])

                if db_cols and not (endIdx == idx + 1 and (
                    nltk_result[idx][1] == "VBZ"
                    or nltk_result[idx][1] == "IN"
                    or nltk_result[idx][1] == "CC"
                    or nltk_result[idx][1] == "DT"
                    or origin_question_toks[idx] == "'"
                    or (nltk_result[idx][1] == "VBP" and origin_question_toks[idx] == "are")
                    or (nltk_result[idx][1] == "VBP" and origin_question_toks[idx] == "do")
                    or (nltk_result[idx][1] == "VBP" and origin_question_toks[idx] == "doe")
                    or (
                        nltk_result[idx][1] == "VBP" and origin_question_toks[idx] == "does"
                    )
                )):
                    sub_tok_types.append(["db", db_cols])

                if endIdx == idx + 1:
                    if nltk_result[idx][1] == "RBR" or nltk_result[idx][1] == "JJR":
                        sub_tok_types.append(["MORE"])

                    if nltk_result[idx][1] == "RBS" or nltk_result[idx][1] == "JJS":
                        sub_tok_types.append(["MOST"])

                    # string match for Time Format
                    if num2year(sub_toks):
                        question_toks[idx] = "year"
                        partial_year_match = partial_matches(["year"], header_toks)
                        if "year" in header_toks:
                            sub_tok_types.append(["col", header_toks.index("year")])
                        elif partial_year_match:
                            sub_tok_types.append(["col_part", partial_year_match])

                    result = group_digital(question_toks, idx)
                    if result is True:
                        sub_tok_types.append(["value"])
                    if question_toks[idx] == ["ha"]:
                        question_toks[idx] = ["have"]
                    if len(sub_tok_types) == 0:
                        sub_tok_types.append(["NONE"])
                if len(sub_tok_types) != 0:
                    tok_concol.append(sub_toks_list)
                    type_concol.append(sub_tok_types)
                    idx = endIdx
                    break
        entry["question_arg"] = tok_concol
        entry["question_arg_type"] = type_concol
        entry["nltk_pos"] = nltk_result

    return datas


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", type=str, help="dataset", required=True)
    arg_parser.add_argument(
        "--table_path", type=str, help="table dataset", required=True
    )
    arg_parser.add_argument("--output", type=str, help="output data")
    args = arg_parser.parse_args()
    args.conceptNet = "./conceptNet"

    # loading dataSets
    datas, table = load_dataSets(args)
    # process datasets
    dataset_name = args.data_path.split("data/")[1].split("/")[0]
    process_result = process_datas(datas, args, dataset_name)
    print("length: {}".format(len(datas)))

    with open(args.output, "w") as f:
        f.write(json.dumps(datas, indent=4))
