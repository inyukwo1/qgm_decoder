# -*- coding: utf-8 -*-
import os
import copy
import json
import time
import random
import torch
import pickle
import numpy as np
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import logging

from src.dataset import Example
from preprocess.rule import lf
from preprocess.rule.semQL import *
from decoder.qgm.utils import filter_datas
import src.relation as relation
from src.dataset import Batch
from rule.semql.semql import SemQL


wordnet_lemmatizer = WordNetLemmatizer()
log = logging.getLogger(__name__)


def idx2seq(seq, indices, cur_idx):
    if indices[cur_idx]:
        cur = indices[cur_idx]
        new_seq = []
        for item in cur:
            new_seq += idx2seq(seq, indices, item)
        return [seq[cur_idx]] + new_seq
    else:
        return [seq[cur_idx]]


def get_terminal_idx(seq, parent, begin_idx):
    """
    :param seq: List (list of SemQL in left-to-right order)
    :param parent: Action (parent action of current action)
    :param begin_idx: Int (idx of current action in seq)
    :return: Int (terminal idx of the current action with respect to seq)
    """
    cur = seq[begin_idx]
    assert str(parent) == str(cur.parent)

    if cur.children:
        children_idx = []
        for child in cur.children:
            target_idx = children_idx[-1] + 1 if children_idx else begin_idx + 1
            children_idx += [get_terminal_idx(seq, cur, target_idx)]
        return children_idx[-1]
    else:
        return begin_idx


def seq2idx(seq):
    """
    :param seq: List (list of SemQL in left-to-right order)
    :return: List (list of list containing indices as a tree)
    """
    indices = []

    for idx, cur in enumerate(seq):
        children_idx = [idx + 1] if cur.children else []
        children_idx += [
            get_terminal_idx(seq, cur, children_idx[-1]) + 1
            for children_num in range(1, len(cur.children))
        ]
        indices += [children_idx]

    return indices


def load_word_emb(file_name, use_small=False):
    log.info("Loading word embedding from {}".format(file_name))
    ret = {}

    cache_name = file_name.replace("txt", "pkl")
    if os.path.isfile(cache_name) and not use_small:
        with open(cache_name, "rb") as cache_file:
            ret = pickle.load(cache_file)
    else:
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 5000:
                    break
                info = line.strip().split(" ")
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(list(map(lambda x: float(x), info[1:])))
        with open(cache_name, "wb") as cache_file:
            pickle.dump(ret, cache_file)
    # # Add key words
    # key_words = ["[db]", "[table]", "[column]", "[value]"]
    # for key_word in key_words:
    #     ret[key_word] = np.random.randn(300)
    return ret


def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x


def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result


def get_col_tab_dic(tab_cols, tab_ids, sql):
    table_dict = {}
    for c_id, c_v in enumerate(sql["col_set"]):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [
                    c_id
                ]

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]

    return col_table_dict


def get_tab_col_dic(col_tab_dic):
    tab_col_dic = []
    b_tmp = []
    tab_len = len(col_tab_dic[0])
    for t_idx in range(tab_len):
        tab_tmp = [idx for idx in range(len(col_tab_dic)) if t_idx in col_tab_dic[idx]]
        b_tmp += [tab_tmp]
    tab_col_dic += [b_tmp]
    return tab_col_dic


def schema_linking(
    question_arg,
    question_arg_type,
    one_hot_type,
    col_set_type,
    col_set_iter,
    tab_set_type,
    tab_set_iter,
    sql,
):

    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == "NONE":
            continue
        elif t == "table":
            one_hot_type[count_q][0] = 1
            try:
                for tab_set_idx in range(len(tab_set_iter)):
                    if tab_set_iter[tab_set_idx] == question_arg[count_q]:
                        tab_set_type[tab_set_idx][1] = 5
                question_arg[count_q] = ["[table]"] + question_arg[count_q]
            except:
                print(tab_set_iter, question_arg[count_q])
                raise RuntimeError("not in tab set")
        elif t == "col":
            one_hot_type[count_q][1] = 1
            try:
                if len(t_q) > 1:
                    for col_name_idx in range(1, len(t_q)):
                        col_set_type[col_set_iter.index(t_q[col_name_idx])][8] += 1
                else:
                    for col_set_idx in range(len(col_set_iter)):
                        if col_set_iter[col_set_idx] == question_arg[count_q]:
                            col_set_type[col_set_idx][1] = 5
                question_arg[count_q] = ["[column]"] + question_arg[count_q]
            except:
                print(col_set_iter, question_arg[count_q])
                raise RuntimeError("not in col set")
        elif t == "agg":
            one_hot_type[count_q][2] = 1
        elif t == "MORE":
            one_hot_type[count_q][3] = 1
        elif t == "MOST":
            one_hot_type[count_q][4] = 1
        elif t == "value":
            one_hot_type[count_q][5] = 1
            question_arg[count_q] = ["[value]"] + question_arg[count_q]
        elif t == "db":
            one_hot_type[count_q][6] = 1
            question_arg[count_q] = ["[db]"] + question_arg[count_q]
            for col_name_idx in range(1, len(t_q)):
                c_cand = [
                    wordnet_lemmatizer.lemmatize(v).lower()
                    for v in t_q[col_name_idx].split(" ")
                ]
                col_set_type[col_set_iter.index(c_cand)][4] = 5
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == "asd":
                        continue
                    try:
                        for col_set_idx in range(len(sql["col_set"])):
                            if sql["col_set"][col_set_idx] == col_probase:
                                col_set_type[col_set_idx][2] = 5
                        question_arg[count_q] = ["[value]"] + question_arg[count_q]
                    except:
                        print(sql["col_set"], col_probase)
                        raise RuntimeError("not in col")
                    one_hot_type[count_q][5] = 1
            else:
                for col_probase in t_q:
                    if col_probase == "asd":
                        continue
                    for col_set_idx in range(len(sql["col_set"])):
                        if sql["col_set"][col_set_idx] == col_probase:
                            col_set_type[col_set_idx][3] += 1


def process(sql, db_data):
    process_dict = {}

    origin_sql = sql["question_toks"]
    table_names = [
        [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]
        for x in db_data["table_names"]
    ]

    sql["pre_sql"] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in db_data["column_names"]]
    tab_ids = [col[0] for col in db_data["column_names"]]

    col_set_iter = [
        [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]
        for x in sql["col_set"]
    ]
    tab_set_iter = [
        [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]
        for x in sql["table_names"]
    ]
    col_iter = [
        [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]
        for x in tab_cols
    ]
    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql["question_arg"])
    question_arg_type = sql["question_arg_type"]
    one_hot_type = np.zeros((len(question_arg_type), 7))

    col_set_type = np.zeros((len(col_set_iter), 9))  # 5:primary 6:foreign 7: multi
    tab_set_type = np.zeros((len(table_names), 5))

    for col_set_idx, col_name in enumerate(col_set_iter):
        indices = [i for i, x in enumerate(col_iter) if x == col_name]
        assert len(indices) > 0
        if len(indices) == 1:
            foreigns = [f for f, p in db_data["foreign_keys"]]
            primaries = [p for f, p in db_data["foreign_keys"]]
            if indices[0] in primaries:
                col_set_type[col_set_idx, 5] = 1
            if indices[0] in foreigns:
                col_set_type[col_set_idx, 6] = 1
        else:
            col_set_type[col_set_idx, 7] = 1

    process_dict["col_set_iter"] = col_set_iter  # for col encoding
    process_dict["q_iter_small"] = q_iter_small
    process_dict["col_set_type"] = col_set_type
    process_dict["tab_set_type"] = tab_set_type
    process_dict["question_arg"] = question_arg  # for src encoding
    process_dict["question_arg_type"] = question_arg_type
    process_dict["one_hot_type"] = one_hot_type
    process_dict["tab_cols"] = tab_cols
    process_dict["tab_ids"] = tab_ids
    process_dict["col_iter"] = col_iter
    process_dict["table_names"] = table_names
    process_dict["tab_set_iter"] = tab_set_iter  # for tab encoding

    return process_dict


def is_valid(rule_label, col_set_table_dict, sql):
    try:
        lf.build_tree(copy.copy(rule_label))
    except:
        print(rule_label)

    flag = False
    for r_id, rule in enumerate(rule_label):
        if type(rule) == C:
            try:
                assert (
                    rule_label[r_id + 1].id_c in col_set_table_dict[rule.id_c]
                ), print(sql["question"])
            except:
                flag = True
                print(sql["question"])
    return flag is False


def to_batch_seq(data_list, table_data):
    examples = []
    for data in data_list:
        # src
        table = table_data[data["db_id"]]
        process_dict = process(data, table)

        question_arg = copy.deepcopy(data["question_arg"])
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
            data,
        )
        # column
        col_set_iter = [
            [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]
            for x in data["col_set"]
        ]
        col_set_iter[0] = ["count", "number", "many"]

        # table
        table_names = [
            [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]
            for x in data["table_names"]
        ]

        # col table dic
        tab_cols = [col[1] for col in data["column_names"]]
        tab_ids = [col[0] for col in data["column_names"]]
        col_tab_dic = get_col_tab_dic(tab_cols, tab_ids, data)
        tab_col_dic = get_tab_col_dic(col_tab_dic)

        example = Example(
            src_sent=question_arg,  # src encoding (length as well)
            tab_cols=col_set_iter,  # col for encoding
            col_num=len(col_set_iter),  # col length
            table_names=table_names,  # tab encoding
            table_len=len(table_names),  # tab length
            sql=data["query"],
            col_tab_dic=col_tab_dic,
            tab_col_dic=tab_col_dic,
            qgm=data["qgm"],
            relation=data["relation"] if "relation" in data else None,
            gt=data["gt"],
            db_id=data["db_id"],
            db=data["db"],
            data=data,
            one_hot_type=process_dict["one_hot_type"],
            col_hot_type=process_dict["col_set_type"],
            tab_hot_type=process_dict["tab_set_type"],
            used_table_set=data["used_table_set"],
        )

        example.sql_json = copy.deepcopy(data)
        examples.append(example)

    examples.sort(key=lambda elem: len(elem.gt))

    return examples


def epoch_train(
    model,
    optimizer,
    bert_optimizer,
    batch_size,
    sql_data,
    clip_grad,
    decoder_name,
    is_train=True,
    optimize_freq=1,
):

    model.train()
    # shuffle
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    new_sql_data = []
    for sql_data_chunk in chunks(sql_data, batch_size * 3):
        random.shuffle(sql_data_chunk)
        new_sql_data += [sql_data_chunk]
    random.shuffle(new_sql_data)
    sql_data = []
    for sql_data_chunk in new_sql_data:
        sql_data += sql_data_chunk

    optimizer.zero_grad()
    if bert_optimizer:
        bert_optimizer.zero_grad()

    total_loss = {}
    for idx, st in enumerate(tqdm(range(0, len(sql_data), batch_size))):
        ed = min(st + batch_size, len(sql_data))
        examples = sql_data[st:ed]
        examples.sort(key=lambda example: -len(example.src_sent))

        result = model.forward(examples, True)
        if decoder_name == "lstm":
            tmp = {key: [] for key in result[0].get_keys()}
            for losses in result:
                for key, item in losses.get_dic().items():
                    tmp[key] += [item]
            tmp = {key: torch.mean(torch.stack(item)) for key, item in tmp.items()}
            loss = tmp["sketch"] + tmp["detail"]

            # Save
            if not total_loss:
                total_loss = {key: [] for key in result[0].get_keys()}
            for key, item in tmp.items():
                total_loss[key] += [float(item)]
        elif decoder_name == "transformer":
            loss = result.loss_dic["sketch"] + result.loss_dic["detail"]

            # Save
            if not total_loss:
                total_loss = {key: [] for key in result.get_keys()}
            for key, item in result.loss_dic.items():
                total_loss[key] += [float(item)]

        elif decoder_name == "semql":
            loss = result.loss_dic["sketch"] + result.loss_dic["detail"]

            # Save
            if not total_loss:
                total_loss = {key: [] for key in result.get_keys()}
            for key, item in result.loss_dic.items():
                total_loss[key] += [float(item)]
        else:
            raise RuntimeError("Unsupported model")

        if is_train:
            if loss.requires_grad:
                loss.backward()
            if idx % optimize_freq == 0:
                if clip_grad > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                if bert_optimizer:
                    bert_optimizer.step()
                optimizer.zero_grad()
                if bert_optimizer:
                    bert_optimizer.zero_grad()

    # Average loss
    for key in total_loss.keys():
        total_loss[key] = sum(total_loss[key]) / len(total_loss[key])

    return total_loss


def epoch_acc(
    model, batch_size, sql_data, model_name, use_table_only, return_details=False,
):
    model.eval()
    if model_name == "transformer":
        pred = {
            "preds": [],
            "refined_preds": [],
            "arbitrated_preds": [],
            "initial_preds": [],
        }
    else:
        pred = []

    gold = []
    example_list = []
    for st in tqdm(range(0, len(sql_data), batch_size)):
        ed = min(st + batch_size, len(sql_data))
        examples = sql_data[st:ed]
        examples.sort(key=lambda example: -len(example.src_sent))
        example_list += examples
        if model_name == "transformer":
            tmp = model.parse(examples)
            pred["preds"] += tmp["preds"]
            pred["refined_preds"] += tmp["refined_preds"]
            pred["arbitrated_preds"] += tmp["arbitrated_preds"]
            pred["initial_preds"] += tmp["initial_preds"]
        else:
            pred += model.parse(examples)

        gold += [
            example.used_table_set if use_table_only else example.gt
            for example in examples
        ]

    # Calculate acc
    if use_table_only:
        correct = 0
        for idx, (one_pred, one_gold) in enumerate(zip(pred["preds"], gold)):
            # if idx in [
            #     0,
            #     1,
            #     2,
            #     3,
            #     6,
            #     7,
            #     8,
            #     9,
            #     10,
            #     11,
            #     12,
            #     13,
            #     14,
            #     15,
            #     16,
            #     17,
            #     18,
            #     19,
            #     20,
            #     21,
            #     22,
            #     23,
            #     24,
            #     25,
            #     26,
            #     27,
            #     28,
            #     29,
            #     30,
            #     32,
            #     33,
            #     34,
            #     35,
            #     36,
            #     37,
            #     38,
            #     39,
            #     44,
            #     45,
            #     47,
            #     49,
            #     50,
            #     51,
            #     52,
            #     54,
            #     56,
            #     57,
            #     59,
            #     60,
            #     61,
            #     64,
            #     65,
            #     66,
            #     67,
            #     68,
            #     69,
            #     70,
            #     71,
            #     73,
            #     74,
            #     75,
            #     76,
            #     77,
            #     78,
            #     79,
            #     80,
            #     81,
            #     82,
            #     84,
            #     89,
            #     90,
            #     95,
            #     96,
            #     97,
            #     98,
            #     99,
            #     100,
            #     101,
            #     102,
            #     103,
            #     105,
            #     108,
            #     109,
            #     110,
            #     111,
            #     112,
            #     113,
            #     114,
            #     118,
            #     121,
            #     122,
            #     124,
            #     126,
            #     127,
            #     128,
            #     129,
            #     131,
            #     133,
            #     134,
            #     137,
            #     138,
            #     141,
            #     143,
            #     144,
            #     145,
            #     146,
            #     148,
            #     150,
            #     151,
            #     152,
            #     153,
            #     154,
            #     155,
            #     156,
            #     157,
            #     158,
            #     159,
            #     160,
            #     163,
            #     164,
            #     166,
            #     168,
            #     169,
            #     177,
            #     178,
            #     179,
            #     180,
            #     185,
            #     191,
            #     192,
            #     193,
            #     194,
            #     195,
            #     196,
            #     197,
            #     198,
            #     200,
            #     201,
            #     202,
            #     207,
            #     209,
            #     211,
            #     217,
            #     218,
            #     223,
            #     225,
            #     229,
            #     234,
            #     235,
            #     236,
            #     237,
            #     238,
            #     239,
            #     240,
            #     243,
            #     245,
            #     246,
            #     247,
            #     248,
            #     249,
            #     250,
            #     251,
            #     252,
            #     253,
            #     259,
            #     263,
            #     264,
            #     268,
            #     269,
            #     271,
            #     273,
            #     274,
            #     275,
            #     276,
            #     277,
            #     278,
            #     281,
            #     282,
            #     283,
            #     284,
            #     285,
            #     286,
            #     287,
            #     288,
            #     289,
            #     290,
            #     293,
            #     294,
            #     295,
            #     296,
            #     297,
            #     298,
            #     299,
            #     300,
            #     302,
            #     303,
            #     304,
            #     305,
            #     308,
            #     310,
            #     311,
            #     312,
            #     314,
            #     315,
            #     317,
            #     320,
            #     322,
            #     328,
            #     329,
            #     330,
            #     331,
            #     334,
            #     335,
            #     336,
            #     337,
            #     338,
            #     339,
            #     341,
            #     344,
            #     345,
            #     346,
            #     347,
            #     350,
            #     351,
            #     353,
            #     354,
            #     355,
            #     356,
            #     357,
            #     358,
            #     359,
            #     360,
            #     361,
            #     363,
            #     364,
            #     365,
            #     366,
            #     367,
            #     368,
            #     369,
            #     370,
            #     371,
            #     375,
            #     379,
            #     380,
            #     382,
            #     383,
            #     384,
            #     385,
            #     387,
            #     396,
            #     397,
            #     398,
            #     408,
            #     419,
            #     431,
            #     433,
            #     434,
            # ]:
            if one_pred[0][1] + 1 == len(one_gold) and set(
                [action[1] for action in one_pred[1:]]
            ) == set(one_gold):
                correct += 1
            else:
                print("wrong")
        total_acc = correct / len(gold)
        return {"total": total_acc}, {"total": 0.0}, {"total": 0.0}, {"total": 0.0}
    elif model_name == "ensemble":
        total_acc, is_correct_list = model.decoder.decoders[0].grammar.cal_acc(
            pred, gold
        )
    elif model_name == "transformer":
        total_acc_pred, is_correct_list_pred = SemQL.semql.cal_acc(pred["preds"], gold)
        total_acc_refined, is_correct_list_refined = SemQL.semql.cal_acc(
            pred["refined_preds"], gold
        )
        (total_acc_arbitrated, is_correct_list_arbitrated,) = SemQL.semql.cal_acc(
            pred["arbitrated_preds"], gold
        )
        (total_acc_init_pred, is_correct_list_init_pred,) = SemQL.semql.cal_acc(
            pred["initial_preds"], gold
        )

        return (
            total_acc_pred,
            total_acc_refined,
            total_acc_arbitrated,
            total_acc_init_pred,
        )
    else:
        total_acc, is_correct_list = SemQL.semql.cal_acc(pred, gold)

    if return_details:
        return total_acc, is_correct_list, pred, gold, example_list
    else:
        return total_acc, 0, 0, 0


def get_used_table_set(sql):
    table_set = set()
    if isinstance(sql, dict):
        for key in sql:
            if key == "from":
                for elem in sql[key]["table_units"]:
                    assert elem[0] == "table_unit"
                    table_set.add(elem[1])
            else:
                table_set |= get_used_table_set(sql[key])
    elif isinstance(sql, list):
        for sub_sql in sql:
            table_set |= get_used_table_set(sub_sql)
    return table_set


def load_data_new(
    sql_path, table_data, use_small=False, is_bert=False, query_type="simple"
):
    sql_data = []
    log.info("Loading data from {}".format(sql_path))

    with open(sql_path) as f:
        data = lower_keys(json.load(f))
        for datum in data:
            if "FROM (" not in datum["query"]:
                sql_data += [datum]

    # Add db info
    for data in sql_data:
        db = table_data[data["db_id"]]
        data["db"] = db
        data["column_names"] = db["column_names"]
        # Append ground truth
        if query_type == "all":
            gt_str = data["rule_label"]
        else:
            gt_str = SemQL.create_data(data["qgm"])
        gt = [SemQL.str_to_action(item) for item in gt_str.split(" ")]
        data["gt"] = gt
        data["used_table_set"] = list(get_used_table_set(data["sql"]))

    # Filter some db
    if is_bert:
        sql_data = [
            data
            for data in sql_data
            if data["db_id"] != "baseball_1"
            and data["db_id"] != "cre_Drama_Workshop_Groups"
            and data["db_id"] != "sakila_1"
            and data["db_id"] != "formula_1"
            and data["db_id"] != "soccer_1"
        ]

    # Filter data with qgm that has nested query
    sql_data = filter_datas(sql_data, query_type)

    return sql_data[:10] if use_small else sql_data


def load_dataset(is_toy, is_bert, dataset_path, query_type, use_down_schema):
    # Get paths
    table_path = os.path.join(dataset_path, "tables.json")
    train_path = os.path.join(dataset_path, "train.json")
    val_path = os.path.join(dataset_path, "dev.json")
    table_data = []

    # Tables as dictionary
    with open(table_path) as f:
        table_data += json.load(f)
    table_data = {table["db_id"]: table for table in table_data}

    # Append neighbor mapping
    for key, db in table_data.items():
        neighbors = {}
        for left, right in db["foreign_keys"]:
            left_tab = db["column_names"][left][0]
            right_tab = db["column_names"][right][0]
            # Append left
            if left_tab not in neighbors:
                neighbors[left_tab] = [right_tab]
            elif right_tab not in neighbors[left_tab]:
                neighbors[left_tab] += [right_tab]
            # Append right
            if right_tab not in neighbors:
                neighbors[right] = [left_tab]
            elif left_tab not in neighbors[right_tab]:
                neighbors[right_tab] += [left_tab]
        table_data[key]["neighbors"] = neighbors

    # Load data
    train_data = load_data_new(train_path, table_data, is_toy, is_bert, query_type)
    val_data = load_data_new(val_path, table_data, is_toy, is_bert, query_type)

    # Create relations
    train_data = [
        relation.create_relation(item, table_data, True) for item in train_data
    ]
    val_data = [relation.create_relation(item, table_data, True) for item in val_data]

    # Show dataset length
    log.info("Total training set: {}".format(len(train_data)))
    log.info("Total validation set: {}\n".format(len(val_data)))

    # filter schema using 1-hop scheme
    if use_down_schema:
        # train_data = down_schema(train_data)
        val_data = down_schema(val_data)

    # Parse datasets into exampels:
    train_data = to_batch_seq(train_data, table_data)
    val_data = to_batch_seq(val_data, table_data)

    return train_data, val_data, table_data


def down_schema(datas):
    for data in datas:
        selected_table_indices = list(
            set([item[1] for item in data["gt"] if item[0] == "T"])
        )
        use_one_hop_neighbors = True
        if use_one_hop_neighbors:
            # Append neighbor tables
            selected_table_indices = append_one_hop_neighbors(
                data["db"]["neighbors"], selected_table_indices
            )
        sub_schema = create_sub_schema(
            selected_table_indices,
            data["table_names"],
            data["column_names"],
            data["col_set"],
            data["gt"],
        )

        # Alter schema
        for key, item in sub_schema.items():
            data[key] = item

        # Alter Relation matrix
        relation = data["relation"]
        new_relation = create_sub_relation(
            relation, data["column_mapping"], data["table_mapping"]
        )

        # Replace relations
        for key, item in new_relation.items():
            relation[key] = item

    return datas


def create_sub_relation(relation, column_mapping, table_mapping):
    # qc
    qc = []
    for item in relation["qc"]:
        tmp = [id for idx, id in enumerate(item) if idx in column_mapping]
        qc += [tmp]

    # qt
    qt = []
    for item in relation["qt"]:
        tmp = [id for idx, id in enumerate(item) if idx in table_mapping]
        qt += [tmp]

    # cq
    cq = [item for idx, item in enumerate(relation["cq"]) if idx in column_mapping]

    # cc
    cc = []
    for idx_1, item in enumerate(relation["cc"]):
        if idx_1 in column_mapping:
            tmp = [id for idx, id in enumerate(item) if idx in column_mapping]
            cc += [tmp]

    # ct
    ct = []
    for idx_1, item in enumerate(relation["ct"]):
        if idx_1 in column_mapping:
            tmp = [id for idx, id in enumerate(item) if idx in table_mapping]
            ct += [tmp]

    # tq
    tq = [item for idx, item in enumerate(relation["tq"]) if idx in table_mapping]

    # tc
    tc = []
    for idx_1, item in enumerate(relation["tc"]):
        if idx_1 in table_mapping:
            tmp = [id for idx, id in enumerate(item) if idx in column_mapping]
            tc += [tmp]

    # tt
    tt = []
    for idx_1, item in enumerate(relation["tt"]):
        if idx_1 in table_mapping:
            tmp = [id for idx, id in enumerate(item) if idx in table_mapping]
            tt += [tmp]

    # Create new relation
    new_relation = {}
    new_relation["qc"] = qc
    new_relation["qt"] = qt
    new_relation["cq"] = cq
    new_relation["cc"] = cc
    new_relation["ct"] = ct
    new_relation["tq"] = tq
    new_relation["tc"] = tc
    new_relation["tt"] = tt

    return new_relation


def append_one_hop_neighbors(neighbor_dic, selected_table_indices):
    # Append neighbor tables
    new_indices = copy.deepcopy(selected_table_indices)
    for item in selected_table_indices:
        if item in neighbor_dic:
            new_indices += neighbor_dic[item]
    return list(set(new_indices))


def create_sub_schema(selected_table_ids, table_names, column_names, col_set, gt=None):
    # New tables
    new_table_names = []
    table_mapping = {}
    for idx in selected_table_ids:
        # Mapping
        table_mapping[idx] = len(new_table_names)
        # New table names
        new_table_names += [table_names[idx]]

    # New column_names
    new_column_names = []
    for tab_id, column_name in column_names:
        if tab_id == -1:
            new_column_names += [(-1, column_name)]
        elif tab_id in selected_table_ids:
            new_tab_id = table_mapping[tab_id]
            new_column_names += [(new_tab_id, column_name)]

    # New col_set
    selected_cols = [item[1] for item in new_column_names]
    column_mapping = {}
    new_col_set = []
    for idx, column_name in enumerate(col_set):
        if column_name in selected_cols:
            column_mapping[idx] = len(column_mapping)
            new_col_set += [column_name]

    # New gt
    if gt:
        new_gt = []
        for symbol, local_idx in gt:
            if symbol == "C":
                new_col = column_mapping[local_idx]
                new_gt += [("C", new_col)]
            elif symbol == "T":
                new_tab = table_mapping[local_idx]
                new_gt += [("T", new_tab)]
            else:
                new_gt += [(symbol, local_idx)]

    # Append
    dic = {}
    dic["column_mapping"] = column_mapping
    dic["table_mapping"] = table_mapping
    dic["table_names"] = new_table_names
    dic["column_names"] = new_column_names
    dic["col_set"] = new_col_set
    if gt:
        dic["gt"] = new_gt

    return dic


def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)


def save_args(args, path):
    with open(path, "w") as f:
        f.write(json.dumps(vars(args), indent=4))


def init_log_checkpoint_path(args):
    save_path = args.save
    dir_name = save_path + str(int(time.time()))
    save_path = os.path.join(os.path.curdir, "saved_model", dir_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path


def write_eval_result_as(
    file_name, datas, is_corrects, accs, preds, golds, use_col_set=False
):
    col_key = "col_set" if use_col_set else "col"

    def sort_dic(dic):
        if isinstance(dic, dict):
            dic = {key: sort_dic(dic[key]) for key in sorted(dic.keys())}
            if "join_predicates" in dic.keys():
                del dic["join_predicates"]
        elif isinstance(dic, list):
            for idx in range(len(dic)):
                dic[idx] = sort_dic(dic[idx])
        return dic

    with open(file_name, "w") as f:
        # Length
        f.write("Data len: {}\n\n".format(len(golds)))

        # Accuracy
        for key, value in accs.items():
            f.write("acc {}:{}{}\n".format(key, (20 - len(key)) * " ", value))
        f.write("\n")

        # Results
        for idx in range(len(golds)):
            pred = preds[idx]
            gold = golds[idx]
            sql_json = datas[idx].sql_json

            # Sort for pretty print
            pred = sort_dic(pred)
            gold = sort_dic(gold)
            ans = "Correct" if is_corrects[idx] else "Wrong"

            f.write("\nidx:    {}\n".format(idx))
            f.write("ans:    {}\n".format(ans))
            f.write("db_id:  {}\n".format(sql_json["db_id"]))
            f.write("SQL:    {}\n".format(sql_json["query"]))
            # Pretty print question and question type
            q_list = []
            q_type_list = []
            for idx in range(len(sql_json["question_arg"])):
                q = " ".join(sql_json["question_arg"][idx])
                try:
                    q_type = " ".join(sql_json["question_arg_type"][idx])
                except:
                    q_type = " ".join(
                        [sql_json["question_arg_type"][idx][0]]
                        + sql_json["question_arg_type"][idx][1]
                    )
                q_pad_len = max(len(q), len(q_type)) - len(q)
                q_type_pad_len = max(len(q), len(q_type)) - len(q_type)
                q_pad = " " * (q_pad_len // 2)
                q_pad_extra = " " if q_pad_len % 2 else ""
                q_type_pad = " " * (q_type_pad_len // 2)
                q_type_pad_extra = " " if q_type_pad_len % 2 else ""
                q_list += ["{}{}{}{}".format(q_pad, q, q_pad, q_pad_extra)]
                q_type_list += [
                    "{}{}{}{}".format(q_type_pad, q_type, q_type_pad, q_type_pad_extra)
                ]
            f.write("\nq:      | {} |\n".format(" | ".join(q_list)))
            f.write("q_type: | {} |\n".format(" | ".join(q_type_list)))

            f.write(
                "\ntable:  {}\n".format(
                    str(
                        " ".join(
                            [
                                "<{}: {}>".format(idx, sql_json["table_names"][idx])
                                for idx in range(len(sql_json["table_names"]))
                            ]
                        )
                    )
                )
            )
            # To-Do: Need to print column's parent table as well
            f.write("column: ")
            # Format column info
            col_infos = [
                "({}-{}: {})".format(
                    sql_json["col_table"][idx], idx, sql_json[col_key][idx]
                )
                for idx in range(len(sql_json[col_key]))
            ]
            # Split line by 10 items
            for idx, col_info in enumerate(col_infos):
                if idx % 10 == 0 and idx != 0:
                    f.write("\n\t")
                f.write("{} ".format(col_info))

            f.write("\n\ngold:   {}\n".format(gold))
            f.write("pred:   {}\n".format(pred))
            f.write("\n\n{}\n\n".format("-" * 200))


def logging_to_tensorboard(summary_writer, prefix, summary, epoch):
    if isinstance(summary, dict):
        for key in summary.keys():
            summary_writer.add_scalar(prefix + key, summary[key], epoch)
    else:
        summary_writer.add_scalar(prefix, summary, epoch)


def calculate_total_acc(total_accs, data_lens):
    # Calculate Total Acc
    total_acc = {}
    for idx in range(len(data_lens)):
        data_len = data_lens[idx]
        for key, value in total_accs[idx].items():
            if key in total_acc:
                total_acc[key] += value * data_len
            else:
                total_acc[key] = value * data_len

    # Average
    total_acc = {key: value / sum(data_lens) for key, value in total_acc.items()}

    return total_acc


def set_random_seed(seed):
    torch.autograd.set_detect_anomaly(True)
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def analyze_regarding_schema_size(examples, is_correct, preds, golds, table_data):
    tab_acc_dic = {}
    col_acc_dic = {}
    tab_cnt_dic = {}
    col_cnt_dic = {}
    # table: action, column, table
    tab_cnt_act_dic = {}
    tab_cnt_col_dic = {}
    tab_cnt_tab_dic = {}

    # column: action, column, table
    col_cnt_act_dic = {}
    col_cnt_col_dic = {}
    col_cnt_tab_dic = {}

    # Column number for table num
    col_cnt_tab_cnt = {}

    dbs = set()
    for example, correct, pred, gold in zip(examples, is_correct, preds, golds):
        # Categorize by schema size
        db = table_data[example.db_id]
        dbs.add(example.db_id)
        col_len = len(db["column_names"])
        tab_len = len(db["table_names"])

        # Initialize and count for col
        if col_len not in col_acc_dic:
            print("db: {} col_len: {}".format(example.db_id, col_len))
            col_cnt_tab_cnt[col_len] = set()
            col_cnt_tab_cnt[col_len].add(tab_len)

            col_acc_dic[col_len] = 0
            col_cnt_dic[col_len] = 1
            # Detailed analysis
            col_cnt_act_dic[col_len] = 0
            col_cnt_col_dic[col_len] = 0
            col_cnt_tab_dic[col_len] = 0
        else:
            col_cnt_dic[col_len] += 1
            col_cnt_tab_cnt[col_len].add(tab_len)

        # Initialize and count for tab
        if tab_len not in tab_acc_dic:
            tab_acc_dic[tab_len] = 0
            tab_cnt_dic[tab_len] = 1

            # Detailed analysis
            tab_cnt_act_dic[tab_len] = 0
            tab_cnt_col_dic[tab_len] = 0
            tab_cnt_tab_dic[tab_len] = 0
        else:
            tab_cnt_dic[tab_len] += 1

        # Count correct
        if pred == gold:
            tab_acc_dic[tab_len] += 1
            col_acc_dic[col_len] += 1
        # Count wrong
        else:
            min_len = min(len(pred), len(gold))
            for idx in range(min_len):
                if pred[idx] != gold[idx]:
                    if gold[idx][0] == "C":
                        tab_cnt_col_dic[tab_len] += 1
                        col_cnt_col_dic[col_len] += 1
                    elif gold[idx][0] == "T":
                        tab_cnt_tab_dic[tab_len] += 1
                        col_cnt_tab_dic[col_len] += 1
                    else:
                        tab_cnt_act_dic[tab_len] += 1
                        col_cnt_act_dic[col_len] += 1
                    break

    for key, item in sorted(col_acc_dic.items()):
        cnt = col_cnt_dic[key]
        acc = item / cnt
        action_acc = 1 - col_cnt_act_dic[key] / cnt
        col_acc = 1 - col_cnt_col_dic[key] / cnt
        tab_acc = 1 - col_cnt_tab_dic[key] / cnt
        print(
            "Column size: {}  cnt: {}  Total Acc: {}  Action Acc: {}  Column Acc: {}  Table Acc: {}".format(
                key, cnt, acc, action_acc, col_acc, tab_acc
            )
        )
    print("")
    for key, item in sorted(tab_acc_dic.items()):
        cnt = tab_cnt_dic[key]
        acc = item / cnt
        action_acc = 1 - tab_cnt_act_dic[key] / cnt
        col_acc = 1 - tab_cnt_col_dic[key] / cnt
        tab_acc = 1 - tab_cnt_tab_dic[key] / cnt
        print(
            "Table size: {}  cnt: {}  Total Acc: {}  Action Acc: {}  Column Acc: {}  Table Acc: {}".format(
                key, cnt, acc, action_acc, col_acc, tab_acc
            )
        )

    tab_cnt_col_cnt = {}
    for col_len in sorted(col_cnt_tab_cnt.keys()):
        tab_lens = col_cnt_tab_cnt[col_len]
        print("col_len: {} tab_len: {}".format(col_len, tab_lens))

        for tab_len in tab_lens:
            if tab_len not in tab_cnt_col_cnt.keys():
                tab_cnt_col_cnt[tab_len] = set()
                tab_cnt_col_cnt[tab_len].add(col_len)
            else:
                tab_cnt_col_cnt[tab_len].add(col_len)

    print("")
    for tab_len in sorted(tab_cnt_col_cnt.keys()):
        col_len = tab_cnt_col_cnt[tab_len]
        print("tab_len: {} col_len: {}".format(tab_len, col_len))
    print("number of db: {}".format(len(db)))
