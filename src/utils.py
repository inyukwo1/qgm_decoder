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
import src.relation as relation
from rule.noqgm.noqgm import NOQGM
from rule.semql.semql import SemQL
from heat_map import *
from transformers import *

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
        tab_tmp = [
            idx
            for idx in range(len(col_tab_dic))
            if t_idx in col_tab_dic[idx]
        ]
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
    process_dict["question_arg_type"] = question_arg_type

    for c_id, col_ in enumerate(process_dict["col_set_iter"]):
        for q_id, ori in enumerate(process_dict["q_iter_small"]):
            if ori in col_:
                process_dict["col_set_type"][c_id][0] += 1

    for t_id, tab_ in enumerate(process_dict["table_names"]):
        for q_id, ori in enumerate(process_dict["q_iter_small"]):
            if ori in tab_:
                process_dict["tab_set_type"][t_id][0] += 1

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
        question_arg = copy.deepcopy(data["question_arg"])

        # Append [db], [value] types
        assert len(question_arg) == len(data["question_arg_type"])
        for idx, type in enumerate(data["question_arg_type"]):
            if "db" in type:
                question_arg[idx] = ["[db]"] + question_arg[idx]
            elif "value" in type:
                question_arg[idx] = ["[value]"] + question_arg[idx]
            elif "table" in type:
                question_arg[idx] = ["[table]"] + question_arg[idx]
            elif "col" in type:
                question_arg[idx] = ["[column]"] + question_arg[idx]
            elif "MOST" in type:
                question_arg[idx] = ["[most]"] + question_arg[idx]
            elif "MORE" in type:
                question_arg[idx] = ["[more]"] + question_arg[idx]

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

        # Hot type
        process_dict = process(data, data["db"])
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

        example = Example(
            src_sent=question_arg,  # src encoding (length as well)
            src_sent_type=process_dict["question_arg_type"],
            tab_cols=col_set_iter,  # col for encoding
            col_num=len(col_set_iter),  # col length
            table_names=table_names,  # tab encoding
            table_len=len(table_names),  # tab length
            col_hot_type=process_dict["col_set_type"],
            sql=data["query"],
            col_tab_dic=col_tab_dic,
            tab_col_dic=tab_col_dic,
            relation=data["relation"] if "relation" in data else None,
            gt=data["gt"],
            db_id=data["db_id"],
            db=data["db"],
            data=data,
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

        result = model.forward(examples, is_train=True)
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
    model, batch_size, sql_data, return_details=False,
):
    model.eval()
    pred = []
    gold = []
    details_list = []
    example_list = []
    for st in tqdm(range(0, len(sql_data), batch_size)):
        ed = min(st + batch_size, len(sql_data))
        examples = sql_data[st:ed]
        examples.sort(key=lambda example: -len(example.src_sent))
        example_list += examples
        output = model.parse(examples, return_details=return_details)
        if return_details:
            out, details = output
            details_list += details
            pred += out
        else:
            pred += output
        gold += [example.gt for example in examples]

    # Calculate acc
    if model.cfg.rule == "noqgm":
        total_acc, is_correct_list = NOQGM.noqgm.cal_acc(pred, gold)
    elif model.cfg.rule == "semql":
        total_acc, is_correct_list = SemQL.semql.cal_acc(pred, gold)
    else:
        raise NotImplementedError("not yet")

    if return_details:
        return total_acc, is_correct_list, pred, gold, example_list, details_list
    else:
        return total_acc


def load_data_new(
    sql_path, table_data, use_small=False, is_bert=False, query_type="simple", remove_punc=False, cfg=None,
):
    sql_data = []
    log.info("Loading data from {}".format(sql_path))

    with open(sql_path) as f:
        data = lower_keys(json.load(f))
        for datum in data:
            # Filter out some datas
            if remove_punc and datum['question_arg'][-1] in [['?'], ['.']]:
                del datum['question_arg'][-1]
                del datum['question_arg_type'][-1]

            if "FROM (" not in datum["query"]:
                sql_data += [datum]

    # Add db info
    for data in sql_data:
        db = table_data[data["db_id"]]
        db["col_set"] = data["col_set"]
        data["db"] = db
        data["column_names"] = db["column_names"]

        # Append ground truth
        if cfg.rule == "noqgm":
            gt_str = NOQGM.create_data(data["sql"], db)
        elif cfg.rule == "semql":
            # gt_str = SemQL.create_data(data["qgm"])
            gt_str = data["rule_label"]
        else:
            raise NotImplementedError("not yet")

        if gt_str and NOQGM.create_data(data["sql"], db) and gt_str != True:
            gt = [SemQL.str_to_action(item) for item in gt_str.split(" ")]
            data["gt"] = gt
    sql_data = [item for item in sql_data if "gt" in item]

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
    # sql_data = filter_datas(sql_data, query_type)

    return sql_data[:20] if use_small else sql_data


def load_dataset(model, is_toy, is_bert, dataset_path, query_type, use_down_schema, remove_punc, cfg):
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
    train_data = load_data_new(train_path, table_data, is_toy, is_bert, query_type, remove_punc, cfg)
    val_data = load_data_new(val_path, table_data, is_toy, is_bert, query_type, remove_punc, cfg)

    # # Append sql
    # for data in train_data:
    #     tmp = NOQGM.to_sql(data["gt"], data)

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
        train_data = down_schema(train_data)
        val_data = down_schema(val_data)

    # Parse datasets into exampels:
    train_data = to_batch_seq(train_data, table_data)
    val_data = to_batch_seq(val_data, table_data)

    # Append bert input data
    if is_bert:
        if model.encoder_name == "bert":
            bert_encoder = model.encoder
        elif model.encoder_name == "ra_transformer":
            bert_encoder = model.bert
        else:
            raise NotImplementedError("not yet")

        append_bert_input(train_data, bert_encoder.tokenizer)
        append_bert_input(val_data, bert_encoder.tokenizer)
        bert_encoder.create_cache([train_data, val_data])

    return train_data, val_data, table_data


def down_schema(datas):
    for idx, data in enumerate(datas):
        selected_table_indices = list(
            set([item[1] for item in data["gt"] if item[0] == "T"])
        )
        use_one_hop_neighbors = False
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
    file_name, datas, is_corrects, accs, preds, golds
):
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
            for idx_2 in range(len(sql_json["question_arg"])):
                q = " ".join(sql_json["question_arg"][idx_2])
                try:
                    q_type = " ".join(sql_json["question_arg_type"][idx_2])
                except:
                    q_type = " ".join(
                        [sql_json["question_arg_type"][idx_2][0]]
                        + sql_json["question_arg_type"][idx_2][1]
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

            tab_tmp = str(
                        " ".join(
                            [
                                "<{}: {}>".format(idx, sql_json["table_names"][idx])
                                for idx in range(len(sql_json["table_names"]))
                            ]
                        )
                    )
            f.write(
                "\ntable:  {}\n".format(tab_tmp)
            )
            # To-Do: Need to print column's parent table as well
            f.write("column: ")
            # Format column info
            col_infos = [
                "({}-{}: {})".format(
                    sql_json["col_table"][idx_2], idx_2, sql_json["col_set"][idx_2]
                )
                for idx_2 in range(len(sql_json["col_set"]))
            ]
            # Split line by 10 items
            col_tmp = ""
            for idx_2, col_info in enumerate(col_infos):
                if idx_2 % 10 == 0 and idx_2 != 0:
                    f.write("\n\t")
                    col_tmp += "\n\t"
                col_tmp += "{} ".format(col_info)
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
        col_dic_key = (col_len, len(example.tab_cols))
        tab_dic_key = (tab_len, len(example.table_names))

        # Initialize and count for col
        if col_dic_key not in col_acc_dic:
            print("db: {} col_len: {}".format(example.db_id, col_dic_key))
            col_cnt_tab_cnt[col_dic_key] = set()
            col_cnt_tab_cnt[col_dic_key].add(tab_dic_key)

            col_acc_dic[col_dic_key] = 0
            col_cnt_dic[col_dic_key] = 1
            # Detailed analysis
            col_cnt_act_dic[col_dic_key] = 0
            col_cnt_col_dic[col_dic_key] = 0
            col_cnt_tab_dic[col_dic_key] = 0
        else:
            col_cnt_dic[col_dic_key] += 1
            col_cnt_tab_cnt[col_dic_key].add(tab_dic_key)

        # Initialize and count for tab
        if tab_dic_key not in tab_acc_dic:
            tab_acc_dic[tab_dic_key] = 0
            tab_cnt_dic[tab_dic_key] = 1

            # Detailed analysis
            tab_cnt_act_dic[tab_dic_key] = 0
            tab_cnt_col_dic[tab_dic_key] = 0
            tab_cnt_tab_dic[tab_dic_key] = 0
        else:
            tab_cnt_dic[tab_dic_key] += 1

        # Count correct
        if pred == gold:
            tab_acc_dic[tab_dic_key] += 1
            col_acc_dic[col_dic_key] += 1
        # Count wrong
        else:
            min_len = min(len(pred), len(gold))
            for idx in range(min_len):
                if pred[idx] != gold[idx]:
                    if gold[idx][0] == "C":
                        tab_cnt_col_dic[tab_dic_key] += 1
                        col_cnt_col_dic[col_dic_key] += 1
                    elif gold[idx][0] == "T":
                        tab_cnt_tab_dic[tab_dic_key] += 1
                        col_cnt_tab_dic[col_dic_key] += 1
                    else:
                        tab_cnt_act_dic[tab_dic_key] += 1
                        col_cnt_act_dic[col_dic_key] += 1
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
    print("number of db: {}".format(len(dbs)))

# Analysis
def first_diff_symbol(pred, gold):
    for idx in range(min(len(pred), len(gold))):
        if pred[idx] != gold[idx]:
            return pred[idx][0]
    return None

# Num of column in the select clause
def wrong_in_select_col_num(pred, gold):
    for idx in range(min(len(pred), len(gold))):
        if pred[idx] != gold[idx] and pred[idx][0] == "Sel":
            return True
    return False

def wrong_in_agg_op(pred, gold):
    for idx in range(min(len(pred), len(gold))):
        if pred[idx] != gold[idx] and pred[idx][0] == "A":
            for idx2 in range(idx, -1, -1):
                return True
    return False

def wrong_in_where_yes_no(pred, gold):
    for idx in range(min(len(pred), len(gold))):
        if pred[idx] != gold[idx] and pred[idx][0] == "Root":
            return True
    return False

def wrong_in_where_op(pred, gold):
    for idx in range(min(len(pred), len(gold))):
        if pred[idx] != gold[idx] and pred[idx][0] == "Filter":
            return True
    return False

def categorize(pred, gold):
    where_flag = False
    if pred == gold:
        return 'None'
    for idx in range(min(len(pred), len(gold))):
        pred_action_symbol = pred[idx][0]
        where_flag = where_flag or pred_action_symbol == "Filter"
        if pred[idx] != gold[idx]:
            # Determine why
            if pred_action_symbol == "T":
                if "Filter" in [item[0] for item in pred[:idx]]:
                    reason = 'table_where'
                else:
                    reason = 'table_select'
            elif pred_action_symbol == "C":
                # In WHERE clause
                if "Filter" in [item[0] for item in pred[:idx]]:
                    reason = 'column_where'
                # In SELECT clause
                else:
                    if pred[idx][1] == 0:
                        reason = 'column_select_pred_star'
                    elif gold[idx][1] == 0:
                        reason = 'column_select_gold_star'
                    else:
                        reason = 'column_select'
            else:
                # Structural reason
                if pred_action_symbol == "Sel":
                    reason = 'select_num_of_column'
                elif pred_action_symbol == "A" and not where_flag:
                    reason = 'select_agg'
                elif pred_action_symbol == "Root":
                    reason = 'where_existence'
                elif pred_action_symbol == "Filter":
                    if pred[idx][1] in [0, 1] or gold[idx][1] in [0,1]:
                        reason = 'where_num'
                    else:
                        reason = 'where_operator'
                else:
                    raise RuntimeError("Should not be here")
            return reason

def save_data_for_analysis(tag, datas, preds, golds, details_list, dataset, save_path):
    log = {
        'tag': tag,
        'dataset': [{
            'name': dataset,
            'path': '/home/hkkang/debugging/irnet_qgm_transformer/data/spider/',
        }],
        'data': [],
        'grammar': ["{} -> {}".format(value[0],NOQGM.noqgm.actions[value[0]][value[1]]) for key, value in NOQGM.noqgm.aid_to_action.items()],
    }
    # Create folder for image
    image_folder_path = os.path.join(save_path, "images")
    if not os.path.exists(image_folder_path):
        print(image_folder_path)
        os.makedirs(image_folder_path)

    for idx in tqdm(range(len(datas))):
        data = datas[idx]
        details = details_list[idx]
        
        data_log = {}
        data_log['idx'] = idx
        data_log['query'] = [' '.join(tmp) for tmp in data.src_sent]
        data_log['sql'] = data.sql
        data_log['columns'] = [' '.join(tmp) for tmp in data.tab_cols]
        data_log['tables'] = [' '.join(tmp) for tmp in data.table_names]
        data_log['db'] = data.db_id,
        data_log['gold'] = golds[idx]
        data_log['pred'] = preds[idx]

        # Create tensor image and save its path
        for layer_idx, (weight_tensors, relation_weight_tensors) in enumerate(zip(details['qk_weights'], details['qk_relation_weights'])):
            weight_tensor_key = 'weight_tensors_{}'.format(layer_idx)
            relation_tensor_key = 'relation_weight_tensors_{}'.format(layer_idx)
            data_log[weight_tensor_key] = []
            data_log[relation_tensor_key] = []
            for head_idx, (head_weight_tensor, relation_head_weight_tensor) in enumerate(zip(weight_tensors, relation_weight_tensors)):
                # Create nl
                nl = data_log["query"] + data_log["columns"] + data_log["tables"]
                # weight tensor
                value = head_weight_tensor.cpu().numpy()
                image_path = os.path.join(image_folder_path, "{}_att_layer_{}_head_{}.png".format(idx, layer_idx, head_idx))
                # draw_heat_map(nl, value, 'att_layer_{}_head_{}'.format(layer_idx, head_idx), image_path)
                data_log[weight_tensor_key] += [image_path]

                # relation weight tensor
                value = relation_head_weight_tensor.cpu().numpy()
                image_path = os.path.join(image_folder_path, "{}_relation_att_layer_{}_head_{}.png".format(idx, layer_idx, head_idx))
                # draw_heat_map(nl, value, 'relation_att_layer_{}_head_{}'.format(layer_idx, head_idx), image_path)
                data_log[relation_tensor_key] += [image_path]

        # Create prob image and save its path
        inference_key = 'inference'
        data_log[inference_key] = []
        for pred_idx, (value, pred_action) in enumerate(zip(details["probs"], preds[idx])):
            key = "inference step {}".format(pred_idx)
            if pred_action[0] == "C":
                arg1 = data_log["columns"]
            elif pred_action[0] == "T":
                arg1 = data_log["tables"]
            else:
                arg1 = log["grammar"]
            image_path = os.path.join(image_folder_path, "{}_inference_{}.png".format(idx, pred_idx))
            # draw_inference_score(arg1, value, key, image_path)
            data_log[inference_key] += [image_path]


        # Detailed Analysis with pred and gold
        data_log['filter'] = categorize(preds[idx], golds[idx])
        log['data'] += [data_log]

    # Save
    log_file_path = os.path.join(save_path, 'result.pkl')
    with open(log_file_path, 'wb') as f:
        pickle.dump(log, f)


def append_bert_input(examples, tokenizer):
    if tokenizer.pad_token != '[PAD]':
        raise NotImplementedError("Need to fix _pad_bert_input method in Batch class")

    log.info("Tokenizing with bert")
    for idx in tqdm(range(len(examples))):
        example = examples[idx]
        sentence = example.src_sent
        cols = example.tab_cols
        tabs = example.table_names

        bert_input = "[CLS]"
        word_start_ends = []
        col_start_ends = []
        tab_start_ends = []
        # NL
        for word in sentence:
            start = len(tokenizer.tokenize(bert_input))
            bert_input += " {}".format(" ".join(word))
            end = len(tokenizer.tokenize(bert_input))
            word_start_ends.append((start, end))

        # Cols
        for col in cols:
            start = len(tokenizer.tokenize(bert_input))
            bert_input += " {} {}".format("[SEP]", " ".join(col))
            end = len(tokenizer.tokenize(bert_input))
            col_start_ends.append((start, end))

        # Tabs
        for tab in tabs:
            start = len(tokenizer.tokenize(bert_input))
            bert_input += " {} {}".format("[SEP]", " ".join(tab))
            end = len(tokenizer.tokenize(bert_input))
            tab_start_ends.append((start, end))

        if end >= tokenizer.max_len:
            raise RuntimeError("Bert input exceed {}".format(tokenizer.max_len))

        example.bert_input = bert_input
        example.bert_input_indices = [word_start_ends, col_start_ends, tab_start_ends]
