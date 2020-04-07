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


def get_col_table_dict(tab_cols, tab_ids, sql, is_col_set=True):
    table_dict = {}
    cols = sql["col_set"] if is_col_set else sql["col"]
    for c_id, c_v in enumerate(cols):
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

    # Modify
    if not is_col_set:
        col_table_dict = {
            idx: [item] if item != -1 else list(range(max(tab_ids) + 1))
            for idx, item in enumerate(tab_ids)
        }

    return col_table_dict


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


def process(sql, table, is_col_set=True):

    process_dict = {}

    origin_sql = sql["question_toks"]
    table_names = [
        [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]
        for x in table["table_names"]
    ]

    sql["pre_sql"] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table["column_names"]]
    tab_ids = [col[0] for col in table["column_names"]]

    cols = sql["col_set"] if is_col_set else sql["col"]
    col_set_iter = [
        [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in cols
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
            foreigns = [f for f, p in table["foreign_keys"]]
            primaries = [p for f, p in table["foreign_keys"]]
            if indices[0] in primaries:
                col_set_type[col_set_idx, 5] = 1
            if indices[0] in foreigns:
                col_set_type[col_set_idx, 6] = 1
        else:
            col_set_type[col_set_idx, 7] = 1

    process_dict["col_set_iter"] = col_set_iter
    process_dict["q_iter_small"] = q_iter_small
    process_dict["col_set_type"] = col_set_type
    process_dict["tab_set_type"] = tab_set_type
    process_dict["question_arg"] = question_arg
    process_dict["question_arg_type"] = question_arg_type
    process_dict["one_hot_type"] = one_hot_type
    process_dict["tab_cols"] = tab_cols
    process_dict["tab_ids"] = tab_ids
    process_dict["col_iter"] = col_iter
    process_dict["table_names"] = table_names
    process_dict["tab_set_iter"] = tab_set_iter
    process_dict["qgm"] = sql["qgm"]
    # process_dict["qgm_action"] = sql["qgm_action"]

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


def to_batch_seq(sql_data, table_data, idxes, st, ed, is_col_set=True):
    examples = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        table = table_data[sql["db_id"]]

        process_dict = process(sql, table, is_col_set=is_col_set)

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
            sql,
        )

        col_table_dict = get_col_table_dict(
            process_dict["tab_cols"], process_dict["tab_ids"], sql, is_col_set
        )
        table_col_name = get_table_colNames(
            process_dict["tab_ids"], process_dict["col_iter"]
        )

        process_dict["col_set_iter"][0] = ["count", "number", "many"]

        rule_label = None
        if "rule_label" in sql:
            # handle the subquery on From cause
            if "from" in sql["rule_label"]:
                continue
            rule_label = [eval(x) for x in sql["rule_label"].strip().split(" ")]
            if (
                is_valid(rule_label, col_set_table_dict=col_table_dict, sql=sql)
                is False
            ):
                continue

        example = Example(
            src_sent=process_dict["question_arg"],
            col_num=len(process_dict["col_set_iter"]),
            vis_seq=(sql["question"], process_dict["col_set_iter"], sql["query"]),
            tab_cols=process_dict["col_set_iter"],  # col for encoding
            tab_iter=process_dict["tab_set_iter"],  # tab for encoding
            sql=sql["query"],
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
            qgm=process_dict["qgm"],
            # qgm_action=process_dict["rule_label"],
            qgm_action=rule_label,
            relation=sql["relation"] if "relation" in sql else None,
            gt=sql["gt"],
        )

        example.sql_json = copy.deepcopy(sql)
        example.db_id = sql["db_id"]
        examples.append(example)

    examples.sort(key=lambda e: -len(e.src_sent))
    return examples


def epoch_train(
    model,
    optimizer,
    bert_optimizer,
    batch_size,
    sql_data,
    table_data,
    clip_grad,
    decoder_name,
    is_col_set=True,
    is_train=True,
    optimize_freq=1,
):

    model.train()
    # shuffle
    sql_data.sort(key=lambda elem: len(elem["gt"]))

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

    perm = list(range(len(sql_data)))
    optimizer.zero_grad()
    if bert_optimizer:
        bert_optimizer.zero_grad()

    total_loss = {}
    for idx, st in enumerate(tqdm(range(0, len(sql_data), batch_size))):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        examples = to_batch_seq(
            sql_data, table_data, perm, st, ed, is_col_set=is_col_set
        )

        result = model.forward(examples)
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
    model,
    batch_size,
    sql_data,
    table_data,
    model_name,
    is_col_set=True,
    return_details=False,
):
    model.eval()
    perm = list(range(len(sql_data)))
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
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        examples = to_batch_seq(
            sql_data, table_data, perm, st, ed, is_col_set=is_col_set,
        )
        example_list += examples
        if model_name == "transformer":
            tmp, _ = model.parse(examples)
            pred["preds"] += tmp["preds"]
            pred["refined_preds"] += tmp["refined_preds"]
            pred["arbitrated_preds"] += tmp["arbitrated_preds"]
            pred["initial_preds"] += tmp["initial_preds"]
        else:
            pred += model.parse(examples)

        if model_name == "lstm":
            tmp = [
                model.decoder.grammar.create_data(example.qgm) for example in examples
            ]
            tmp2 = []
            for item in tmp:
                tmp2 += [
                    [
                        model.decoder.grammar.str_to_action(value)
                        for value in item.split(" ")
                    ]
                ]
            tmp = tmp2
            gold += tmp
        elif model_name == "transformer":
            tmp = [
                model.decoder.grammar.create_data(example.qgm) for example in examples
            ]
            tmp2 = []
            for item in tmp:
                tmp2 += [
                    [
                        model.decoder.grammar.str_to_action(value)
                        for value in item.split(" ")
                    ]
                ]
            tmp = tmp2
            gold += tmp
        elif model_name == "qgm":
            gold += [example.qgm for example in examples]
        elif model_name == "semql":
            tmp = [
                model.decoder.grammar.create_data(example.qgm) for example in examples
            ]
            tmp2 = []
            for item in tmp:
                tmp2 += [
                    [
                        model.decoder.grammar.str_to_action(value)
                        for value in item.split(" ")
                    ]
                ]
            tmp = tmp2
            gold += tmp
        elif model_name == "ensemble":
            for example in examples:
                gold += [example.gt]
        else:
            raise RuntimeError("Unsupported model name")

    # Calculate acc
    if model_name == "ensemble":
        total_acc, is_correct_list = model.decoder.decoders[0].grammar.cal_acc(
            pred, gold
        )
    elif model_name == "transformer":
        total_acc_pred, is_correct_list_pred = model.decoder.grammar.cal_acc(
            pred["preds"], gold
        )
        total_acc_refined, is_correct_list_refined = model.decoder.grammar.cal_acc(
            pred["refined_preds"], gold
        )
        (
            total_acc_arbitrated,
            is_correct_list_arbitrated,
        ) = model.decoder.grammar.cal_acc(pred["arbitrated_preds"], gold)
        (
            total_acc_init_pred,
            is_correct_list_init_pred,
        ) = model.decoder.grammar.cal_acc(pred["initial_preds"], gold)
        return (
            total_acc_pred,
            total_acc_refined,
            total_acc_arbitrated,
            total_acc_init_pred,
        )
    else:
        total_acc, is_correct_list = model.decoder.grammar.cal_acc(pred, gold)

    if return_details:
        return total_acc, is_correct_list, pred, gold, example_list
    else:
        return total_acc


def load_data_new(sql_path, use_small=False, is_bert=False, query_type="simple"):
    sql_data = []
    log.info("Loading data from {}".format(sql_path))

    with open(sql_path) as f:
        data = lower_keys(json.load(f))
        sql_data += data

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

    return sql_data[:20] if use_small else sql_data


def load_dataset(is_toy, is_bert, dataset_path, query_type):
    # Get paths
    table_path = os.path.join(dataset_path, "tables.json")
    train_path = os.path.join(dataset_path, "train.json")
    val_path = os.path.join(dataset_path, "dev.json")
    table_data = []

    # Tables as dictionary
    with open(table_path) as f:
        table_data += json.load(f)
    table_data = {table["db_id"]: table for table in table_data}

    # Load data
    train_data = load_data_new(train_path, is_toy, is_bert, query_type)
    val_data = load_data_new(val_path, is_toy, is_bert, query_type)

    # Create relations
    train_data = [
        relation.create_relation(item, table_data, True) for item in train_data
    ]
    val_data = [relation.create_relation(item, table_data, True) for item in val_data]

    # Show dataset length
    log.info("Total training set: {}".format(len(train_data)))
    log.info("Total validation set: {}\n".format(len(val_data)))

    return train_data, val_data, table_data


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


def append_ground_truth(grammar, data_list):
    for idx, data in enumerate(data_list):
        gt_str = grammar.create_data(data["qgm"])
        data_list[idx]["gt"] = [
            grammar.str_to_action(item) for item in gt_str.split(" ")
        ]

    return data_list


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
