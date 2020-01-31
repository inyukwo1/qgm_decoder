# -*- coding: utf-8 -*-
import json
import time
import pickle

import copy
import numpy as np
import os
import torch
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from src.dataset import Example
from src.rule import lf
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1
from qgm.utils import compare_boxes, filter_datas

wordnet_lemmatizer = WordNetLemmatizer()


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
    print("Loading word embedding from %s" % file_name)
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


def get_col_table_dict(tab_cols, tab_ids, sql, is_qgm=True):
    table_dict = {}
    cols = sql["col"] if is_qgm else sql["col_set"]
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
    if is_qgm:
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


def process(sql, table, is_qgm=True):

    process_dict = {}

    origin_sql = sql["question_toks"]
    table_names = [
        [wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]
        for x in table["table_names"]
    ]

    sql["pre_sql"] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table["column_names"]]
    tab_ids = [col[0] for col in table["column_names"]]

    cols = sql["col"] if is_qgm else sql["col_set"]
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


def to_batch_seq(sql_data, table_data, idxes, st, ed, is_qgm=True):
    examples = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        table = table_data[sql["db_id"]]

        process_dict = process(sql, table, is_qgm=is_qgm)

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
            process_dict["tab_cols"], process_dict["tab_ids"], sql, is_qgm
        )
        table_col_name = get_table_colNames(
            process_dict["tab_ids"], process_dict["col_iter"]
        )

        process_dict["col_set_iter"][0] = ["count", "number", "many"]

        rule_label = None
        if "rule_label" in sql and not is_qgm:
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
    is_qgm=True,
    is_train=True,
):
    if is_train:
        model.train()
    else:
        model.eval()
    # shuffle
    perm = np.random.permutation(len(sql_data))
    optimizer.zero_grad()
    if bert_optimizer:
        bert_optimizer.zero_grad()

    total_loss = {}
    for st in tqdm(range(0, len(sql_data), batch_size)):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, is_qgm=is_qgm)

        result = model.forward(examples)
        if is_qgm:
            losses, pred_boxes = result

            # Combine losses
            loss_list = []
            for loss in losses:
                loss_sum = sum([item for key, item in loss.items()])
                loss_list += [loss_sum]

            # Save loss
            if not total_loss:
                for key in losses[0].keys():
                    total_loss[key] = []
            for b_loss in losses:
                for key, item in b_loss.items():
                    total_loss[key] += [float(item)]

            loss = torch.mean(torch.stack(loss_list))

        else:
            sketch_prob_var, lf_prob_var = result

            # Save loss
            if not total_loss:
                total_loss["sketch"] = []
                total_loss["detail"] = []
            for sketch_loss in sketch_prob_var:
                total_loss["sketch"] += [float(sketch_loss)]
            for detail_loss in lf_prob_var:
                total_loss["detail"] += [float(detail_loss)]

            loss = torch.mean(sketch_prob_var) + torch.mean(lf_prob_var)
        if is_train:
            loss.backward()

        if clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        if is_train:
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
    model, batch_size, sql_data, table_data, is_qgm=True, return_details=False
):
    model.eval()
    perm = list(range(len(sql_data)))
    pred = []
    gold = []
    example_list = []
    for st in tqdm(range(0, len(sql_data), batch_size)):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, is_qgm=is_qgm)
        example_list += examples
        if is_qgm:
            pred += model.parse(examples)
            gold += [example.qgm for example in examples]
        else:
            for example in examples:
                pred += [model.parse([example])]
                gold += [example.tgt_actions]

    # Calculate acc
    total_acc, is_correct_list = model.decoder.get_accuracy(pred, gold)

    if return_details:
        return total_acc, is_correct_list, pred, gold, example_list
    else:
        return total_acc


def load_data_new(
    sql_path, table_data, use_small=False, is_simple_query=True, is_single_table=True
):
    sql_data = []

    print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    table_data_new = {table["db_id"]: table for table in table_data}

    # Filter data with qgm that has nested query
    sql_data = filter_datas(sql_data, is_simple_query, is_single_table)

    if use_small:
        return sql_data[:80], table_data_new
    else:
        return sql_data, table_data_new


def load_dataset(
    dataset_dir, use_small=False, is_simple_query=True, is_single_table=True
):
    print("Loading from datasets...")

    TABLE_PATH = os.path.join(dataset_dir, "tables.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train.json")
    DEV_PATH = os.path.join(dataset_dir, "dev.json")

    with open(TABLE_PATH) as inf:
        print("Loading data from %s" % TABLE_PATH)
        table_data = json.load(inf)

    train_sql_data, train_table_data = load_data_new(
        TRAIN_PATH,
        table_data,
        use_small=use_small,
        is_simple_query=is_simple_query,
        is_single_table=is_single_table,
    )
    val_sql_data, val_table_data = load_data_new(
        DEV_PATH,
        table_data,
        use_small=use_small,
        is_simple_query=is_simple_query,
        is_single_table=is_single_table,
    )

    return train_sql_data, train_table_data, val_sql_data, val_table_data


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


def write_eval_result_as(file_name, datas, is_corrects, accs, preds, golds):
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

            f.write("idx:    {}\n".format(idx))
            f.write("ans:    {}\n".format(ans))
            f.write("db_id:  {}\n".format(sql_json["db_id"]))
            f.write("SQL:    {}\n".format(sql_json["query"]))
            # Pretty print question and question type
            q_list = []
            q_type_list = []
            for idx in range(len(sql_json["question_arg"])):
                q = " ".join(sql_json["question_arg"][idx])
                q_type = " ".join(sql_json["question_arg_type"][idx])
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
            f.write("q:      {}\n".format(q_list))
            f.write("q_type: {}\n".format(q_type_list))

            f.write(
                "table:  {}\n".format(
                    str(
                        [
                            (idx, sql_json["table_names"][idx])
                            for idx in range(len(sql_json["table_names"]))
                        ]
                    )
                )
            )
            # To-Do: Need to print column's parent table as well
            f.write(
                "column: {}\n".format(
                    str(
                        [
                            (idx, sql_json["col"][idx])
                            for idx in range(len(sql_json["col"]))
                        ]
                    )
                )
            )
            f.write("gold:   {}\n".format(gold))
            f.write("pred:   {}\n".format(pred))
            f.write("\n")
