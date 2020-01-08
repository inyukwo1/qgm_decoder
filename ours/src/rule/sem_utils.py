# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/27
# @Author  : Jiaqi&Zecheng
# @File    : sem_utils.py
# @Software: PyCharm
"""

import os
import json
import argparse
import re as regex
from nltk.stem import WordNetLemmatizer
from pattern.en import lemma
wordnet_lemmatizer = WordNetLemmatizer()


def load_dataSets(args):
    with open(args.input_path, 'r') as f:
        datas = json.load(f)
    with open(os.path.join(args.data_path, 'tables.json'), 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    schemas = dict()
    for i in range(len(table_datas)):
        schemas[table_datas[i]['db_id']] = table_datas[i]
    return datas, schemas


def partial_match(query, table_name):
    query = [lemma(x) for x in query]
    table_name = [lemma(x) for x in table_name]
    if query in table_name:
        return True
    return False


def is_partial_match(query, table_names):
    query = lemma(query)
    table_names = [[lemma(x) for x in names.split(' ') ] for names in table_names]
    same_count = 0
    result = None
    for names in table_names:
        if query in names:
            same_count += 1
            result = names
    return result if same_count == 1 else False


def multi_option(question, q_ind, names, N):
    for i in range(q_ind + 1, q_ind + N + 1):
        if i < len(question):
            re = is_partial_match(question[i][0], names)
            if re is not False:
                return re
    return False


def multi_equal(question, q_ind, names, N):
    for i in range(q_ind + 1, q_ind + N + 1):
        if i < len(question):
            if question[i] == names:
                return i
    return False


def random_choice(question_arg, question_arg_type, names, ground_col_labels, q_ind, N, origin_name):
    # first try if there are other table
    for t_ind, t_val in enumerate(question_arg_type):
        if t_val == ['table']:
            return names[origin_name.index(question_arg[t_ind])]
    for i in range(q_ind + 1, q_ind + N + 1):
        if i < len(question_arg):
            if len(ground_col_labels) == 0:
                for n in names:
                    if partial_match(question_arg[i][0], n) is True:
                        return n
            else:
                for n_id, n in enumerate(names):
                    if n_id in ground_col_labels and partial_match(question_arg[i][0], n) is True:
                        return n
    if len(ground_col_labels) > 0:
        return names[ground_col_labels[0]]
    else:
        return names[0]


def find_table(cur_table, origin_table_names, question_arg_type, question_arg):
    h_table = None
    for i in range(len(question_arg_type))[::-1]:
        if question_arg_type[i] == ['table']:
            h_table = question_arg[i]
            h_table = origin_table_names.index(h_table)
            if h_table != cur_table:
                break
    if h_table != cur_table:
        return h_table

    # find partial
    for i in range(len(question_arg_type))[::-1]:
        if question_arg_type[i] == ['NONE']:
            for t_id, table_name in enumerate(origin_table_names):
                if partial_match(question_arg[i], table_name) is True and t_id != h_table:
                    return t_id

    # random return
    for i in range(len(question_arg_type))[::-1]:
        if question_arg_type[i] == ['table']:
            h_table = question_arg[i]
            h_table = origin_table_names.index(h_table)
            return h_table

    return cur_table


def alter_not_in(datas, schemas):
    for idx, d in enumerate(datas):
        if 'Filter(19)' in d['model_result']:
            current_table = schemas[d['db_id']]
            alter_not_in_one_entry(d, current_table)


def alter_not_in_one_entry(data, schema):
    schema['schema_content_clean'] = [x[1] for x in schema['column_names']]
    schema['col_table'] = [col[0] for col in schema['column_names']]
    origin_table_names = [[wordnet_lemmatizer.lemmatize(x.lower()) for x in names.split(' ')] for names in
                          data['table_names']]
    question_arg_type = data['question_arg_type']
    question_arg = data['question_arg']
    pred_label = data['model_result'].split(' ')

    # get potiantial table
    cur_table = None
    for label_id, label_val in enumerate(pred_label):
        if label_val in ['Filter(19)']:
            cur_table = int(pred_label[label_id - 1][2:-1])
            break

    h_table = find_table(cur_table, origin_table_names, question_arg_type, question_arg)

    for label_id, label_val in enumerate(pred_label):
        if label_val in ['Filter(19)']:
            for primary in schema['primary_keys']:
                if int(schema['col_table'][primary]) == int(pred_label[label_id - 1][2:-1]):
                    pred_label[label_id + 2] = 'C(' + str(
                        data['col_set'].index(schema['schema_content_clean'][primary])) + ')'
                    break
            for pair in schema['foreign_keys']:
                if int(schema['col_table'][pair[0]]) == h_table and data['col_set'].index(
                        schema['schema_content_clean'][pair[1]]) == int(pred_label[label_id + 2][2:-1]):
                    pred_label[label_id + 8] = 'C(' + str(
                        data['col_set'].index(schema['schema_content_clean'][pair[0]])) + ')'
                    pred_label[label_id + 9] = 'T(' + str(h_table) + ')'
                    break
                elif int(schema['col_table'][pair[1]]) == h_table and data['col_set'].index(
                        schema['schema_content_clean'][pair[0]]) == int(pred_label[label_id + 2][2:-1]):
                    pred_label[label_id + 8] = 'C(' + str(
                        data['col_set'].index(schema['schema_content_clean'][pair[1]])) + ')'
                    pred_label[label_id + 9] = 'T(' + str(h_table) + ')'
                    break
            pred_label[label_id + 3] = pred_label[label_id - 1]

        data['model_result'] = " ".join(pred_label)


def alter_inter(datas):
    for d in datas:
        if 'Filter(0)' in d['model_result']:
            alter_inter_one_entry(d)


def alter_inter_one_entry(data):
    if 'Filter(0)' in data['model_result']:
        now_result = data['model_result'].split(' ')
        index = now_result.index('Filter(0)')
        c1 = None
        c2 = None
        for i in range(index + 1, len(now_result)):
            if c1 is None and 'C(' in now_result[i]:
                c1 = now_result[i]
            elif c1 is not None and c2 is None and 'C(' in now_result[i]:
                c2 = now_result[i]

        if c1 != c2 or c1 is None or c2 is None:
            return
        replace_result = ['Root1(0)'] + now_result[1:now_result.index('Filter(0)')]
        for r_id, r_val in enumerate(now_result[now_result.index('Filter(0)') + 2:]):
            if 'Filter' in r_val:
                break

        replace_result = replace_result + now_result[now_result.index('Filter(0)') + 1:r_id + now_result.index(
            'Filter(0)') + 2]
        replace_result = replace_result + now_result[1:now_result.index('Filter(0)')]

        replace_result = replace_result + now_result[r_id + now_result.index('Filter(0)') + 2:]
        replace_result = " ".join(replace_result)
        data['model_result'] = replace_result


def alter_column0(datas):
    """
    Attach column * table
    :return: model_result_replace
    """

    for data in datas:
        alter_column0_one_entry(data)


def alter_column0_one_entry(data):
    if 'rule_count' in data:
        str_replace = 'C(0) T(' + str(data['rule_count']) + ')'
        replace_result = regex.sub('C\(0\) T\(.\)', str_replace, data['model_result'])
        data['model_result_replace'] = replace_result
    else:
        data['model_result_replace'] = data['model_result']

