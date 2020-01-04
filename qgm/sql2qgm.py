import json
import argparse
from process_sql import get_sql

WHERE_OPS = ['not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists']
AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']


def sql2qgm(iidx, sql, info, schema):
    '''
    box unit:
        1. table
        2. operator
        3. box:
            3-1. head
            3-2. body
    box: {'head', 'body', 'operator'}
    box_body: {'quantifiers', 'quantifier_types', 'join_predicates', 'local_predicates'}
    box_head: []
    '''

    # Choose Box Operator
    if info['intersect']:
        box_operator = 'intersect'
    elif info['union']:
        box_operator = 'union'
    elif info['except']:
        box_operator = 'except'
    else:
        box_operator = 'select'

    quantifiers = []
    quantifier_types = []
    local_predicates = []
    join_predicates = []

    # Create Box - body - quantifiers
    for table_unit in info['from']['table_units']:
        assert table_unit[0] in ['sql', 'table_unit']
        if 'sql' == table_unit[0]:
            something = sql2qgm(iidx, sql, table_unit[1], schema)
            quantifiers += [something]
            quantifier_types += ['s']
        else:
            table_id = table_unit[1]
            table_name = schema['table_names'][table_id]
            quantifiers += [table_id]
            quantifier_types += ['f']


    # Create Box - body - join predicates
    for conds_unit in info['from']['conds']:
        something = conds_unit
        # 1
        col1_id = something[2][1][1]
        table_id = schema['column_names'][col1_id][0]
        table_name = schema['table_names'][table_id]
        try:
            quantifier1_id = quantifiers.index(table_id)
        except:
            print(iidx)
            return []

        # 2
        col2_id = something[3][1]
        table_id = schema['column_names'][col2_id][0]
        table_name = schema['table_names'][table_id]
        try:
            quantifier2_id = quantifiers.index(table_id)
        except:
            print(iidx)
            stop = 1

        join_predicates += [((quantifier1_id, col1_id), WHERE_OPS.index('='), (quantifier2_id, col2_id))]

    for idx in range(0, len(info['where']), 2):
        where_unit = info['where'][idx]

        # Operand1
        col_id = where_unit[2][1][1]
        table_id = schema['column_names'][col_id][0]
        table_name = schema['table_names'][table_id]
        try:
            quantifier_id = quantifiers.index(table_id)
        except:
            print(iidx)
            return []

        # Operator
        operator = where_unit[1]

        # Operand2
        if isinstance(where_unit[3], dict):
            right_operand = sql2qgm(iidx, sql, where_unit[3], schema)
            quantifiers += [right_operand]
            quantifier_types += ['s']
        else:
            right_operand = where_unit[3]

        local_predicates += [(quantifier_id, operator, right_operand)]

    heads = []
    # Create Box - head - select agg and columns
    for select_unit in info['select'][1]:
        agg = select_unit[0]
        col_id = select_unit[1][1][1]
        heads += [(agg, col_id)]

    # Create Box1
    body = {'quantifiers': quantifiers, 'quantifier_types': quantifier_types,
                'join_predicates': join_predicates, 'local_predicates': local_predicates}
    box1 = {'head': heads, 'body': body, 'operator': box_operator}

    boxes = [box1]
    # Create Box2
    if box_operator != 'select':
        box2 = sql2qgm(iidx, sql, info[box_operator], schema)
        boxes += box2

    return boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=None, type=str, help='db tables path')
    parser.add_argument('--source', default=None, type=str, help='loading path')
    parser.add_argument('--destin', default=None, type=str, help='saving path')
    args = parser.parse_args()

    # Load DB info
    dbs = json.load(open(args.db, 'rb'))
    db_dic = {}
    for db in dbs:
        db_dic[db['db_id']] = db

    # Load data
    datas = json.load(open(args.source, 'rb'))

    print('Loading complete!')

    # Convert
    for idx, data in enumerate(datas):
        tmp = sql2qgm(idx, data['query'], data['sql'], db_dic[data['db_id']])
        data['qgm'] = tmp

    print('Translating complete!')

    # Save
    with open(args.destin, 'w') as f:
        json.dump(datas, f)

    print('Saving QGM complete!')
