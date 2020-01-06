import json
import argparse
from ops import WHERE_OPS, AGG_OPS, BOX_OPS


def decode_qgm_box(iidx, info, boxes, schema):
    # Construct from clause
    # get quantifiers with type f
    # get quantifiers with type s and has no predicate
    # get all the join predicates between quantifiers with type f
    # assumption is that if there is a quantifier with type s for the from clause, there are no other quantifiers in the box for the from clause.
    # set dictioanry for alias

    select_box = group_box = order_box = None
    for box in boxes:
        if box['operator'] == BOX_OPS.index('select'):
            select_box = box
        elif box['operator'] == BOX_OPS.index('groupBy'):
            group_box = box
        elif box['operator'] in [BOX_OPS.index(key) for key in ['orderByDesc', 'orderByAsc']]:
            order_by_direction = 'DESC' if box['operator'] == BOX_OPS.index('orderByDesc') else 'ASC'
            order_box = box

    if iidx == 211:
        stop = 1

    # Construct from clause
    query_from = 'FROM '
    alias = {}
    for idx in range(len(select_box['body']['quantifiers'])):
        quantifier = select_box['body']['quantifiers'][idx]
        quantifier_type = select_box['body']['quantifier_types'][idx]

        if quantifier_type == 'f':
            table_name = schema['table_names_original'][quantifier]
            alias[table_name] = len(alias) + 1
            if query_from != 'FROM ':
                query_from += ' JOIN '
            query_from += table_name + ' AS T{}'.format(str(alias[table_name]))
    if query_from == 'FROM ':
        assert len(select_box['body']['quantifiers']) == 1
        query_from += '(' + str([decode_qgm_box(iidx, info, select_box['body']['quantifiers'][0], schema)]) + ')'

    if len(alias) > 1:
        for idx in range(len(select_box['body']['join_predicates'])):
            col1_id, operator_id, col2_id = select_box['body']['join_predicates'][idx]
            table1_id, col1_name = schema['column_names_original'][col1_id]
            table2_id, col2_name = schema['column_names_original'][col2_id]
            table1_name = schema['table_names_original'][table1_id]
            table2_name = schema['table_names_original'][table2_id]
            alias1_id = alias[table1_name]
            alias2_id = alias[table2_name]
            query_from_cond = ' ON T{}.{} = T{}.{}'.format(alias1_id, col1_name, alias2_id, col2_name)
            key = max(alias1_id, alias2_id)
            try:
                split_idx = query_from.index('T{}'.format(key)) + len('T{}'.format(key))
            except:
                stop = 1
            query_from = query_from[:split_idx] + query_from_cond + query_from[split_idx:]

    query_where = ''
    # Construct where clause
    # get all local predicates and join predicates with quantifier type s
    for idx in range(len(select_box['body']['local_predicates'])):
        agg_id, col_id, operator_id, right_operand = select_box['body']['local_predicates'][idx]
        if isinstance(right_operand, list):
            right_operand = '(' + str(decode_qgm_box(iidx, info, right_operand, schema)) + ')'
        table_id, col_name = schema['column_names_original'][col_id]
        table_name = schema['table_names_original'][table_id]
        alias_id = alias[table_name]
        operator = WHERE_OPS[operator_id]
        query_where = query_where + ' AND ' if query_where else 'WHERE'
        query_where += ' T{}.{} {} {}'.format(alias_id, col_name, operator, right_operand)

    if query_where:
        stop = 1

    # construct group by clause
    # if box with group by operator
    # head -> group by
    query_group_by = ''
    query_having = ''
    if group_box:
        for head in group_box['head']:
            agg_id, col_id = head
            agg = AGG_OPS[agg_id]
            table_id, col_name = schema['column_names_original'][col_id]
            table_name = schema['table_names_original'][table_id]

            if table_id == -1:
                agg_col = col_name if agg == 'none' else '{}({})'.format(agg, col_name)
            else:
                alias_id = alias[table_name]
                agg_col = 'T{}.{}'.format(alias_id, col_name) if agg == 'none' else '{}(T{}.{})'.format(agg, alias_id, col_name)
            query_group_by += ', {}'.format(agg_col) if query_group_by else 'GROUP BY {}'.format(agg_col)

        # local_predicates -> having
        for local_predicate in group_box['body']['local_predicates']:
            agg_id, col_id, operator, right_operand = local_predicate
            agg = AGG_OPS[agg_id]
            table_id, col_name = schema['column_names_original'][col_id]
            table_name = schema['table_names_original'][table_id]

            if table_id == -1:
                agg_col = col_name if agg == 'none' else '{}({})'.format(agg, col_name)
            else:
                alias_id = alias[table_name]
                agg_col = 'T{}.{}'.format(alias_id, col_name) if agg == 'none' else '{}(T{}.{})'.format(agg, alias_id, col_name)

            query_having += ', {}'.format(agg_col) if query_having else 'HAVING {}'.format(agg_col)

    if query_group_by:
        stop = 1

    if query_having:
        stop = 1

    # Construct order by clause
    # if box with order by operator
    query_order_by = ''
    if order_box:
        assert len(order_box['head']) == 1
        assert len(order_box['body']['quantifier_types'][0]) == 1
        assert order_box['body']['quantifier_types'][0] == 'f'
        agg_id, col_id = order_box['head'][0]
        agg = AGG_OPS[agg_id]
        table_id, col_name = schema['column_names_original'][col_id]
        table_name = schema['table_names_original'][table_id]

        if table_id == -1:
            agg_col = col_name if agg == 'none' else '{}({})'.format(agg, col_name)
        else:
            alias_id = alias[table_name]
            agg_col = 'T{}.{}'.format(alias_id, col_name) if agg == 'none' else '{}(T{}.{})'.format(agg, alias_id, col_name)
        query_order_by = 'ORDER BY {} {}'.format(agg_col, order_by_direction)

    # Construct select clause
    # read in head of box with operator select
    query_select = ''
    for idx in range(len(select_box['head'])):
        agg_id, col_id = select_box['head'][idx]
        agg = AGG_OPS[agg_id]
        table_id, col_name = schema['column_names_original'][col_id]
        table_name = schema['table_names_original'][table_id]

        if table_id == -1:
            agg_col = col_name if agg == 'none' else '{}({})'.format(agg, col_name)
        else:
            alias_id = alias[table_name]
            agg_col = 'T{}.{}'.format(alias_id, col_name) if agg == 'none' else '{}(T{}.{})'.format(agg, alias_id, col_name)
        query_select += ', {}'.format(agg_col) if query_group_by else 'SELECT {}'.format(agg_col)

    return None


def qgm2sql(iidx, info, qgm, schema):
    print(iidx)
    key = None
    s_idx = None
    for idx, box in enumerate(qgm):
        if box['operator'] in [BOX_OPS.index(key) for key in ['intersect', 'union', 'except']]:
            s_idx = idx
            key = BOX_OPS[box['operator']]
    if s_idx:
        q1 = decode_qgm_box(iidx, info, qgm[:s_idx], schema)
        q2 = decode_qgm_box(iidx, info, qgm[s_idx:], schema)
        query = q1 + key.upper() + q2
    else:
        query = decode_qgm_box(iidx, info, qgm, schema)

    return query


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=None, type=str, help='db tables path')
    parser.add_argument('--source', default=None, type=str, help='loading path')
    parser.add_argument('--destin', default=None, type=str, help='saving path')
    args = parser.parse_args()

    # Load DB info
    dbs = json.load(open(args.db, 'rb'))
    db_dic = {db['db_id']: db for db in dbs}

    # Load data
    datas = json.load(open(args.source, 'rb'))

    print('Loading complete!')

    # Convert
    tmp = [qgm2sql(idx, data, data['qgm'], db_dic[data['db_id']]) for idx, data in enumerate(datas)]

    print('Translating complete!')

    # Save
    with open(args.destin, 'w') as f:
        for item in tmp:
            f.write(item + '\n')

    print('Saving QGM complete!')
