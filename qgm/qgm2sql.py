import json
import argparse
from ops import WHERE_OPS, AGG_OPS, BOX_OPS


def decode_qgm_box(boxes, schema):
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
            order_box = box

    # Construct from clause
    tables = []
    alias = {}
    for idx in range(len(select_box['body']['quantifiers'])):
        quantifier = select_box['body']['quantifiers'][idx]
        quantifier_type = select_box['body']['quantifier_types'][idx]

        if quantifier_type == 'f':
            table_name = schema['table_names'][quantifier]
            tables += [table_name]
            alias[table_name] = len(alias) + 1
    if not tables:
        assert len(select_box['body']['quantifiers']) == 1
        tables += [decode_qgm_box(select_box['body']['quantifiers'])]

    join_conditions = []
    if len(tables) > 1:
        for idx in range(len(select_box['body']['join_predicates'])):
            table1_name = select_box['body']['join_predicates'][idx]

            pass

    # Construct where clause
    # get all local predicates and join predeicates with quantifier type s

    # construct group by clause
    # if box with group by operator

    # Construct having clause
    # if predicates in the box with group by operator

    # Construct order by clause
    # if box with order by operator

    # Construct select clause
    # read in head of box with operator select

    # Intersect, union, except
    # read in head of box with operator other than select

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
        q1 = decode_qgm_box(qgm[:s_idx], schema)
        q2 = decode_qgm_box(qgm[s_idx:], schema)
        query = q1 + key.upper() + q2
    else:
        query = decode_qgm_box(qgm, schema)

    # Debugging
    print(query == info['query'])

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
        json.dump(tmp, f)

    print('Saving QGM complete!')


