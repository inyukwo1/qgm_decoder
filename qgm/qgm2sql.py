import json
import argparse


def qgm2sql(qgm, schema):
    pass

    return None

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
    tmp = [qgm2sql(data, db_dic[data['db_id']]) for data in datas]

    print('Translating complete!')

    # Save
    with open(args.destin, 'wb') as f:
        json.dump(f, tmp)

    print('Saving QGM complete!')