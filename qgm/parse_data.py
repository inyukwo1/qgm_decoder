import json
import argparse
from process_sql import get_sql

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='spider', help='Among spider, patient, wikisql, wikitablequestions')
parser.add_argument('--data_type', type=str, default='train', help='Either dev or train')
args = parser.parse_args()

data_path = 'data/{}/'.format(args.dataset_name)
db_dir = data_path + 'database/'
table_file = data_path + 'tables.json'

sql_path = data_path + '{}_original.json'.format(args.data_type)
output_file = data_path + '{}.json'.format(args.data_type)

class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        # print 'column_names_original: ', column_names_original
        # print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap


def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables


schemas, db_names, tables = get_schemas_from_json(table_file)

with open(sql_path) as inf:
    sql_data = json.load(inf)

sql_data_new = []
for idx, data in enumerate(sql_data):
    try:
        db_id = data["db_id"]
        schema = schemas[db_id]
        table = tables[db_id]
        schema = Schema(schema, table)
        sql = data["query"]
        sql_label = get_sql(schema, sql)
        data["sql"] = sql_label
        sql_data_new.append(data)
    except Exception as e:
        print(e)
        print('Skipping idx: ', idx)
        print("db_id: ", db_id)
        print("sql: ", sql)

with open(output_file, 'wt') as out:
    json.dump(sql_data_new, out, sort_keys=True, indent=4, separators=(',', ': '))

print('Saving as {}'.format(output_file))