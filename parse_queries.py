import os
import sys
import json
from nltk import word_tokenize

from process_sql import get_sql, tokenize


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
        column_names_original = table["column_names_original"]
        table_names_original = table["table_names_original"]
        # print 'column_names_original: ', column_names_original
        # print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {"*": i}
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
    db_names = [db["db_id"] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db["db_id"]
        schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db["column_names_original"]
        table_names_original = db["table_names_original"]
        tables[db_id] = {
            "column_names_original": column_names_original,
            "table_names_original": table_names_original,
        }
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables


def tokenize_question(string):
    return word_tokenize(string)


def parse_one_query(schema, db_name, table, sql, question, query_index):
    sql_dict = {}
    sql_dict["db_id"] = db_name
    sql_dict["query"] = sql
    sql_dict["query_toks"] = tokenize(sql, has_value=True)
    sql_dict["query_toks_no_value"] = tokenize(sql, has_value=False)
    sql_dict["question"] = question
    sql_dict["question_toks"] = tokenize_question(question)
    try:
        sql_dict["sql"] = get_sql(schema, sql, table)
    except Exception as e:
        print("{}...NL:{}...SQL:{}".format(e, question, sql))
        return None
    sql_dict["index"] = query_index
    return sql_dict


def parse_queries(table_file, tsv_file, header=True):
    schemas, db_names, tables = get_schemas_from_json(table_file)

    lines = []
    with open(tsv_file) as f:
        if header:
            _ = f.readline()
        for line in f.readlines():
            db_id, query_index, _, question, _, sql1, sql2 = line.strip().split("\t")
            if db_id == "advising-questionsplit":
                db_id = "advising"
            if db_id == "advising-querysplit":
                db_id = "advising"
            sql2 = sql2.replace("=", " = ")
            sql2 = sql2.replace("<", " < ")
            sql2 = sql2.replace(">", " > ")
            sql2 = sql2.replace(";", "")
            if " '" in sql2:
                sql2 = sql2.replace("'", '"')
            lines.append((db_id.lower(), query_index, sql2, question))

    queries = []
    gold_sqls = []
    for db_name, query_index, sql, question in lines:
        if db_name.startswith("wiki"):
            words = [word.lower() for word in sql.split()]
            for i in range(1, len(words)):
                if words[i - 1] == "from" and words[i].startswith("table"):
                    db_name = words[i]
                    break
        # print(sql)
        schema = schemas[db_name]
        table = tables[db_name]
        # print(db_name, schema, table)
        schema = Schema(schema, table)
        query_dict = parse_one_query(schema, db_name, table, sql, question, query_index)
        if query_dict is None:
            # queries.append(None)
            continue
        gold_sqls.append(sql)
        queries.append(query_dict)
    return queries, gold_sqls


def write_queries(queries, gold_sqls, outfile, gold_sql_outfile):
    with open(outfile, "w") as f:
        json.dump(queries, f, sort_keys=True, indent=4, separators=(",", ": "))

    with open(gold_sql_outfile, "w") as f:
        for gold_sql in gold_sqls:
            f.write(gold_sql + "\n")
    print("Write {} queries to {}".format(len(queries), outfile))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "How to run: python parse_queries.py {input table file (https://github.com/taoyds/spider/preprocess/get_table.py)} {input tsv file (db_id, query_index, _, question, _, _, sql)} {output json file} {output gold sql file}"
        )
        sys.exit(-1)
    table_file = sys.argv[1]
    tsv_file = sys.argv[2]
    outfile = sys.argv[3]
    gold_sql_outfile = sys.argv[4]
    queries, gold_sqls = parse_queries(table_file, tsv_file)
    write_queries(queries, gold_sqls, outfile, gold_sql_outfile)
