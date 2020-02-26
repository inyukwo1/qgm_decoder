import os
import sqlite3

import json
from gnn.semparse.worlds.evaluate import (
    Evaluator,
    build_valid_col_units,
    rebuild_sql_val,
    rebuild_sql_col,
    build_foreign_key_map_from_json,
)
from gnn.spider_evaluation.process_sql import Schema, get_schema, get_sql
import logging

_schemas = {}
kmaps = None
tables_JSON = None
logger = logging.getLogger(__name__)


def evaluate(gold, predict, db_name, db_dir, table, check_valid: bool = True) -> bool:
    global kmaps
    global tables_JSON
    # try:
    evaluator = Evaluator()

    if kmaps is None:
        kmaps = build_foreign_key_map_from_json(table)
    if tables_JSON is None:
        with open(table) as f:
            data = json.load(f)
        tables_JSON = {}
        for db in data:
            db_id = db["db_id"]
            column_names_original = db["column_names_original"]
            table_names_original = db["table_names_original"]
            tables_JSON[db_id] = {
                "column_names_original": column_names_original,
                "table_names_original": table_names_original,
            }

    if db_name in _schemas:
        schema = _schemas[db_name]
    else:
        db = os.path.join(db_dir, db_name, db_name + ".sqlite")
        schema = _schemas[db_name] = Schema(get_schema(db))
    try:
        g_sql = get_sql(schema, gold, tables_JSON[db_name])
    except Exception as e:
        return False
    try:
        p_sql = get_sql(schema, predict, tables_JSON[db_name])
    except Exception as e:
        return False

    # rebuild sql for value evaluation
    kmap = kmaps[db_name]
    g_valid_col_units = build_valid_col_units(g_sql["from"]["table_units"], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    p_valid_col_units = build_valid_col_units(p_sql["from"]["table_units"], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

    exact_score = evaluator.eval_exact_match(p_sql, g_sql)

    if not check_valid:
        return exact_score
    else:
        return exact_score and check_valid_sql(predict, db_name, db_dir)
    # except Exception as e:
    #     return 0


_conns = {}


def check_valid_sql(sql, db_name, db_dir, return_error=False):
    db = os.path.join(db_dir, db_name, db_name + ".sqlite")

    if db_name == "wta_1":
        # TODO: seems like there is a problem with this dataset - slow response - add limit 1
        return True if not return_error else (True, None)

    if db_name not in _conns:
        _conns[db_name] = sqlite3.connect(db)

        # fixes an encoding bug
        _conns[db_name].text_factory = bytes

    conn = _conns[db_name]
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        cursor.fetchall()
        return True if not return_error else (True, None)
    except Exception as e:
        return False if not return_error else (False, e.args[0])
