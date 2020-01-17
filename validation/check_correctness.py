import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from collections import Counter
from itertools import permutations, product
from validation.parser.myParser import alias_sql
from validation.canonicaliser_after_alias import read_schema, make_canonical, tokenize_query
from validation.string_match import is_equal_query, timeout
import difflib

def compute_ex(dbfile, gen_sql, gold_sql):
    sqlite_db = dbfile
    disk_engine = create_engine(sqlite_db)
    try:
        gen_output = pd.read_sql_query(gen_sql, disk_engine)
    except:
        return False, "Not executable generated query"

    try:
        gold_output = pd.read_sql_query(gold_sql, disk_engine)
    except:
        return False, "Not executable gold query"
    ordered_result = 'order by' in gen_sql.lower() or 'order by' in gold_sql.lower()
    gold_output = tuple(tuple(row) for row in gold_output.values)
    gen_output = tuple(tuple(row) for row in gen_output.values)
    def pack_column(table, col_index):
       return list(zip(*table))[col_index]

    accurate_ex = True
    if len(gold_output) == 0 and len(gen_output) == 0:
        return True, "DONE"
    elif len(gold_output) == 0 or len(gen_output) == 0:
        return False, "DONE"
    elif ordered_result:
        gold_cols=tuple(tuple(pack_column(gold_output, i) for i in range(len(gold_output[0]))))
        gen_cols=tuple(tuple(pack_column(gen_output, i) for i in range(len(gen_output[0]))))
        accurate_ex = True if Counter(gold_cols) == Counter(gen_cols) else False
    else:
        gold_cols=tuple(pack_column(gold_output, i) for i in range(len(gold_output[0])))
        gen_cols=tuple(pack_column(gen_output, i) for i in range(len(gen_output[0])))
        if len(gold_cols) != len(gen_cols) or len(gold_output) != len(gen_output):
            accurate_ex = False
        elif len(gold_cols) == 0:
            accurate_ex = True
        else:
            row_maps=[[] for i in range(len(gold_cols))]
            for i in range(len(gen_cols)):
                for j in range(len(gold_cols)):
                    if Counter(gen_cols[i]) == Counter(gold_cols[j]):
                        row_maps[j].append(i)
            permus=[ permu for permu in permutations([i for i in range(len(gold_cols))], len(gold_cols)) ]
            mappings=list(mapping for mapping in product(*row_maps) if mapping in permus)
            accurate_ex = False
            for mapping in mappings:
                gen_cols_permus=tuple(gen_cols[i] for i in mapping)
                gen_output_permus=tuple(pack_column(gen_cols_permus, i) for i in range(len(gen_cols_permus[0])))
                if Counter(gen_output_permus) == Counter(gold_output):
                    accurate_ex=True
                    break
    return accurate_ex, "DONE"

def extract_fields(dbfile, fname):
    sqlite_db = dbfile
    disk_engine = create_engine(sqlite_db)
    try:
        with open(fname, 'w') as f:
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", disk_engine)
            for table_name in tables.values:
                table_name = table_name[0]
                columns = pd.read_sql_query("PRAGMA table_info('%s')" % table_name, disk_engine)
                for column_name in columns.values:
                    f.write('{} {}\n'.format(table_name, column_name[1]))
    except:
        return False
    return True

def compute_string_match(dbfile, dbname, gen_sql, gold_sql):
    field_file=dbname+'.fields'

    if not extract_fields(dbfile, field_file):
        return False, "Cannot access sqlite3 file %s" % dbfile

    accurate_str=False
    try:
        gen_sql=gen_sql.replace("<>", "!=")
        gold_sql=gold_sql.replace("<>", "!=")
        gen_sql_alias=alias_sql(gen_sql)
        gold_sql_alias=alias_sql(gold_sql)
        schema = read_schema(field_file)
        gen_canonical = make_canonical(gen_sql_alias, schema, {})
        gold_canonical = make_canonical(gold_sql_alias, schema, {})

        accurate_str = timeout(is_equal_query, args=(gen_canonical, gold_canonical, schema), timeout_duration=30, default=0)
    except Exception as e:
        return False, e

    print(gen_canonical, gold_canonical)
    return accurate_str, "DONE"

def diff_two_queries(dbfile, dbname, gen_sql, gold_sql, diffmode):
    if diffmode == 'canonical':
        try:
            gen_sql=gen_sql.replace("<>", "!=")
            gold_sql=gold_sql.replace("<>", "!=")
            field_file=dbname+'.fields'
            if not extract_fields(dbfile, field_file):
                return False, "Cannot access sqlite3 file %s" % dbfile
            schema = read_schema(field_file)
            gen_sql_canonical=make_canonical(alias_sql(gen_sql), schema, {})
            gold_sql_canonical=make_canonical(alias_sql(gold_sql), schema, {})
        except:
            gen_sql_canonical=gen_sql
            gold_sql_canonical=gold_sql
        gen_sql=gen_sql_canonical
        gold_sql=gold_sql_canonical

    gen_sql_tok=tokenize_query(gen_sql)
    gold_sql_tok=tokenize_query(gold_sql)

    s=difflib.SequenceMatcher(None, gen_sql_tok, gold_sql_tok)
    opcodes = s.get_opcodes()
    # [('equal', 0, 5, 0, 5), ('replace', 5, 6, 5, 6), ('equal', 6, 21, 6, 21), ('insert', 21, 21, 21, 22)]
    diff=[]
    similarity = len(gen_sql_tok) + len(gold_sql_tok)
    for opcode, st_gen, ed_gen, st_gold, ed_gold in opcodes:
        if opcode in ('replace', 'delete', 'insert'):
            similarity = similarity - ((ed_gen - st_gen) + (ed_gold - st_gold))
            char_st_gen=0
            char_ed_gen=0
            for i in range(st_gen):
                if i == st_gen-1:
                    char_st_gen=char_st_gen + len(gen_sql_tok[i])
                else:
                    char_st_gen=char_st_gen + len(gen_sql_tok[i]) + 1
            for i in range(ed_gen):
                char_ed_gen=char_ed_gen + len(gen_sql_tok[i]) + 1
            diff.append([char_st_gen, char_ed_gen])
    similarity = similarity * 100.0 / (len(gen_sql_tok) + len(gold_sql_tok))
    return diff, gen_sql, gold_sql, similarity

def compare_two_queries(dbfile, dbname, gen_sql, gold_sql):
    equivalence, log=compute_ex(dbfile, gen_sql, gold_sql)
    if equivalence:
       equivalencey, log=compute_string_match(dbfile, dbname, gen_sql, gold_sql)
    return equivalence, log

if __name__ == '__main__':
    gold_sql = 'SELECT T1.title FROM movie AS T1 JOIN director AS T2 WHERE T2.name = "Jim Jarmusch" ORDER BY T1.release_year DESC LIMIT 1'
    gen_sql = "SELECT keyword.keyword FROM keyword order by keyword.id limit 1"
    gold_sql = "SELECT count(keyword.keyword) FROM keyword order by keyword.id limit 1"
   # gold_sql = "SELECT keyword.keyword FROM keyword order by keyword.id limit 1"
   # print( compute_ex("sqlite:///imdb.sqlite", gen_sql, gold_sql )  )
  #  print( compute_string_match("sqlite:///imdb.sqlite", "imdb", gen_sql, gold_sql )  )
    print( diff_two_queries("sqlite:///imdb.sqlite", "imdb", gen_sql, gold_sql, 'origin')  )
    print( diff_two_queries("sqlite:///imdb.sqlite", "imdb", gen_sql, gold_sql, 'canonical')  )
#    gold_sql = "SELECT count(T1.title) FROM movie AS T1 JOIN actor AS T2 WHERE T2.name <> 'Daffy Duck'"
   # print( diff_two_queries("sqlite:///imdb.sqlite", "imdb", gen_sql, gold_sql, 'canonical')  )
