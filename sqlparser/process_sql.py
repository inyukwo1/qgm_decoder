################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json

import sqlite3
from nltk import word_tokenize
import argparse
import copy
import re

# reload(sys)
# sys.setdefaultencoding('utf8')

HOST = "localhost"
USER = "root"
PASSWD = "root"

CLAUSE_KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
)
JOIN_KEYWORDS = ("join", "on", "as", "inner")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")

FLOAT_MATCH = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$").match

STR_VALUES_START_WITH = ('"', "'", "(")

PARENTHESIZE_FLAG = True  # for irnet
# PARENTHESIZE_FLAG = False #for gnn


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {"*": "__all__", "1": "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = (
                    "__" + key.lower() + "." + val.lower() + "__"
                )
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


# def get_schema_mysql(dbname):
#    """
#    Get database's schema, which is a dict with table name as key
#    and list of column names as value
#    :param db: database path
#    :return: schema dict
#    """
#
#    schema = {}
#    with closing(MySQLdb.connect(host=HOST, user=USER, passwd=PASSWD, charset='utf8', local_infile=1)) as dbconn:
#       try:
#          dbconn.begin()
#          with closing(dbconn.cursor()) as cu:
#            cu.execute('''USE {}'''.format(dbname))
#            cu.execute('''SHOW TABLES''')
#            results=cu.fetchall()
# fetch table names
#            tables = [str(table[0].lower()) for table in results]

# fetch table info
#            for table in tables:
#               cu.execute("SHOW columns FROM {}".format(table))
#               results=cu.fetchall()
#               schema[table] = [str(col[0].lower()) for col in results]
#          dbconn.commit()
#       except:
#          print("ERROR durring fetch the schema of {} from mySQL".format(dbname))
#          try:
#             dbconn.rollback()
#          except:
#             pass
#    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry["table"].lower())
        cols = [str(col["column_name"].lower()) for col in entry["col_data"]]
        schema[table] = cols

    return schema


def in_quote(sent, quotes):
    quote = None
    mark = ""
    prev_char = ""
    for c in sent:
        if c in quotes:
            if c == quote and prev_char != "\\":
                quote = None
            elif quote is None and prev_char != "\\":
                quote = c
        if quote is None:
            mark += "0"
        else:
            mark += "1"
        prev_char = c
    return mark


def get_quote_idxs(string, quotes="\"'"):
    quote_marks = in_quote(string, quotes)
    prev_char = "0"
    idxs = []
    for idx, char in enumerate(quote_marks):
        if char == "1" and prev_char == "0":
            idxs.append(idx)
        if char == "0" and prev_char == "1":
            idxs.append(idx)
        prev_char = char
    return idxs


def tokenize(string, has_value=True):
    string = str(string)
    # string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    # quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    quote_idxs = get_quote_idxs(string)
    assert len(quote_idxs) % 2 == 0, (
        "Unexpected quote " + string + ", " + " ".join(str(x) for x in quote_idxs)
    )

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        val = string[qidx1 : qidx2 + 1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2 + 1 :]
        vals[key] = val

    toks = [word for word in word_tokenize(string)]
    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ("!", ">", "<")
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[: eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1 :]

    # find if there exists <> -> transform '!="
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == ">"]
    eq_idxs.reverse()
    prefix = "<"
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[: eq_idx - 1] + ["!="] + toks[eq_idx + 1 :]

    toks_no_value = [word for word in toks]
    # replace with string value token
    for i in range(len(toks)):
        toks[i] = toks[i].lower()
        if toks[i] in vals:
            toks[i] = vals[toks[i]]
    # replace with string "value" token
    for i in range(len(toks_no_value)):
        if toks_no_value[i] in vals:
            toks_no_value[i] = "value"
        elif (
            bool(FLOAT_MATCH(toks_no_value[i]))
            and i != 0
            and toks_no_value[i - 1] in WHERE_OPS
        ):
            toks_no_value[i] = "value"

    if has_value:
        return toks
    return toks_no_value


def scan_alias(toks, tables):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == "as"]
    alias = {}
    for idx in as_idxs:
        alias[toks[idx + 1]] = toks[idx - 1]

    table_idxs = [idx for idx, tok in enumerate(toks) if tok in tables]
    special_tokens = [",", ".", '"', " ", "'", "`", "join", "on", "as", ")", "inner"]
    toks_as = list(toks)
    count = 0
    for idx in table_idxs:
        if (
            idx < len(toks) - 1
            and toks[idx + 1] not in special_tokens
            and toks[idx + 1] not in CLAUSE_KEYWORDS
            and toks[idx + 1] not in WHERE_OPS
            and toks[idx + 1] not in UNIT_OPS
            and toks[idx + 1] not in AGG_OPS
            and toks[idx + 1] not in COND_OPS
            and toks[idx + 1] not in ORDER_OPS
        ):
            toks_as.insert(idx + count + 1, "as")
            count += 1
            alias[toks[idx + 1]] = toks[idx]
    # print("Alias..", alias, toks, tables)
    return toks_as, alias


def get_tables_with_alias(schema, toks, tables):
    if isinstance(tables, dict) and "table_names_original" in tables:
        tables = tables["table_names_original"]
    toks_as, tables = scan_alias(toks, tables)
    # print(toks_as, tables)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return toks_as, tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    # print(toks[start_idx:])
    if tok == "*" or tok == "1":
        return start_idx + 1, schema.idMap[tok]
    if "." in tok:  # if token is a composite
        alias, col = tok.split(".")
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema.idMap[key]

    assert (
        default_tables is not None and len(default_tables) > 0
    ), "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx + 1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == "("
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ")"
        idx += 1
        #        if idx < len(toks)-1 and toks[idx].lower() == 'as':
        #           next_idx=idx+2
        #           for cand_idx, cand_tok in enumerate(toks[idx+2:]):
        #               if cand_tok in (',') or cand_tok in CLAUSE_KEYWORDS or cand_tok in JOIN_KEYWORDS or cand_tok in UNIT_OPS or cand_tok in WHERE_OPS or cand_tok in AGG_OPS or cand_tok in COND_OPS or cand_tok in SQL_OPS or cand_tok in ORDER_OPS:
        #                  next_idx=cand_idx
        #                  break
        #           print(start_idx, next_idx)
        #           idx=next_idx
        return idx, (agg_id, col_id, isDistinct)
    else:
        agg_id = AGG_OPS.index("none")
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'

    #    if idx < len(toks)-1 and toks[idx].lower() == 'as':
    #       next_idx=idx+2
    #       for cand_idx, cand_tok in enumerate(toks[idx+2:]):
    #           if cand_tok in (',') or cand_tok in CLAUSE_KEYWORDS or cand_tok in JOIN_KEYWORDS or cand_tok in UNIT_OPS or cand_tok in WHERE_OPS or cand_tok in AGG_OPS or cand_tok in COND_OPS or cand_tok in SQL_OPS or cand_tok in ORDER_OPS:
    #              next_idx=cand_idx
    #              break
    #       print(start_idx, next_idx)
    #       idx=next_idx
    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index("none")

    idx, col_unit1 = parse_col_unit(
        toks, idx, tables_with_alias, schema, default_tables
    )
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'

    #    if idx < len(toks)-1 and toks[idx].lower() == 'as':
    #        next_idx=idx+2
    #        for cand_idx, cand_tok in enumerate(toks[idx+2:]):
    #            if cand_tok in (',') or cand_tok in CLAUSE_KEYWORDS or cand_tok in JOIN_KEYWORDS or cand_tok in UNIT_OPS or cand_tok in WHERE_OPS or cand_tok in AGG_OPS or cand_tok in COND_OPS or cand_tok in SQL_OPS or cand_tok in ORDER_OPS:
    #               next_idx=cand_idx
    #               break
    #        print(start_idx, next_idx)
    #        idx=next_idx

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] == "select":
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif '"' in toks[idx] or "value" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while (
                end_idx < len_
                and toks[end_idx] != ","
                and toks[end_idx] != ")"
                and toks[end_idx] != "and"
                and toks[end_idx] not in CLAUSE_KEYWORDS
            ):
                end_idx += 1

            idx, val = parse_col_unit(
                toks[start_idx:end_idx], 0, tables_with_alias, schema, default_tables
            )
            idx = end_idx

    if isBlock:
        assert toks[idx] == ")"
        idx += 1

    return idx, val


# def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
#     idx = start_idx
#     len_ = len(toks)
#
#     isBlock = False
#     if toks[idx] == '(':
#         isBlock = True
#         idx += 1
#
#     if toks[idx] == 'select':
#         idx, val = parse_sql(toks, idx, tables_with_alias, schema)
#     elif "\"" in toks[idx]:  # token is a string value
#         val = toks[idx]
#         idx += 1
#     else:
#         end_idx = idx
#         while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
#             and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS:
#             end_idx += 1
#
#         tok = "".join(toks[idx: end_idx])
#         val = tok
#
#         try:
#             idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
#         except:
#             # print "Value is not a column"
#             try:
#                 val = float(val)
#             except:
#                 pass
#                 # print "Value is not a number"
#         idx = end_idx
#
#     if isBlock:
#         assert toks[idx] == ')'
#         idx += 1
#
#     return idx, val


def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    # print("STRING: {}".format(' '.join(toks[idx:])))
    conds = []
    brackets_cnt = 0
    bracket_points = []
    not_points = []
    # initially put in conds -> move some conditions to join_conds
    while idx < len_:
        # print("Go: {}, {}".format(toks[idx:], toks[idx] == '('))
        # brackets
        if PARENTHESIZE_FLAG and toks[idx] in ("("):
            brackets_cnt += 1
            bracket_points.append(idx)
            conds.append("(")
            idx += 1
            continue
        if PARENTHESIZE_FLAG and toks[idx] in (")"):
            if brackets_cnt == 0:
                break

            conds.append(")")
            brackets_cnt -= 1

            bracket_points.pop()
            idx += 1
            continue

        # terminate
        if (
            toks[idx] in CLAUSE_KEYWORDS
            or toks[idx] in (";")
            or toks[idx] in JOIN_KEYWORDS
        ):
            break
        if toks[idx] in (")") and not PARENTHESIZE_FLAG:
            break

        # remove not point: not ( A is not B and ( A is not () ) ) and not A = B
        not_points_res = []
        for not_point in not_points:
            if idx == not_point + 1:
                not_points_res.append(not_point)
                continue
            is_terminated = True
            for bracket_point in bracket_points:
                if not_point + 1 == bracket_point:
                    is_terminated = False
                    break
            if not is_terminated:
                not_points_res.append(not_point)
        not_points = copy.copy(not_points_res)

        if toks[idx] in COND_OPS:
            cond_op_id = COND_OPS.index(toks[idx])
            if (
                len(not_points) % 2 == 1
            ):  # not (A and B) -> (not A or not B), not (A or B) -> (not A and not B)
                if cond_op_id == 0:
                    cond_op_id = 1
                elif cond_op_id == 1:
                    cond_op_id = 0
            conds.append(COND_OPS[cond_op_id])
            idx += 1  # skip and/or
            continue

        if toks[idx] == "not":
            not_points.append(idx)
            idx += 1
            continue

        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )

        if toks[idx] == "not":
            not_points.append(idx)
            idx += 1

        assert (
            idx < len_ and toks[idx] in WHERE_OPS
        ), "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index(
            "between"
        ):  # between..and... special case: dual values
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
            assert toks[idx] == "and"
            idx += 1
            idx, val2 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
        elif (
            op_id == WHERE_OPS.index("in")
            and idx < len_ - 2
            and (
                '"' in toks[idx + 1]
                or is_number_tryexcept(toks[idx + 1])
                or "value" in toks[idx + 1]
            )
        ):
            assert toks[idx] == "("
            idx += 1
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
            if toks[idx] == ")":
                val2 = None
            else:
                assert toks[idx] == ","
                idx += 1
                idx, val2 = parse_value(
                    toks, idx, tables_with_alias, schema, default_tables
                )
                assert toks[idx] == ")"
                idx += 1
        elif (
            toks[idx][0] not in STR_VALUES_START_WITH
            and is_number_tryexcept(toks[idx]) is False
        ):  # (COL) (OP) (COL)
            idx, val1 = parse_val_unit(
                toks, idx, tables_with_alias, schema, default_tables
            )
            val2 = None
        else:  # normal case: single value
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
            val2 = None

        if len(not_points) % 2 == 0:
            not_op = False
        else:
            not_op = True
        conds.append((not_op, op_id, val_unit, val1, val2))

    # D or A and B or C
    # join_conds: (    A   =   B   and  (   B   =   C   )    )  and ( B = C or A = C )
    # bracket_points = [-1, 0, 5, 12]
    # bracket_end_points = [21, 10, 9, 20]
    # or_points = [16] --> or_bracket_id: 3
    # there is no OR at the same or outer..(?) level
    or_points = []
    # algorithm refine plz
    bracket_points = [
        tuple([-1, len(conds)]),
    ]
    cur_bracket_stack = []
    for cond_idx, cond in enumerate(conds):
        if cond == "(":
            cur_bracket_stack.append(cond_idx)
        if cond == "or":
            or_points.append(cond_idx)
        if cond == ")":
            assert len(cur_bracket_stack) > 0
            top = cur_bracket_stack.pop()
            bracket_points.append(tuple([top, cond_idx]))
    bracket_points.sort(key=lambda x: x[0])

    def find_brackets(point, bracket_points):
        bracket_ids = []
        for br_idx in range(len(bracket_points)):
            if point > bracket_points[br_idx][0] and point < bracket_points[br_idx][1]:
                bracket_ids.append(br_idx)
        return bracket_ids

    or_bracket_ids = [
        max(find_brackets(or_point, bracket_points)) for or_point in or_points
    ]
    join_conds = []
    conds_res = []

    conds_print = copy.copy(conds)
    for print_idx in range(len(conds_print)):
        if isinstance(conds_print[print_idx], tuple):
            if isinstance(conds_print[print_idx][3], tuple):
                conds_print[print_idx] = "JOIN_CONDS"
            else:
                conds_print[print_idx] = "CONDS"
    # print("INITIAL_CONDS: {}, bracket_points: {}, or_points: {}, or_bracket_ids: {}".format(conds_print, bracket_points, or_points, or_bracket_ids))
    for cond_idx in range(len(conds)):
        cond = conds[cond_idx]
        bracket_ids = find_brackets(cond_idx, bracket_points)
        has_or = False
        for or_bracket_id in or_bracket_ids:
            if or_bracket_id in bracket_ids:
                has_or = True
                break
        if (
            isinstance(cond, tuple) and isinstance(cond[3], tuple) and not has_or
        ):  # join cond
            join_conds.append(cond)
        else:
            if isinstance(cond, tuple):
                conds_res.append(cond)
            elif cond in (")"):
                assert len(conds_res) > 0
                if not (isinstance(conds_res[-1], tuple) or conds_res[-1] == ")"):
                    conds_res.pop()
                if len(conds_res) > 0 and (
                    isinstance(conds_res[-1], tuple) or conds_res[-1] == ")"
                ):
                    conds_res.append(cond)
            elif cond in COND_OPS:
                if len(conds_res) > 0 and (
                    isinstance(conds_res[-1], tuple) or conds_res[-1] == ")"
                ):
                    conds_res.append(cond)
            else:
                conds_res.append(cond)
    conds = copy.copy(conds_res)

    conds_print = copy.copy(conds)
    for print_idx in range(len(conds_print)):
        if isinstance(conds_print[print_idx], tuple):
            if isinstance(conds_print[print_idx][3], tuple):
                conds_print[print_idx] = "JOIN_CONDS"
            else:
                conds_print[print_idx] = "CONDS"
    # print("CONDS: {}, JOIN_CONDS: {}".format(conds_print, join_conds))
    return idx, conds, join_conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == "select", "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        pre_idx = idx
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        if idx < len_ and toks[idx] in UNIT_OPS:
            agg_id = AGG_OPS.index("none")
            idx, val_unit = parse_val_unit(
                toks, pre_idx, tables_with_alias, schema, default_tables
            )
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert "from" in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index("from", start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == "(":
            isBlock = True
            idx += 1
        if toks[idx] == "select":
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE["sql"], sql))
        elif idx < len_ and toks[idx] == "inner":
            idx += 1  # skip join
        elif idx < len_ and toks[idx] == "join":
            idx += 1  # skip join
        elif idx < len_ and toks[idx] == ",":
            idx += 1  # skip join
        elif idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, _, this_conds = parse_condition(
                toks, idx, tables_with_alias, schema, default_tables
            )
            conds.extend(this_conds)
        else:
            idx, table_unit, table_name = parse_table_unit(
                toks, idx, tables_with_alias, schema
            )
            table_units.append((TABLE_TYPE["table_unit"], table_unit))
            default_tables.append(table_name)

        if isBlock:
            assert toks[idx] == ")"
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != "where":
        return idx, [], []

    idx += 1
    idx, conds, join_conds = parse_condition(
        toks, idx, tables_with_alias, schema, default_tables
    )
    return idx, conds, join_conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != "group":
        return idx, col_units

    idx += 1
    assert toks[idx] == "by"
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = "asc"  # default type is 'asc'

    if idx >= len_ or toks[idx] != "order":
        return idx, val_units

    idx += 1
    assert toks[idx] == "by"
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != "having":
        return idx, []

    idx += 1
    idx, conds, _ = parse_condition(
        toks, idx, tables_with_alias, schema, default_tables
    )
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == "limit":
        idx += 2
        return idx, int(toks[idx - 1])

    return idx, None


def unique_items(data):
    for i in data:
        if data.count(i) >= 2:
            return False
    return True


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False  # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(
        toks, start_idx, tables_with_alias, schema
    )
    assert unique_items(table_units)
    sql["from"] = {"table_units": table_units, "conds": conds}
    # select clause
    _, select_col_units = parse_select(
        toks, start_idx, tables_with_alias, schema, default_tables
    )
    idx = from_end_idx
    sql["select"] = select_col_units
    # where clause
    idx, where_conds, join_conds = parse_where(
        toks, idx, tables_with_alias, schema, default_tables
    )
    sql["where"] = where_conds
    sql["from"]["conds"] = sql["from"]["conds"] + join_conds
    # group by clause
    idx, group_col_units = parse_group_by(
        toks, idx, tables_with_alias, schema, default_tables
    )
    sql["groupBy"] = group_col_units
    # order by clause
    idx, order_col_units = parse_order_by(
        toks, idx, tables_with_alias, schema, default_tables
    )
    sql["orderBy"] = order_col_units
    # having clause
    idx, having_conds = parse_having(
        toks, idx, tables_with_alias, schema, default_tables
    )
    sql["having"] = having_conds
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql["limit"] = limit_val

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    data = []
    with open(fpath) as f:
        line = f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            data.append(line.strip().split("\t"))
    return data


def get_sql(schema, query, tables):
    toks = tokenize(query)
    toks_as, tables_with_alias = get_tables_with_alias(schema.schema, toks, tables)
    # print(tables_with_alias)
    _, sql = parse_sql(toks_as, 0, tables_with_alias, schema)

    return sql


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_SQL_file", help="SQL File")
    parser.add_argument("dbname", help="DB name")
    parser.add_argument(
        "--parenthesize", action="store_true", default=False, dest="parenthesize_switch"
    )
    args = parser.parse_args()
    output_file_name = args.input_SQL_file + ".parsed"
    output_error_file_name = args.input_SQL_file + ".error"
    schema_origin = get_schema(args.dbname)
    PARENTHESIZE_FLAG = args.parenthesize_switch
    # fpath = '/Users/zilinzhang/Workspace/Github/nl2sql/Data/Initial/table/art_1_table.json'
    print(schema_origin)

    # schema = Schema(get_schema('art_1.sqlite'))
    # print schema.schema
    # schema_origin = {"paragraphs": ["paragraph_text","paragraph_id", "document_id"], "documents": ["document_id", "document_name"], "sentences": ["sentence_id", "content", "paragraph_id"]}
    schema = Schema(schema_origin)
    # print schema.idMap

    data = load_data(args.input_SQL_file)
    total_count = 0
    error_count = 0
    f_out = open(output_file_name, "w")
    f_err = open(output_error_file_name, "w")
    for ix, datum in enumerate(data):
        query = datum[2].strip()
        #     print(query)
        #        continue
        # query = "SELECT template_id FROM Templates WHERE template_type_code  =  \"PP\" OR template_type_code  =  \"PPT\""
        # query = "SELECT count ( * ) FROM Paragraphs AS T1 JOIN Documents AS T2 ON T1.document_ID  =  T2.document_ID WHERE T2.document_name  =  'Summer Show'"
        # query = "SELECT T1.paragraph_id ,   T1.paragraph_text FROM Paragraphs AS T1 JOIN Documents AS T2 ON T1.document_id  =  T2.document_id WHERE T2.Document_Name  =  'Welcome to NY'"
        # query = "SELECT T1.paragraph_id ,   T1.paragraph_text FROM Paragraphs AS T1 JOIN Documents AS T2 ON T1.document_id  =  T2.document_id JOIN Sentences AS S1 ON T1.paragraph_id = S1.paragraph_id WHERE T2.Document_Name  =  'Welcome to NY' AND S1.content = 'NOWAY'"
        # query = "SELECT T1.paragraph_id ,   T1.paragraph_text FROM Paragraphs AS T1, Documents AS T2 WHERE T1.document_id  =  T2.document_id AND T2.Document_Name  =  'Welcome to NY'"
        # query = "select city.city_name from city where city.population = (select max(city.population) from city);"
        try:
            total_count = total_count + 1
            toks = tokenize(query)
            toks_as, tables_with_alias = get_tables_with_alias(
                schema.schema, toks, schema_origin.keys()
            )
            #       print toks
            #       print tables_with_alias
            _, sql = parse_sql(toks_as, 0, tables_with_alias, schema)
            print_sql = {"index": int(datum[1])}
            print_sql["sql"] = sql
            #       print sql
            json.dump(print_sql, f_out)
            f_out.write("\n")
        except Exception as e:
            print("SQL: {}, ERROR: {}".format(query, e))
            error_count = error_count + 1
            f_err.write(query + "\n")
            f_out.write("\n")
    #        print e
    #     break
    f_out.close()
    f_err.close()
    print("TOTAL QUERIES: {}, ERROR: {}".format(total_count, error_count))
