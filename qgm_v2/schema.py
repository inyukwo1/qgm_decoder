from qgm_v2.string_utils import *


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

    def get_tables_with_alias(this, toks):
        if isinstance(this._table, dict) and "table_names_original" in this._table:
            tables = this._table["table_names_original"]
        else:
            tables = this._table
        toks_as, tables = scan_alias(toks, tables)
        # print(toks_as, tables)
        for key in this.schema:
            assert key not in tables, "Alias {} has the same name in table".format(key)
            tables[key] = key
        return toks_as, tables


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


def get_schemas_from_json(db):
    schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
    column_names_original = db["column_names_original"]
    table_names_original = db["table_names_original"]
    returning_table = {
        "column_names_original": column_names_original,
        "table_names_original": table_names_original,
    }
    for i, tabn in enumerate(table_names_original):
        table = str(tabn.lower())
        cols = [str(col.lower()) for td, col in column_names_original if td == i]
        schema[table] = cols

    return Schema(schema, returning_table)
