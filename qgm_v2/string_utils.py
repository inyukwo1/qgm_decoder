from nltk import word_tokenize
import re

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

COND_OPS = ("and", "or", "start")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")

FLOAT_MATCH = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$").match


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
            and toks_no_value[i - 1] in OPS
        ):
            toks_no_value[i] = "value"

    if has_value:
        return toks
    return toks_no_value
