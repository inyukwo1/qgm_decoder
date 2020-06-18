import re
import json
from typing import List

# dataset = "spider"
# dataset = "wikisql"
dataset = "wikitablequestions"

mode = "train"
# mode = "dev"
# mode = "test"

file_name = "./data/{}/{}.json".format(dataset, mode)

WHERE_OPS = ["=", "!=", "<", ">", "<=", ">=", "BETWEEN", "LIKE", "NOT LIKE", "IN", "NOT IN", "NOT", "-"]

def is_column(string: str, cols: List[str]) -> bool:
    string = string.lower()
    if "." in string:
        string = string.split(".")[1]
    return string in cols

def is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except:
        return False

def str_idx_to_list_idx(string: str, str_idx: int, deliminator: str = " ") -> int:
    string = string.split(deliminator)
    cnt = 0
    for idx, item in enumerate(string):
        cnt += len(item) + 1
        if str_idx < cnt:
            return idx
    raise RuntimeError("Should note be here")

def list_idx_to_str_idx(string: str, list_idx: int, deliminator: str = " ") -> int:
    string = string.split(deliminator)
    string = " ".join(string[:list_idx+1])
    return len(string)-1

def find_end_parentheses(string: str, start_idx: int) -> int:
    cnt = 0
    stack = []
    for idx in range(start_idx, len(string)):
        if string[idx] == "(":
            stack += [idx]
            cnt += 1
        elif string[idx] == ")":
            popped_idx = stack.pop()
            if not stack:
                assert popped_idx == start_idx, "original:{} popped:{}".format(start_idx, popped_idx)
                return idx
    if stack:
        print("Total cnt: {}Start_idx:{} Uneven parentheses in the string: {}".format(cnt, start_idx, string))
        return None
    else:
        raise RuntimeError("Should not be here")


def remove_double_space(string: str) -> str:
    while "  " in string:
        string = string.replace("  ", " ")
    return string

def remove_space_in_aggregator(sql: str) -> str:
    def remove_space_in_regex(string, regex):
        result = regex.search(string)
        if result:
            target_phrase = result.group()
            no_distinct_phrase = target_phrase.replace("DISTINCT", "")
            no_space_phrase = no_distinct_phrase.replace(" ", "")
            string = string.replace(target_phrase, no_space_phrase)
            return remove_space_in_regex(string, regex)
        else:
            return string
    sql = remove_space_in_regex(sql, re.compile(r'((sum|avg|count|max|min|SUM|AVG|COUNT|MAX|MIN|Sum|Avg|Count|Max|Min)\()([\s*DISTINCT ]*)([_|\.||a-z|A-Z|0-9]*\s)\)'))
    sql = remove_space_in_regex(sql, re.compile(r'((sum|avg|count|max|min|SUM|AVG|COUNT|MAX|MIN|Sum|Avg|Count|Max|Min)\s?\()([\s*DISTINCT ]*)([_|\.|\*|a-z|A-Z|0-9]*\s)\)'))

    return sql

def pretty_format(string: str) -> str:
    string = string.strip()
    string = string.strip(";")
    string = string.replace("IN(", "IN (")
    string = string.replace("(SELECT", "( SELECT")
    string = string.replace(")", " )")
    string = string.replace(" DISTINCT ", "")
    string = remove_space_in_aggregator(string)

    return string

def extract_where_values_from_sql(sql: str, col_names: List[str]) -> List[str]:
    # Preprocess sql string
    sql = pretty_format(sql)
    sql = remove_double_space(sql)
    sql_split = sql.split(" ")

    # Extract string
    values = []
    where_indices = [idx for idx in range(len(sql_split)) if sql_split[idx] == "WHERE"]
    for idx in where_indices:
        val_dis = 3
        assert sql_split[idx+2] in WHERE_OPS, "{}-sql:{} ".format(sql_split[idx+2], sql)

        # Find starting index of value
        if sql_split[idx+val_dis] in ["IN", "LIKE"]:
            val_dis += 1
        if sql_split[idx+val_dis] == "(":
            continue
        value_start_idx = idx+val_dis
        value = sql_split[value_start_idx]

        # Pass since it is using column instead of value
        if is_column(value, col_names):
            continue

        # Get value
        if is_number(value):
            pass
        elif value[0] in ["'", '"'] and value[-1] in ["'", '"']:
            value = value.strip()
        elif value[0] in ["'", '"']:
            sub_sql = sql_split[value_start_idx+1:]
            if value[0] in sub_sql:
                value_end_idx = sql_split[value_start_idx + 1:].index(value[0])
            else:
                value_end_idx = [idx for idx, word in enumerate(sub_sql) if value[0] in word][0]
            value_end_idx += value_start_idx + 1
            value = " ".join(sql_split[idx:value_end_idx])
        else:
            raise RuntimeError("value:{}-sql:{}".format(value, sql))
        values += [value]

    return [value.strip("'").strip('"') for value in values]

def extract_having_values_from_sql(sql: str, col_names: List[str]) -> List[str]:
    # Preprocess sql string
    ori_sql = sql
    if ori_sql == 'SELECT t1.name FROM organization AS t2 JOIN author AS t1 ON t2.oid  =  t1.oid JOIN writes AS t3 ON t3.aid  =  t1.aid JOIN publication AS t4 ON t3.pid  =  t4.pid WHERE t2.name  =  "University of Michigan" GROUP BY t1.name HAVING SUM ( t4.citation_num )  >  5000;':
        stop = 1
    sql = pretty_format(sql)
    sql = remove_double_space(sql)
    sql_split = sql.split(" ")

    # Extract string
    values = []
    having_indices = [idx for idx in range(len(sql_split)) if sql_split[idx] == "HAVING"]
    for idx in having_indices:
        val_dis = 3
        # Find Value
        if sql_split[idx+val_dis][0] == "(":
            # To string idx
            start_str_idx = list_idx_to_str_idx(sql, idx+val_dis)
            end_str_idx = find_end_parentheses(sql, start_str_idx)

            # To list idx
            start_list_idx = idx+val_dis
            end_list_idx = str_idx_to_list_idx(sql, end_str_idx) + 1

            # Get value
            value = " ".join(sql_split[start_list_idx:end_list_idx])
        else:
            value = sql_split[idx+val_dis]
        values += [value]
        # print("value:{}".format(value))
    if values:
        flag = False
        for item in values:
            if is_number(item):
                pass
            elif item[0] in ['"', "'"] and item[-1] in ['"', "'"]:
                pass
            else:
                flag = True
        if flag:
            print("ori:{}".format(ori_sql))
            print('sql:{}'.format(sql))
            for item in values:
                print('value:{}'.format(item))
            print("\n")
    return values

if __name__ == "__main__":
    # tmp = 'SELECT t1.name FROM organization AS t2 JOIN author AS t1 ON t2.oid  =  t1.oid JOIN writes AS t3 ON t3.aid  =  t1.aid JOIN publication AS t4 ON t3.pid  =  t4.pid WHERE t2.name  =  "University of Michigan" GROUP BY t1.name HAVING SUM ( t4.citation_num  )  >  5000'
    # tmp = 'SELECT t1.name FROM organization AS t2 JOIN author AS t1 ON t2.oid  =  t1.oid JOIN writes AS t3 ON t3.aid  =  t1.aid JOIN publication AS t4 ON t3.pid  =  t4.pid WHERE t2.name  =  "University of Michigan" GROUP BY t1.name HAVING SUM ( t4.citation_num  )  >  5000'
    # regex = re.compile(r'((sum|avg|count|max|min|SUM|AVG|COUNT|MAX|MIN|Sum|Avg|Count|Max|Min)\()([\s*DISTINCT ]*)([_|\.||a-z|A-Z|0-9]*\s)\)')
    # regex2 = re.compile(r'((sum|avg|count|max|min|SUM|AVG|COUNT|MAX|MIN|Sum|Avg|Count|Max|Min)\s?\()([\s*DISTINCT ]*)([_|\.|\*|a-z|A-Z|0-9]*\s)\)')
    # a = regex.search(tmp)
    # b = regex2.search(tmp)
    # result = None
    # if a:
    #     result = a.group()
    # elif b:
    #     result = b.group()
    # if result:
    #     print(tmp.replace(result, result.replace(" ", "")))
    # else:
    #     print("bad")
    #
    # exit(-1)

    table_path = "./data/spider/tables.json"
    table_data = {item["db_id"]: item for item in json.load(open(table_path))}
    train_path = "./data/spider/train.json"
    train_data = json.load(open(train_path))
    dev_path = "./data/spider/dev.json"
    dev_data = json.load(open(dev_path))

    for data in train_data+dev_data:
        sql = data["query"]
        db = table_data[data["db_id"]]
        col_names = [item[1].lower() for item in db["column_names_original"]]
        where_values = extract_where_values_from_sql(sql, col_names)
        having_values = extract_having_values_from_sql(sql, col_names)
        # print("sql:{}".format(sql))
        # print("where values:{}".format(where_values))
        # print("having values:{}\n".format(having_values))
