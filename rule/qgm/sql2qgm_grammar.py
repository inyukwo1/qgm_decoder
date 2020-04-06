import json

table_file_name = "./data/spider/tables.json"
data_file_name = "./data/spider/dev.json"
datas = json.load(open(data_file_name))
dbs = {item["db_id"]: item for item in json.load(open(table_file_name))}


def sql2qgm(query, sql, db):
    if "FROM (SELECT" in query or "FROM ( SELECT" in query:
        stop = 1
    # Filter out those with "except", "intersect", "union"
    # Filter out nested queries

    # From

    # Select

    # Where

    # Order by

    # Group by
    if sql["groupBy"]:
        stop = 1
    return None


qgms = []
for data in datas:
    db_id = data["db_id"]
    sql = data["sql"]
    query = data["query"]
    # sql2qgm
    qgms += [sql2qgm(query, sql, dbs[db_id])]

stop = 1
