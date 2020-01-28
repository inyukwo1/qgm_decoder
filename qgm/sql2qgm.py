import json
import argparse
from ops import WHERE_OPS, AGG_OPS, BOX_OPS


def create_group_by_box(sql, info, schema):
    # Append group by box
    heads = []
    quantifiers = []
    quantifier_types = []
    local_predicates = []
    if info["groupBy"]:
        for group_unit in info["groupBy"]:
            # Get heads
            agg_id = group_unit[0]
            col_id = group_unit[1]
            heads += [(agg_id, col_id)]

            # Get body - quantifiers
            table_id = schema["column_names"][col_id][0]
            quantifiers += [table_id]
            quantifier_types += ["f"]

        for idx in range(0, len(info["having"]), 2):
            having_unit = info["having"][idx]
            operator = having_unit[1]
            agg_id = having_unit[2][1][0]
            col_id = having_unit[2][1][1]
            if col_id != 0:
                table_id = schema["column_names"][col_id][0]
                quantifiers += [table_id]
                quantifier_types += ["f"]
            if isinstance(having_unit[3], dict):
                right_operand = sql2qgm(sql, having_unit[3], schema)
                quantifiers += [right_operand]
                quantifier_types += ["s"]
            elif having_unit[4]:
                right_operand = having_unit[3:5]
            else:
                right_operand = having_unit[3]

            local_predicates += [(agg_id, col_id, operator, right_operand)]
        body = {
            "quantifiers": quantifiers,
            "quantifier_types": quantifier_types,
            "join_predicates": [],
            "local_predicates": local_predicates,
        }
        box = {"head": heads, "body": body, "operator": BOX_OPS.index("groupBy")}
        return box
    else:
        return None


def create_order_by_box(sql, info, schema, group_by_box):
    heads = []
    quantifiers = []
    quantifier_types = []
    if info["orderBy"]:
        is_asc = info["orderBy"][0] == "asc"
        limit_num = info["limit"]
        for order_unit in info["orderBy"][1]:
            # Get info
            agg_id = order_unit[1][0]
            col_id = order_unit[1][1]
            heads += [(agg_id, col_id)]
            if col_id == 0:
                table_id = group_by_box["body"]["quantifiers"][0]
                quantifiers += [table_id]
                quantifier_types += [group_by_box["body"]["quantifier_types"][0]]
            else:
                table_id = schema["column_names"][col_id][0]
                quantifiers += [table_id]
                quantifier_types += ["f"]
        # Create body
        body = {
            "quantifiers": quantifiers,
            "quantifier_types": quantifier_types,
            "join_predicates": [],
            "local_predicates": [],
            "is_asc": is_asc,
            "limit_num": limit_num,
        }
        # Create box
        box = {"head": heads, "body": body, "operator": BOX_OPS.index("orderBy")}
        return box
    else:
        return None


def create_select_box(sql, info, schema):
    heads = []
    quantifiers = []
    quantifier_types = []
    local_predicates = []
    join_predicates = []

    # Create head - select agg and columns
    for select_unit in info["select"][1]:
        agg = select_unit[0]
        col_id = select_unit[1][1][1]
        heads += [(agg, col_id)]

    # Create body - quantifiers
    for table_unit in info["from"]["table_units"]:
        assert table_unit[0] in ["sql", "table_unit"]
        if "sql" == table_unit[0]:
            something = sql2qgm(sql, table_unit[1], schema)
            quantifiers += [something]
            quantifier_types += ["s"]
        else:
            table_id = table_unit[1]
            quantifiers += [table_id]
            quantifier_types += ["f"]

    # Create body - join predicates
    for conds_unit in info["from"]["conds"]:
        something = conds_unit
        # left col
        col1_id = something[2][1][1]
        table1_id = schema["column_names"][col1_id][0]
        assert table1_id in quantifiers

        # right col
        col2_id = something[3][1]
        table2_id = schema["column_names"][col2_id][0]
        assert table2_id in quantifiers

        # join
        join_predicates += [(col1_id, WHERE_OPS.index("="), col2_id)]

    for idx in range(0, len(info["where"]), 2):
        where_unit = info["where"][idx]

        # Agg
        agg_id = where_unit[2][1][0]

        # Operand1
        col_id = where_unit[2][1][1]
        table_id = schema["column_names"][col_id][0]

        assert table_id in quantifiers

        # Operator
        operator = where_unit[1]
        operator = (
            WHERE_OPS.index("not " + WHERE_OPS[operator]) if where_unit[0] else operator
        )

        # Operand2
        if isinstance(where_unit[3], dict):
            right_operand = sql2qgm(sql, where_unit[3], schema)
            quantifiers += [right_operand]
            quantifier_types += ["s"]
        else:
            right_operand = where_unit[3]
            if isinstance(right_operand, list):
                agg2_id = where_unit[3][0]
                col2_id = where_unit[3][1]
                right_operand = (agg2_id, col2_id)
            elif where_unit[4]:
                right_operand = where_unit[3:5]
            else:
                right_operand = where_unit[3]

        # add quantifier id
        local_predicates += [(agg_id, col_id, operator, right_operand)]

    # Combine body
    body = {
        "quantifiers": quantifiers,
        "quantifier_types": quantifier_types,
        "join_predicates": join_predicates,
        "local_predicates": local_predicates,
    }
    # Create Box
    if heads and isinstance(heads[0], list):
        print("error!!!!")
        print(heads)
    box = {"head": heads, "body": body, "operator": BOX_OPS.index("select")}

    return box


def sql2qgm(sql, info, schema):
    """
    box unit:
        1. table
        2. operator
        3. box:
            3-1. head
            3-2. body
    box: {'head', 'body', 'operator'}
    box_body: {'quantifiers', 'quantifier_types', 'join_predicates', 'local_predicates'}
    box_head: []
    """
    select_box = create_select_box(sql, info, schema)
    group_by_box = create_group_by_box(sql, info, schema)
    order_by_box = create_order_by_box(sql, info, schema, group_by_box)

    if info["intersect"]:
        box_operator = "intersect"
    elif info["union"]:
        box_operator = "union"
    elif info["except"]:
        box_operator = "except"
    else:
        box_operator = "select"

    # Combine boxes
    boxes = [select_box]
    if group_by_box:
        boxes += [group_by_box]
    if order_by_box:
        boxes += [order_by_box]

    # Create box2
    if box_operator != "select":
        box2 = sql2qgm(sql, info[box_operator], schema)
        box2[0]["operator"] = BOX_OPS.index(box_operator)
        boxes += box2

    return boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, type=str, help="db tables path")
    parser.add_argument("--source", default=None, type=str, help="loading path")
    parser.add_argument("--destin", default=None, type=str, help="saving path")
    args = parser.parse_args()

    # Load DB info
    dbs = json.load(open(args.db, "rb"))
    db_dic = {db["db_id"]: db for db in dbs}

    # Load data
    datas = json.load(open(args.source, "rb"))

    print("Loading complete!")

    # Convert
    for idx, data in enumerate(datas):
        tmp = sql2qgm(data["query"], data["sql"], db_dic[data["db_id"]])
        data["qgm"] = tmp

    print("Translating complete!")

    # Save
    with open(args.destin, "w") as f:
        json.dump(datas, f)

    print("Saving QGM complete!")
