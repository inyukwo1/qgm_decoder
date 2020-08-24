import json
from qgm_v2.qgm import QGM
from sql_ds.sql_ds_to_string import beutify


if __name__ == "__main__":
    table_path = "data/spider/tables.json"
    dev_path = "data/spider/train.json"
    table_data = []
    with open(table_path) as f:
        table_data += json.load(f)
    table_data = {table["db_id"]: table for table in table_data}
    with open(dev_path) as f:
        data = json.load(f)
        passed_num = 0
        for idx, datum in enumerate(data):
            db = table_data[datum["db_id"]]
            db["col_set"] = datum["col_set"]
            spider_sql = datum["sql"]
            try:
                origin = beutify(datum["query"])
                origin = origin.replace("DISTINCT ", "")
                qgm = QGM.import_from_sql(origin)
                assert qgm.export_to_sql() == origin
            except:
                print(idx)
                print("error")
                print(origin)
                print(datum["question"])
                print("")
                # traceback.print_exc()
                # break
                continue
    print(passed_num)
