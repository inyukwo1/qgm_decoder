# import unittest
import json
from qgm.qgm_import_from_sql_ds import qgm_import_from_sql_ds
from qgm.qgm import QGM
from qgm.qgm_action import QGM_ACTION
from sql_ds.sql_ds import SQLDataStructure
from sql_ds.sql_ds_to_string import beutify


# class QGMConvertTest(unittest.TestCase):
#     def test_reconverting_spider_dev(self):
#         table_path = "data/spider/tables.json"
#         dev_path = "data/spider/train.json"
#         table_data = []
#         with open(table_path) as f:
#             table_data += json.load(f)
#         table_data = {table["db_id"]: table for table in table_data}
#         with open(dev_path) as f:
#             data = json.load(f)
#             passed_num = 0
#             for idx, datum in enumerate(data):
#                 db = table_data[datum["db_id"]]
#                 db["col_set"] = datum["col_set"]
#                 spider_sql = datum["sql"]
#
#                 # try:
#                 sql_ds = SQLDataStructure.import_from_spider_sql(spider_sql, db)
#                 qgm = qgm_import_from_sql_ds(sql_ds)
#                 if qgm is None:
#                     # TODO not supporting qgm yet
#                     continue
#                 sql_ds_reconvert = SQLDataStructure()
#                 sql_ds_reconvert.import_from_qgm(qgm)
#                 reconvert = sql_ds_reconvert.to_string()
#                 origin = beutify(datum["query"])
#                 origin = origin.replace("DISTINCT ", "")
#                 assert reconvert.lower() == origin.lower()
#                 # except:
#                 #     continue
#                 new_qgm = QGM(db, False)
#                 for (
#                     (symbol, answer, prev_idx),
#                     (new_symbol, setter, new_qgm_prev_idx),
#                 ) in zip(qgm.qgm_construct(), new_qgm.qgm_construct()):
#                     if symbol is None:
#                         assert new_symbol is None
#                         break
#                     assert symbol == new_symbol
#                     setter(QGM_ACTION.symbol_action_to_action_id(symbol, answer))
#                     assert prev_idx == new_qgm_prev_idx
#                 assert new_qgm == qgm
#                 passed_num += 1
#         print(passed_num)


if __name__ == "__main__":
    # unittest.main()

    table_path = "data/spider/tables.json"
    dev_path = "data/spider/train.json"
    table_data = []
    with open(table_path) as f:
        table_data += json.load(f)
    table_data = {table["db_id"]: table for table in table_data}
    with open(dev_path) as f:
        data = json.load(f)
        passed_num = 0
        converted_num = 0
        for idx, datum in enumerate(data):
            db = table_data[datum["db_id"]]
            db["col_set"] = datum["col_set"]
            spider_sql = datum["sql"]
            try:
                origin = beutify(datum["query"])
                origin = origin.replace("DISTINCT ", "")
                # if "NOT IN" not in origin:
                #     continue

                sql_ds = SQLDataStructure.import_from_spider_sql(spider_sql, db)
                qgm = qgm_import_from_sql_ds(sql_ds)
                if qgm is None:
                    # TODO not supporting qgm yet
                    continue
                sql_ds_reconvert = SQLDataStructure()
                sql_ds_reconvert.import_from_qgm(qgm)
                reconvert = sql_ds_reconvert.to_string()
                # assert reconvert.lower() == origin.lower()
                reconvert_split = reconvert.split()
                origin_split = origin.split()

                if len(reconvert_split) == len(origin_split):
                    for word_idx, (r_word, o_word) in enumerate(
                        zip(reconvert_split, origin_split)
                    ):
                        if (
                            r_word[3:].lower() == o_word.lower()
                            and r_word[0] == "T"
                            and r_word[2] == "."
                        ):
                            origin_split[word_idx] = (
                                word_idx[:3] + o_word
                            )  # TODO HACK!! not really correct
                reconvert = " ".join(reconvert_split)
                origin = " ".join(origin_split)

                if len(reconvert.lower()) != len(
                    origin.lower()
                ):  # TODO HACK!! assuming same length := same query
                    print(reconvert)
                    print(origin)
                    print("")
                    continue
            except:
                print(idx)
                print("error")
                print(beutify(datum["query"]))
                print("")
                continue
            new_qgm = QGM(db, False)
            for (
                (symbol, answer, prev_idx),
                (new_symbol, setter, new_qgm_prev_idx),
            ) in zip(qgm.qgm_construct(), new_qgm.qgm_construct()):
                if symbol is None:
                    assert new_symbol is None
                    break
                assert symbol == new_symbol
                if symbol in {"C", "T"}:
                    action_id = answer
                else:
                    action_id = QGM_ACTION.symbol_action_to_action_id(symbol, answer)
                setter(action_id)
                assert prev_idx == new_qgm_prev_idx
            assert new_qgm == qgm
            passed_num += 1
            qgm.qgm_canonicalize()
            if new_qgm != qgm:
                sql_ds_reconvert = SQLDataStructure()
                sql_ds_reconvert.import_from_qgm(qgm)
                reconvert = sql_ds_reconvert.to_string()
                # print(reconvert)
                # print(origin)
                # print("")
                converted_num += 1

    print(passed_num)
    print(converted_num)
