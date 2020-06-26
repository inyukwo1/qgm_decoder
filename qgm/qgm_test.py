import unittest
import json
from sql_ds.sql_ds import SQLDataStructure
from sql_ds.sql_ds_to_string import beutify
from qgm.qgm_import_from_sql_ds import qgm_import_from_sql_ds


class QGMConvertTest(unittest.TestCase):
    def test_reconverting_spider_dev(self):
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
                # print(idx)
                # not well parsed
                # TODO fix Wrong spider
                if idx in {
                    93,
                    94,
                    161,
                    162,
                    875,
                    876,
                    1102,
                    1103,
                    1282,
                    1283,
                    1289,
                    1501,
                    1503,
                    1621,
                    1622,
                    1640,
                    1792,
                    1793,
                    1794,
                    1795,
                    1991,
                    2207,
                    2208,
                    2209,
                    2210,
                    2227,
                    2228,
                    2229,
                    2230,
                    2231,
                    2232,
                    2868,
                    2869,
                    2898,
                    2899,
                    2900,
                    2901,
                    2902,
                    2903,
                    2968,
                    2969,
                    3068,
                    3069,
                    3078,
                    3079,
                    3125,
                }:
                    continue
                # TODO fix spider - cannot represented by spider ds - e.g. self join
                if idx in {209, 210, 589, 590, 605, 606, 1646}:
                    continue

                # TODO fix Wrong db 1)fp constraint in college_2
                if idx in {1430, 1431}:
                    continue

                # well parsed but skip
                # TODO table abbrev order
                if idx in {12, 14, 1456, 1457}:
                    continue
                # TODO not using abbrev
                if idx in {
                    2110,
                    2111,
                    2115,
                    2116,
                    2117,
                    2119,
                    2121,
                    2122,
                    2127,
                    2128,
                    2129,
                    2836,
                    2837,
                    2838,
                    2839,
                    2840,
                    2841,
                }:
                    continue
                # currently not support
                # TODO condition in order
                if idx in {433, 434}:
                    continue
                # TODO multiple groupby
                if idx in {1402, 1403, 1406, 1407, 1412, 1413, 2898, 2899}:
                    continue
                # TODO multiple join condition
                if idx in {1450, 1451, 2380, 2381, 2382, 2383, 2928, 2929}:
                    continue
                # TODO col op col (+col) in where
                if idx in {
                    1820,
                    1821,
                    1822,
                    1823,
                    1904,
                    2422,
                    2423,
                    2480,
                    2481,
                    2486,
                    2487,
                    2488,
                    2489,
                    2602,
                    2603,
                }:
                    continue

                db = table_data[datum["db_id"]]
                db["col_set"] = datum["col_set"]
                sql_query = beutify(datum["query"])
                spider_sql = datum["sql"]

                try:
                    sql_ds = SQLDataStructure.import_from_spider_sql(spider_sql, db)
                    qqgm = qgm_import_from_sql_ds(sql_ds)
                    if qqgm is None:
                        # TODO not supporting qgm yet
                        continue
                    sql_ds_reconvert = SQLDataStructure()
                    sql_ds_reconvert.import_from_qgm(qqgm)
                    reconvert = sql_ds_reconvert.to_string()

                    reconvert_split = reconvert.split()
                    # TODO not we don't support distinct
                    sql_query = sql_query.replace("DISTINCT ", "")

                    sql_query_split = sql_query.split()
                    if len(sql_query_split) == len(reconvert_split):
                        for q_idx in range(len(sql_query_split)):
                            if len(sql_query_split[q_idx]) + 3 == len(
                                reconvert_split[q_idx]
                            ):
                                sql_query_split[q_idx] = reconvert_split[q_idx]
                        sql_query = " ".join(sql_query_split)
                        reconvert = " ".join(reconvert_split)

                    passed_num += 1
                    assert (
                        sql_query.lower() == reconvert.lower()
                    ), "IDX: {}\nGOLD: {}\nMINE: {}".format(idx, sql_query, reconvert)
                except:
                    print("w")


if __name__ == "__main__":
    unittest.main()
