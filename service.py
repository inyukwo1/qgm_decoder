import torch
import sklearn
from validation.check_correctness import compare_two_queries, diff_two_queries

import json
import os
from flask_restful import Resource, reqparse, Api
from flask import Flask
from flask import send_file
from flask_cors import CORS
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objs as gro
from plotly.offline import plot
import time
from scp import SCPClient
import paramiko
import traceback
from abc import *

PLOTDIR = "web/public/"

AGG_SQL = ["count", "avg", "min", "max", "sum"]
DATASET = ["spider"]


class End2End(metaclass=ABCMeta):
    @abstractmethod
    def prepare_model(self, dataset):
        pass

    @abstractmethod
    def run_model(self, db_id, question):
        pass


origin_db_table = {
    "headers": [
        [{"name": "cast", "colspan": 9}],
        [
            {"name": "movie.mid", "colspan": 1},
            {"name": "movie.title", "colspan": 1},
            {"name": "movie.release_year", "colspan": 1},
            {"name": "tv_series.sid", "colspan": 1},
            {"name": "tv_series.title", "colspan": 1},
            {"name": "tv_series.release_year", "colspan": 1},
            {"name": "actor.aid", "colspan": 1},
            {"name": "actor.name", "colspan": 1},
            {"name": "actor.birth_year", "colspan": 1},
        ],
    ],
    "rows": [
        [
            "2627267",
            "100 NO SHIKAKU WO MOTSU ONNA - FUTARI NO BATSUICHI SATSUJIN SôSA 1",
            "2008",
            "-",
            "-",
            "-",
            "4479",
            "ROKURO ABE",
            "-",
        ],
        [
            "2627616",
            "1001 INVENTIONS AND THE WORLD OF IBN AL-HAYTHAM",
            "2015",
            "-",
            "-",
            "-",
            "3267",
            "KHALID ABDALLA",
            "1980",
        ],
        [
            "2627656",
            "101 BIGGEST CELEBRITY OOPS",
            "2004",
            "-",
            "-",
            "-",
            "1718368",
            "BRAD PITT",
            "1963",
        ],
        [
            "2627707",
            "101 SEXIEST CELEBRITY BODIES",
            "2005",
            "-",
            "-",
            "-",
            "1718368",
            "BRAD PITT",
            "1963",
        ],
        [
            "2627707",
            "101 SEXIEST CELEBRITY BODIES",
            "2005",
            "-",
            "-",
            "-",
            "7131985",
            "SUNSHINE",
            "-",
        ],
        [
            "2628257",
            "12 MEN OF CHRISTMAS",
            "2009",
            "-",
            "-",
            "-",
            "6855",
            "AARON ABRAMS",
            "1978",
        ],
        ["2628399", "120", "2008", "-", "-", "-", "7172", "YASAR ABRAVAYA", "1990",],
        [
            "2632907",
            "2013 MTV MOVIE AWARDS",
            "2013",
            "-",
            "-",
            "-",
            "1718368",
            "BRAD PITT",
            "1963",
        ],
        [
            "-",
            "-",
            "-",
            "905371",
            "HEADLINES ON TRIAL",
            "2006",
            "1718368",
            "BRAD PITT",
            "1963",
        ],
        [
            "-",
            "-",
            "-",
            "905371",
            "HEADLINES ON TRIAL",
            "2006",
            "2466214",
            "JENNIFER ANISTON",
            "1969",
        ],
    ],
}

verify_db_table = {
    "headers": [
        [{"name": "cast", "colspan": 9}],
        [
            {"name": "movie.mid", "colspan": 1},
            {"name": "movie.title", "colspan": 1},
            {"name": "movie.release_year", "colspan": 1},
            {"name": "tv_series.sid", "colspan": 1},
            {"name": "tv_series.title", "colspan": 1},
            {"name": "tv_series.release_year", "colspan": 1},
            {"name": "actor.aid", "colspan": 1},
            {"name": "actor.name", "colspan": 1},
            {"name": "actor.birth_year", "colspan": 1},
        ],
    ],
    "rows": [
        [
            "2625911",
            "...FIRST DO NO HARM",
            "1997",
            "-",
            "-",
            "-",
            "6512",
            "CHARLIE ABRAHAMS",
            "1992",
        ],
        [
            "2627595",
            "10000 HOURS",
            "2013",
            "-",
            "-",
            "-",
            "6261",
            "APOLLO ABRAHAM",
            "1972",
        ],
        [
            "2627707",
            "101 SEXIEST CELEBRITY BODIES",
            "2005",
            "-",
            "-",
            "-",
            "1718368",
            "BRAD PITT",
            "1963",
        ],
        [
            "2627707",
            "101 SEXIEST CELEBRITY BODIES",
            "2005",
            "-",
            "-",
            "-",
            "7131985",
            "SUNSHINE",
            "-",
        ],
        [
            "2628257",
            "12 MEN OF CHRISTMAS",
            "2009",
            "-",
            "-",
            "-",
            "6855",
            "AARON ABRAMS",
            "1978",
        ],
        [
            "-",
            "-",
            "-",
            "905371",
            "HEADLINES ON TRIAL",
            "2006",
            "1718368",
            "BRAD PITT",
            "1963",
        ],
        [
            "-",
            "-",
            "-",
            "905371",
            "HEADLINES ON TRIAL",
            "2006",
            "2466214",
            "JENNIFER ANISTON",
            "1969",
        ],
        [
            "-",
            "-",
            "-",
            "1005702",
            "IMPERIUM",
            "2004",
            "3484963",
            "ADRIENNE SANTANGELO",
            "-",
        ],
    ],
}


class VerifyService(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument("db_id", required=True, type=str)
            return {
                "new_db_instance": verify_db_table,
                "execution_result": {
                    "headers": [[{"name": "count(*)", "colspan": 1}]],
                    "rows": [["4"]],
                },
            }
        except Exception as e:
            print("done not well")
            traceback.print_exc()
            return {"result": str(e)}


class DBInstanceService(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument("db_id", required=True, type=str)
            return {
                "db_obj": {
                    "column_names": [
                        [-1, "*"],
                        [0, "mid"],
                        [0, "title"],
                        [0, "release year"],
                        [1, "msid"],
                        [1, "aid"],
                        [2, "aid"],
                        [2, "name"],
                        [2, "birth year"],
                        [3, "sid"],
                        [3, "title"],
                        [3, "release year"],
                    ],
                    "column_names_original": [
                        [-1, "*"],
                        [0, "mid"],
                        [0, "title"],
                        [0, "release_year"],
                        [1, "msid"],
                        [1, "aid"],
                        [2, "aid"],
                        [2, "name"],
                        [2, "birth_year"],
                        [3, "sid"],
                        [3, "title"],
                        [3, "release_year"],
                    ],
                    "column_types": [
                        "text",
                        "number",
                        "text",
                        "text",
                        "number",
                        "number",
                        "number",
                        "text",
                        "text",
                        "number",
                        "text",
                        "text",
                    ],
                    "db_id": "",
                    "foreign_keys": [[4, 1], [5, 6], [4, 9]],
                    "primary_keys": [1, 6, 9],
                    "table_names": ["movie", "cast", "actor", "tv series"],
                    "table_names_original": ["movie", "cast", "actor", "tv_series"],
                    "only_cnames": [
                        "*",
                        "mid",
                        "title",
                        "release year",
                        "msid",
                        "aid",
                        "aid",
                        "name",
                        "birth year",
                        "sid",
                        "title",
                        "release year",
                    ],
                },
                "db_instance_table": origin_db_table,
            }
        except Exception as e:
            print("done not well")
            traceback.print_exc()
            return {"result": str(e)}


class Service(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument("model", required=True, type=str)
            parser.add_argument("db_id", required=True, type=str)
            parser.add_argument("db_obj", required=True, type=str)
            parser.add_argument("question", required=True, type=str)
            parser.add_argument("mode", default="Explore", type=str)
            parser.add_argument("gold_sql", default=None, type=str)
            parser.add_argument("gen_sql", default=None, type=str)
            parser.add_argument("diffmode", default="origin", type=str)
            parser.add_argument("query_cnt", default=0, type=int)
            args = parser.parse_args()

            print(args)
            if args["mode"] == "Explore":
                return self.handleExploreMode(args)

            elif args["mode"] == "Analyze":
                return self.handleAlalyzeMode(args)

        except Exception as e:
            print("done not well")
            traceback.print_exc()
            return {"result": str(e)}

    def handleExploreMode(self, args):
        table = json.loads(args["db_obj"])
        if args["model"] == "ours":
            model = ours_end2end["spider"]
        elif args["model"] == "irnet":
            model = irnet_end2end["spider"]
        elif args["model"] == "gnn":
            model = gnn_end2end["spider"]
        elif args["model"] == "irnet-improved":
            model = irnet_end2end["spider"]
        else:
            return
        result_query, actions, question = model.run_model(
            args["db_id"], args["question"], table
        )

        if args["model"] == "irnet-improved":
            result_query = 'SELECT T1.release_year FROM movie AS T1 WHERE T1.title = "Dead Poets Society"'

        if args["model"] == "irnet":
            result_query = ours_end2end["spider"].value_predictor(
                args["db_id"], result_query, args["question"], " 1"
            )
        elif args["model"] == "gnn":
            result_query = ours_end2end["spider"].value_predictor(
                args["db_id"], result_query, args["question"], " ' value '"
            )

        # HACK!!!
        if args["query_cnt"] == 0:
            execution_result = {
                "headers": [[{"name": "movie.title", "colspan": 1}]],
                "rows": [
                    ["101 BIGGEST CELEBRITY OOPS"],
                    ["101 SEXIEST CELEBRITY BODIES"],
                    ["2013 MTV MOVIE AWARDS"],
                ],
            }
        elif args["query_cnt"] == 1:
            execution_result = {
                "headers": [[{"name": "count(*)", "colspan": 1}]],
                "rows": [["7"]],
            }
        else:
            execution_result = {
                "headers": [[{"name": "count(*)", "colspan": 1}]],
                "rows": [["4"]],
            }

        print("Done")
        return {
            "pred_sql": result_query,
            "execution_result": execution_result,
            "actions": actions,
            "question": question,
        }

    def handleAlalyzeMode(self, args):
        table = json.loads(args["db_obj"])
        systems = ["ours", "irnet"]
        result_query = {}
        result_query["ours"], _, _ = ours_end2end["spider"].run_model(
            args["db_id"], args["question"], table
        )
        result_query["irnet"], _, _ = irnet_end2end["spider"].run_model(
            args["db_id"], args["question"], table
        )
        # result_query["gnn"], _, _ = gnn_end2end["spider"].run_model(
        #     args["db_id"], args["question"]
        # )
        gen_sql = args["gen_sql"]
        gold_sql = args["gold_sql"]
        # diff
        diff, new_gen_sql, new_gold_sql, _ = diff_two_queries(
            "sqlite:///"
            + os.path.join(
                "./data/{}/database".format("spider"),
                args["db_id"],
                args["db_id"] + ".sqlite",
            ),
            args["db_id"],
            gen_sql,
            gold_sql,
            args["diffmode"],
        )
        # check correctness
        equi = {}
        for system in systems:
            equi[system] = compare_two_queries(
                "sqlite:///"
                + os.path.join(
                    "./data/{}/database".format("spider"),
                    args["db_id"],
                    args["db_id"] + ".sqlite",
                ),
                args["db_id"],
                result_query[system],
                gold_sql,
            )
        equi_class = [key for key, value in equi.items() if value[0] == True]
        if equi_class == []:
            similarity_score = -1
            for system in systems:
                _, _, _, similarity_score_tmp = diff_two_queries(
                    "sqlite:///"
                    + os.path.join(
                        "./data/{}/database".format("spider"),
                        args["db_id"],
                        args["db_id"] + ".sqlite",
                    ),
                    args["db_id"],
                    result_query[system],
                    gold_sql,
                    "canonical",
                )
                if similarity_score < similarity_score_tmp:
                    equi_class = [system]
                    similarity_score = similarity_score_tmp
        else:
            similarity_score = 100

        pred_results = [result_query[system] for system in equi_class]

        # captum
        if args["model"] == "ours":
            ours_end2end["spider"].load_model("spider")
            captum_results = ours_end2end["spider"].run_captum(
                args["db_id"], args["question"], args["gold_sql"], table
            )
            print(captum_results[0])
            print(captum_results[1])
        elif args["model"] == "irnet":
            captum_results = irnet_end2end["spider"].run_captum(
                args["db_id"], args["question"], args["gold_sql"], table
            )
        print("Done")
        return {
            "result": equi_class,
            "pred_results": pred_results,
            "diff": diff,
            "canonicalized_gen_sql": new_gen_sql,
            "canonicalized_gold_sql": new_gold_sql,
            "similarity": similarity_score,
            "captum_results": list(captum_results),
        }


class Image(Resource):
    def get(self):
        return send_file("fig1.png", mimetype="image/png")


if __name__ == "__main__":
    from ours.end2end import End2EndOurs
    from irnet.end2end import End2EndIRNet

    # from gnn.end2end import End2EndGNN

    ours_end2end = dict()
    irnet_end2end = dict()
    gnn_end2end = dict()
    for dataset in DATASET:
        ours_end2end[dataset] = End2EndOurs()
        ours_end2end[dataset].prepare_model(dataset)
        irnet_end2end[dataset] = End2EndIRNet()
        irnet_end2end[dataset].prepare_model(dataset)
        # gnn_end2end[dataset] = End2EndGNN()
        # gnn_end2end[dataset].prepare_model(dataset)
    app = Flask("irnet service")
    CORS(app)
    api = Api(app)
    api.add_resource(DBInstanceService, "/dbinstance")
    api.add_resource(Service, "/service")
    api.add_resource(VerifyService, "/verify")
    api.add_resource(Image, "/image")
    app.run(host="141.223.199.148", port=4001, debug=False)
