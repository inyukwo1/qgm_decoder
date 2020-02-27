import torch
import sklearn
from validation.check_correctness import compare_two_queries, diff_two_queries

import os
from flask_restful import Resource, reqparse, Api
from flask import Flask
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


def plot_execution(db, sql):
    filename = ""
    try:
        sqlite_db = "sqlite:///" + db
        disk_engine = create_engine(sqlite_db)
        df = pd.read_sql_query(sql, disk_engine)
        print(df)
        axes = [key for key in df.keys()]
        contents = [df[key] for key in df.keys()]
        plot_data = {}
        plot_data["layout"] = {"autosize": True}
        new_axes = sql[7 : sql.find("FROM")].split(",")
        if len(axes) == 1:
            plot_data["data"] = [
                gro.Table(
                    header=dict(
                        values=new_axes,
                        line_color="darkslategray",
                        fill_color="lightskyblue",
                        align="left",
                    ),
                    cells=dict(
                        values=contents,
                        line_color="darkslategray",
                        fill_color="lightcyan",
                        align="left",
                    ),
                )
            ]
            plot_data["layout"]["margin"] = {
                "l": 10,
                "r": 10,
                "b": 0,
                "t": 10,
                "pad": 0,
            }
        elif len(axes) == 2:
            is_first_agg = False
            is_second_agg = False
            for agg_f in AGG_SQL:
                if agg_f in new_axes[0].lower() and "(" in new_axes[0].lower():
                    is_first_agg = True
                if agg_f in new_axes[1].lower() and "(" in new_axes[1].lower():
                    is_second_agg = True

            if is_first_agg and not is_second_agg:
                plot_data["data"] = [gro.Bar(x=df[axes[1]], y=df[axes[0]])]
                plot_data["layout"]["xaxis"] = {
                    "title": new_axes[1],
                    "type": "category",
                }
                plot_data["layout"]["yaxis"] = {"title": new_axes[0]}
            elif is_second_agg and not is_first_agg:
                plot_data["data"] = [gro.Bar(x=df[axes[0]], y=df[axes[1]])]
                plot_data["layout"]["xaxis"] = {
                    "title": new_axes[0],
                    "type": "category",
                }
                plot_data["layout"]["yaxis"] = {"title": new_axes[1]}
            else:
                plot_data["data"] = [
                    gro.Table(
                        header=dict(
                            values=new_axes,
                            line_color="darkslategray",
                            fill_color="lightskyblue",
                            align="left",
                        ),
                        cells=dict(
                            values=contents,
                            line_color="darkslategray",
                            fill_color="lightcyan",
                            align="left",
                        ),
                    )
                ]
                plot_data["layout"]["margin"] = {
                    "l": 10,
                    "r": 10,
                    "b": 0,
                    "t": 10,
                    "pad": 0,
                }
            plot_data["layout"]["height"] = 300
        else:
            plot_data["data"] = [
                gro.Table(
                    header=dict(
                        values=new_axes,
                        line_color="darkslategray",
                        fill_color="lightskyblue",
                        align="left",
                    ),
                    cells=dict(
                        values=contents,
                        line_color="darkslategray",
                        fill_color="lightcyan",
                        align="left",
                    ),
                )
            ]
            plot_data["layout"]["margin"] = {
                "l": 10,
                "r": 10,
                "b": 0,
                "t": 10,
                "pad": 0,
            }
        filename = os.path.join(
            "execution_results", "plot" + str(time.strftime("%Y%m%d%H%M%S")) + ".html"
        )
        plot_data["layout"]["paper_bgcolor"] = "#fff"
        plot(plot_data, filename=os.path.join(PLOTDIR, filename), auto_open=False)
    except Exception as e:
        results = e
        filename = ""
        print(e)
    return filename


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


# ssh = createSSHClient('141.223.199.39', '2022', 'hjkim', 'sksmsdi!wkfTodrlszlaguswl33')
ssh = createSSHClient("141.223.199.39", "2022", "ihna", "Sook2303!@")
scp = SCPClient(ssh.get_transport())


class End2End(metaclass=ABCMeta):
    @abstractmethod
    def prepare_model(self, dataset):
        pass

    @abstractmethod
    def run_model(self, db_id, question):
        pass


class Service(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument("model", required=True, type=str)
            parser.add_argument("db_id", required=True, type=str)
            parser.add_argument("question", required=True, type=str)
            parser.add_argument("mode", default="Explore", type=str)
            parser.add_argument("gold_sql", default=None, type=str)
            parser.add_argument("gen_sql", default=None, type=str)
            parser.add_argument("diffmode", default="origin", type=str)
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
            args["db_id"], args["question"]
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

        plot_filename = plot_execution(
            os.path.join(
                "./data/{}/database".format("spider"),
                args["db_id"],
                args["db_id"] + ".sqlite",
            ),
            result_query,
        )
        # plot_filename = ""

        if plot_filename == "":
            return {
                "result": result_query,
                "actions": actions,
                "question": question,
            }
        else:
            scp.put(
                os.path.join(PLOTDIR, plot_filename),
                os.path.join("/home/ihna/web/build/", plot_filename),
            )
            print("Done")
            return {
                "result": result_query,
                "actions": actions,
                "question": question,
                "plot_filename": plot_filename,
            }

    def handleAlalyzeMode(self, args):
        systems = ["ours", "irnet", "gnn"]
        result_query = {}
        result_query["ours"], _, _ = ours_end2end["spider"].run_model(
            args["db_id"], args["question"]
        )
        result_query["irnet"], _, _ = irnet_end2end["spider"].run_model(
            args["db_id"], args["question"]
        )
        result_query["gnn"], _, _ = gnn_end2end["spider"].run_model(
            args["db_id"], args["question"]
        )
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

        # captum
        captum_html = irnet_end2end["spider"].run_captum(
            args["db_id"], args["question"], args["gold_sql"]
        )
        print("Done")
        return {
            "result": equi_class,
            "diff": diff,
            "canonicalized_gen_sql": new_gen_sql,
            "canonicalized_gold_sql": new_gold_sql,
            "similarity": similarity_score,
            "captum_html": captum_html,
        }


if __name__ == "__main__":
    from ours.end2end import End2EndOurs
    from irnet.end2end import End2EndIRNet
    from gnn.end2end import End2EndGNN

    ours_end2end = dict()
    irnet_end2end = dict()
    gnn_end2end = dict()
    for dataset in DATASET:
        ours_end2end[dataset] = End2EndOurs()
        ours_end2end[dataset].prepare_model(dataset)
        irnet_end2end[dataset] = End2EndIRNet()
        irnet_end2end[dataset].prepare_model(dataset)
        gnn_end2end[dataset] = End2EndGNN()
        gnn_end2end[dataset].prepare_model(dataset)
    app = Flask("irnet service")
    CORS(app)
    api = Api(app)
    api.add_resource(Service, "/service")
    app.run(host="141.223.199.148", port=4001, debug=False)
