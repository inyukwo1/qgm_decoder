import torch
import json
from ours.src import args as arg, utils
from ours.src.models.model import IRNet
from ours.src.rule import semQL
from ours.preprocess.data_process import process_data_one_entry
from ours.src.utils import process, schema_linking, get_col_table_dict, get_table_colNames
from ours.src.dataset import Example
from ours.sem2SQL import transform
from ours.src.rule.sem_utils import alter_column0_one_entry, alter_inter_one_entry, alter_not_in_one_entry
import os
import pickle
import re
from flask_restful import Resource, reqparse, Api
from flask import Flask
from flask_cors import CORS
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objs as gro
from pattern.en import lemma
from plotly.offline import plot
import time
from scp import SCPClient
import paramiko
from abc import *

PLOTDIR = "web/public/"

AGG_SQL = ['count', 'avg', 'min', 'max', 'sum']
DATASET = ['spider']


def plot_execution(db, sql):
    filename = ''
    try:
        sqlite_db = 'sqlite:///' + db
        disk_engine = create_engine(sqlite_db)
        df = pd.read_sql_query(sql, disk_engine)
        print(df)
        axes = [key for key in df.keys()]
        contents = [df[key] for key in df.keys()]
        plot_data = {}
        plot_data['layout'] = {'autosize': True}
        if len(axes) == 1:
            plot_data['data'] = [
                gro.Table(header=dict(values=axes, line_color='darkslategray', fill_color='lightskyblue', align='left'),
                          cells=dict(values=contents, line_color='darkslategray', fill_color='lightcyan',
                                     align='left'))]
            plot_data['layout']['margin'] = {'l': 10, 'r': 10, 'b': 0, 't': 10, 'pad': 0}
        elif len(axes) == 2:
            is_first_agg = False
            is_second_agg = False
            for agg_f in AGG_SQL:
                if agg_f in axes[0].lower() and '(' in axes[0].lower():
                    is_first_agg = True
                if agg_f in axes[1].lower() and '(' in axes[1].lower():
                    is_second_agg = True

            if is_first_agg and not is_second_agg:
                plot_data['data'] = [gro.Bar(x=df[axes[1]], y=df[axes[0]])]
                plot_data['layout']['xaxis'] = {'title': axes[1], 'type': 'category'}
                plot_data['layout']['yaxis'] = {'title': axes[0]}
            elif is_second_agg and not is_first_agg:
                plot_data['data'] = [gro.Bar(x=df[axes[0]], y=df[axes[1]])]
                plot_data['layout']['xaxis'] = {'title': axes[0], 'type': 'category'}
                plot_data['layout']['yaxis'] = {'title': axes[1]}
            else:
                plot_data['data'] = [gro.Table(
                    header=dict(values=axes, line_color='darkslategray', fill_color='lightskyblue', align='left'),
                    cells=dict(values=contents, line_color='darkslategray', fill_color='lightcyan', align='left'))]
                plot_data['layout']['margin'] = {'l': 10, 'r': 10, 'b': 0, 't': 10, 'pad': 0}
        else:
            plot_data['data'] = [
                gro.Table(header=dict(values=axes, line_color='darkslategray', fill_color='lightskyblue', align='left'),
                          cells=dict(values=contents, line_color='darkslategray', fill_color='lightcyan',
                                     align='left'))]
            plot_data['layout']['margin'] = {'l': 10, 'r': 10, 'b': 0, 't': 10, 'pad': 0}

        filename = os.path.join('execution_results', 'plot' + str(time.strftime("%Y%m%d%H%M%S")) + '.html')
        plot_data['layout']['paper_bgcolor'] = "#fff"
        plot(plot_data, filename=os.path.join(PLOTDIR, filename), auto_open=False)

    except Exception as e:
        results = e
        filename = ''
        print(e)

    return filename


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


ssh = createSSHClient('141.223.199.39', '2022', 'hjkim', 'sksmsdi!wkfTodrlszlaguswl33')
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
            parser.add_argument('model', required=True, type=str)
            parser.add_argument('db_id', required=True, type=str)
            parser.add_argument('question', required=True, type=str)
            args = parser.parse_args()

            if args['model'] == 'ours':
                model = ours_end2end["spider"]
            elif args['model'] == 'irnet':
                model = irnet_end2end["spider"]
            elif args['model'] == 'gnn':
                model = gnn_end2end["spider"]
            else:
                return
            result_query, actions, question = model.run_model(args["db_id"], args["question"])
            plot_filename = plot_execution(os.path.join("./data/{}/database".format("spider"), args["db_id"], args["db_id"] + ".sqlite"), result_query)

            if plot_filename == '':
                return {'result': result_query,
                        'actions': actions,
                        'question': question}
            else:
                scp.put(os.path.join(PLOTDIR, plot_filename),
                        os.path.join('/data1/Benchmark_RA/irnet/web/public', plot_filename))
                return {'result': result_query,
                        'actions': actions,
                        'question': question,
                        'plot_filename': plot_filename}
        except Exception as e:
            print("done not well")
            return {'result': str(e)}


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
    app = Flask('irnet service')
    CORS(app)
    api = Api(app)
    api.add_resource(Service, '/service')
    app.run(host='141.223.199.148', port=4000, debug=False)
