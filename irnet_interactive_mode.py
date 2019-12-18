import torch
from src import args as arg
from src import utils
from src.models.model import IRNet
from src.rule import semQL
from preprocess.utils import symbol_filter, re_lemma, fully_part_header, group_header, partial_header, num2year, group_symbol, group_values, group_digital
from preprocess.utils import AGG, wordnet_lemmatizer
from preprocess.data_process import process_data_one_entry
from src.utils import process, schema_linking, get_col_table_dict, get_table_colNames
from src.dataset import Example
from src.rule.sem_utils import *
from sem2SQL import transform
import nltk
import os
import pickle
import re
from flask_restful import Resource, reqparse, Api
from flask import Flask
from flask_cors import CORS
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import plotly.tools as tls
import plotly.plotly as ply
import plotly.graph_objs as gro
from pattern.en import lemma
from plotly.offline import plot
import time

with open('preprocess/conceptNet/english_RelatedTo.pkl', 'rb') as f:
    english_RelatedTo = pickle.load(f)

with open('preprocess/conceptNet/english_IsA.pkl', 'rb') as f:
    english_IsA = pickle.load(f)

DBDIR="data/database"
PLOTDIR="web/public/"

AGG_SQL=['count', 'avg', 'min', 'max', 'sum']

def plot_execution(db, sql):

    filename=''
    try:
       sqlite_db='sqlite:///'+db
       disk_engine = create_engine(sqlite_db)
       df=pd.read_sql_query(sql, disk_engine)
       print(df)
       axes=[key for key in df.keys()]
       contents=[ df[key] for key in df.keys() ]
       plot_data={}
       plot_data['layout']={'autosize': True}
       if len(axes) == 1:
          plot_data['data']=[gro.Table(header=dict( values=axes, line_color='darkslategray', fill_color='lightskyblue', align='left'), cells=dict(values=contents, line_color='darkslategray', fill_color='lightcyan', align='left'))]
          plot_data['layout']['margin']={'l': 10, 'r': 10, 'b': 0, 't': 10, 'pad': 0}
       elif len(axes) == 2:
           is_first_agg=False
           is_second_agg=False
           for agg_f in AGG_SQL:
              if agg_f in axes[0].lower() and '(' in axes[0].lower():
                 is_first_agg=True
              if agg_f in axes[1].lower() and '(' in axes[1].lower():
                 is_second_agg=True
           
           if is_first_agg and not is_second_agg:
               plot_data['data']=[gro.Bar(x=df[axes[1]], y=df[axes[0]])]
               plot_data['layout']['xaxis']={'title': axes[1], 'type': 'category'}
               plot_data['layout']['yaxis']={'title': axes[0]}
           elif is_second_agg and not is_first_agg:
               plot_data['data']=[gro.Bar(x=df[axes[0]], y=df[axes[1]])]
               plot_data['layout']['xaxis']={'title': axes[0], 'type': 'category'}
               plot_data['layout']['yaxis']={'title': axes[1]}
           else:
                plot_data['data']=[gro.Table(header=dict( values=axes, line_color='darkslategray', fill_color='lightskyblue', align='left'), cells=dict(values=contents, line_color='darkslategray', fill_color='lightcyan', align='left'))]
                plot_data['layout']['margin']={'l': 10, 'r': 10, 'b': 0, 't': 10, 'pad': 0}
       else:
          plot_data['data']=[gro.Table(header=dict( values=axes, line_color='darkslategray', fill_color='lightskyblue', align='left'), cells=dict(values=contents, line_color='darkslategray', fill_color='lightcyan', align='left'))]
          plot_data['layout']['margin']={'l': 10, 'r': 10, 'b': 0, 't': 10, 'pad': 0}

       filename=os.path.join('execution_results', 'plot'+str(time.strftime("%Y%m%d%H%M%S"))+'.html')
       plot_data['layout']['paper_bgcolor']="#fff"
       plot(plot_data, filename=os.path.join(PLOTDIR, filename), auto_open=False)

    except Exception as e:
       results = e
       filename = ''
       print(e)

    return filename

def execution(db, sql):
    print(db, sql)
    try:
       conn = sqlite3.connect(db)
       cursor = conn.cursor()
       cursor.execute(sql)
       results = cursor.fetchall()
       names = [description[0] for description in cursor.description]
    except Exception as e:
       results = e
       print(e)

    return results, names

def get_runner(model_path):

    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)

    grammar = semQL.Grammar()
    sql_data, table_data, val_sql_data,\
    val_table_data= utils.load_dataset(args.dataset, use_small=args.toy)

    model = IRNet(args, grammar)

    if args.cuda: model.cuda()

    print('load pretrained model from %s'% (model_path))
    pretrained_model = torch.load(model_path,
                                     map_location=lambda storage, loc: storage)
    import copy
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    # model.word_emb = utils.load_word_emb(args.glove_embed_path)
    model.eval()

    db_values = dict()

    with open("data/tables.json") as f:
        schema_tables = json.load(f)
    schema_dict = dict()
    for one_schema in schema_tables:
        schema_dict[one_schema["db_id"]] = one_schema
        schema_dict[one_schema["db_id"]]["only_cnames"] = [c_name.lower() for tid, c_name in one_schema["column_names_original"]]

    for db_id in val_table_data:
        if db_id not in db_values:
            schema_json = schema_dict[db_id]
            primary_foreigns = set()
            for f, p in schema_json["foreign_keys"]:
                primary_foreigns.add(f)
                primary_foreigns.add(p)

            conn = sqlite3.connect("data/database/{}/{}.sqlite".format(db_id, db_id))
            # conn.text_factory = bytes
            cursor = conn.cursor()

            schema = {}

            # fetch table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [str(table[0].lower()) for table in cursor.fetchall()]

            # fetch table info
            for table in tables:
                cursor.execute("PRAGMA table_info({})".format(table))
                schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
            col_value_set = dict()
            for table in tables:
                name_list = [name for _, name in val_table_data[db_id]["column_names"]]
                for col in schema[table]:
                    col_idx = schema_json["only_cnames"].index(col)
                    if col_idx in primary_foreigns and schema_json["column_types"][col_idx] == "number":
                        continue
                    cursor.execute("SELECT \"{}\" FROM \"{}\"".format(col, table))
                    col = name_list[col_idx]
                    value_set = set()
                    try:
                        for val in cursor.fetchall():
                            if isinstance(val[0], str):
                                value_set.add(str(val[0].lower()))
                                value_set.add(lemma(str(val[0].lower())))

                    except:
                        print("not utf8 value")
                    if col in col_value_set:
                        col_value_set[col] |= value_set
                    else:
                        col_value_set[col] = value_set
            db_values[db_id] = col_value_set


    def runner(db_id, nl_string):
        table = val_table_data[db_id]
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]

        entry = {}
        entry['db_id'] = db_id
        entry["question"] = nl_string
        entry["question_toks"] = re.findall(r"[^,.:;\"`?! ]+|[,.:;\"?!]", nl_string.replace("'", " '"))
        entry['names'] = table['schema_content']
        entry['table_names'] = table['table_names']
        entry['col_set'] = table['col_set']
        entry['col_table'] = table['col_table']
        keys = {}
        for kv in table['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in table['primary_keys']:
            keys[id_k] = id_k
        entry['keys'] = keys
        print("Start preprocessing.. {}".format(time.strftime("%Y%m%d-%H%M%S")))
        process_data_one_entry(entry, english_RelatedTo, english_IsA, db_values)
        print("End preprocessing.. {}".format(time.strftime("%Y%m%d-%H%M%S")))

        process_dict = process(entry, table)

        for c_id, col_ in enumerate(process_dict['col_set_iter']):
            for q_id, ori in enumerate(process_dict['q_iter_small']):
                if ori in col_:
                    process_dict['col_set_type'][c_id][0] += 1

        for t_id, tab_ in enumerate(process_dict['table_names']):
            for q_id, ori in enumerate(process_dict['q_iter_small']):
                if ori in tab_:
                    process_dict['tab_set_type'][t_id][0] += 1

        schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                       process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'],
                       process_dict['tab_set_type'], process_dict['table_names'],
                       entry)

        col_table_dict = get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], entry)
        table_col_name = get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])

        process_dict['col_set_iter'][0] = ['count', 'number', 'many']

        rule_label = None

        example = Example(
            src_sent=process_dict['question_arg'],
            col_num=len(process_dict['col_set_iter']),
            vis_seq=(entry['question'], process_dict['col_set_iter'], None),
            tab_cols=process_dict['col_set_iter'],
            tab_iter=process_dict['tab_set_iter'],
            sql=None,
            one_hot_type=process_dict['one_hot_type'],
            col_hot_type=process_dict['col_set_type'],
            tab_hot_type=process_dict['tab_set_type'],
            table_names=process_dict['table_names'],
            table_len=len(process_dict['table_names']),
            col_table_dict=col_table_dict,
            cols=process_dict['tab_cols'],
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            tokenized_src_sent=process_dict['col_set_type'],
            tgt_actions=rule_label
        )
        example.sql_json = copy.deepcopy(entry)
        example.db_id = entry['db_id']
        print("End schema linking and start model running.. {}".format(time.strftime("%Y%m%d-%H%M%S")))

        results_all = model.parse(example, beam_size=5)
        print("End model running and start postprocessing.. {}".format(time.strftime("%Y%m%d-%H%M%S")))
        results = results_all[0]
        list_preds = []
        # list_actions = []
        # list_attentions = []
        try:
            pred = " ".join([str(x) for x in results[0].actions])
            for x in results:
                list_preds.append(" ".join(str(x.actions)))
        except Exception as e:
            pred = ""

        simple_json = example.sql_json['pre_sql']
        simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1])
        simple_json['model_result'] = pred
        simple_json["query"] = None

        print(simple_json)


        alter_not_in_one_entry(simple_json, table)
        alter_inter_one_entry(simple_json)
        alter_column0_one_entry(simple_json)

        try:
            result = transform(simple_json, table)
        except Exception as e:
            result = transform(simple_json, table,
                               origin='Root1(3) Root(5) Sel(0) N(0) A(3) C(0) T(0)')
        sql=result[0] 
        
        def is_number_tryexcept(s):
            try:
               float(s)
               return True
            except ValueError:
               return False
        def first_substring(strings, substring):
            return next(i for i, string in enumerate(strings) if substring in string.lower())
        def find_values(question_arg, question_arg_type, question_origin, mapper):
            values=[]
            flag_double_q=False
            flag_double_q_for_schema=False
            cur_val=[]
            flag_single_q=False
            flag_single_q_for_schema=False
            flag_upper=False
            cur_upper_val=[]
            for idx, (token, tag) in enumerate(zip(question_arg, question_arg_type)):
                start_idx=mapper[idx][0]
                end_idx=mapper[idx][1]
                if len(token) == 0:
                   continue
                if flag_double_q:
                   if '"' not in token[0]:
                      cur_val.append(' '.join(question_origin[start_idx:end_idx]))
                      if tag[0] in ('table', 'col'):
                          flag_double_q_for_schema=True
                      continue
                if flag_single_q:
                   if "'" not in token[0]:
#                      for i, t in enumerate(token):
#                          idx = first_substring( question_origin[start_idx:end_idx], t )
#                          if idx != -1:
#                              token[i]=question_origin[idx]
                      cur_val.append(' '.join(question_origin[start_idx:end_idx]))
                      if tag[0] in ('table', 'col'):
                          flag_single_q_for_schema=True
                      continue

                if flag_upper:
                    # If Jason 'Two ... separate
                    if len(question_origin[start_idx]) > 0 and question_origin[start_idx][0].isupper() and tag[0] not in ('col', 'table'):
                        cur_upper_val.append(' '.join(question_origin[start_idx:end_idx]))
                        continue
                    else:
                        values.append(' '.join(cur_upper_val))
                        cur_upper_val=[]
                        flag_upper=False
                     
                              
                is_inserted_already=False
                if len(token) == 1 and token[0] == 'year' and is_number_tryexcept(question_origin[start_idx]):
                    is_inserted_already=True
                    values.append(question_origin[start_idx])
                 
                if '"' in token[0]:
                    if flag_double_q:
                       is_inserted_already=True
                       flag_double_q=False
                       if not flag_double_q_for_schema:
                          values.append(' '.join(cur_val)) 
                       cur_val=[]
                       flag_double_q_for_schema=False
                    elif len(token[0]) == 1:
                       is_inserted_already=True
                       flag_double_q=True
                elif "'" in token[0]:
                    if flag_single_q:
                       is_inserted_already=True
                       flag_single_q=False
                       if not flag_single_q_for_schema:
                          values.append(' '.join(cur_val)) 
                       cur_val=[]
                       flag_single_q_for_schema=False
                    elif len(token[0]) == 1:
                       is_inserted_already=True
                       flag_single_q=True

                if (not is_inserted_already) and len(question_origin[start_idx]) > 0 and question_origin[start_idx][0].isupper() and start_idx != 0:
                     if tag[0] not in ('col', 'table'):
                         is_inserted_already=True
                         flag_upper=True
                         cur_upper_val.append(' '.join(question_origin[start_idx:end_idx]))
                if (not is_inserted_already) and tag[0] in ('value', '*', 'db'):
                    is_inserted_already=True
                    values.append(' '.join(question_origin[start_idx:end_idx]))
            return values
        print("question_arg: {} \n question_arg_type: {} \n original_question: {} \n mapper: {}".format(simple_json['question_arg'], simple_json['question_arg_type'], simple_json['origin_question_toks_for_value'], simple_json['mapper']))
        sql_values=find_values(simple_json['question_arg'], simple_json['question_arg_type'], simple_json['origin_question_toks_for_value'], simple_json['mapper'])
        print(sql_values)
        cur_index = sql.find(' 1')
        sql_with_value=""
        before_index = 0
        values_index=0
        while cur_index != -1 and values_index < len(sql_values):
           sql_with_value=sql_with_value+sql[before_index:cur_index]
           if sql[cur_index-1] in ('=', '>', '<'):
              cur_value=sql_values[values_index]
              values_index=values_index+1
              if not is_number_tryexcept(cur_value):
                  cur_value='"'+cur_value+'"'
              sql_with_value=sql_with_value+' '+cur_value
           elif cur_index-3 > 0 and sql[cur_index-4:cur_index] in ('like'):
              cur_value='%'+sql_values[values_index]+'%'
              values_index=values_index+1
              if not is_number_tryexcept(cur_value):
                  cur_value='"'+cur_value+'"'
              sql_with_value=sql_with_value+' '+cur_value
           elif cur_index-6 > 0 and sql[cur_index-7:cur_index] in ('between'):
              if values_index + 1 < len(sql_values):
                   cur_value1=sql_values[values_index]
                   values_index=values_index+1
                   cur_value2=sql_values[values_index]
                   values_index=values_index+1
              else:
                 cur_value1=sql_values[values_index]
                 cur_value2=sql_values[values_index]
                 values_index=values_index+1
              if not is_number_tryexcept(cur_value1):
                  cur_value1='1'
              if not is_number_tryexcept(cur_value2):
                  cur_value2='2'
              sql_with_value=sql_with_value+' '+cur_value1+' AND '+cur_value2
              cur_index=cur_index+6
           else:
              sql_with_value=sql_with_value+sql[cur_index:cur_index+2]
           before_index=cur_index+2
           cur_index=sql.find(' 1', cur_index+1)
        sql_with_value=sql_with_value+sql[before_index:]
        print(sql_with_value)
#        ex_result, ex_result_header=execution(os.path.join(DBDIR, db_id, db_id + ".sqlite"), sql)
        print("End post processing and start plotting.. {}".format(time.strftime("%Y%m%d-%H%M%S")))
        plot_filename=plot_execution(os.path.join(DBDIR, db_id, db_id + ".sqlite"), sql_with_value) 
        print("End plotting.. {}".format(time.strftime("%Y%m%d-%H%M%S")))
        return sql_with_value, None, entry["question_toks"], plot_filename

    return runner


runner = get_runner("saved_model/best_model_leaderboard.model")


class Service(Resource):
    def get(self):
        # try:
            parser = reqparse.RequestParser()
            parser.add_argument('db_id', required=True, type=str)
            parser.add_argument('question', required=True, type=str)
            args = parser.parse_args()
            print("done well")
            result_query, actions, question, plot_filename = runner(args["db_id"], args["question"])
            if plot_filename == '': 
                return {'result': result_query,
                    'actions': actions,
                    'question': question}
            else:
                return {'result': result_query,
                    'actions': actions,
                    'question': question,
                    'plot_filename': plot_filename}
        # except Exception as e:
        #     print("done not well")
        #     return {'result': str(e)}

app = Flask('irnet service')
CORS(app)
api = Api(app)
api.add_resource(Service, '/service')

if __name__ == "__main__":
    # res, act, att = runner("dog_kennels", "Which professionals have done at least two treatments? List the professional's id, role, and first name.")
    # print(res)
    # print(act)
    # print(att)
    app.run(host='141.223.199.148', port=5000, debug=False)
