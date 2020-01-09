#!/usr/bin/python
# -*- coding: utf-8 -*-
import decimal
import time
import datetime
import simplejson as json
import MySQLdb
import sqlite3
import os
import re
import subprocess
import sys
import traceback
from collections import Counter
from contextlib import closing
from Cosette import solver
from itertools import izip, product, permutations
from string import Template
import argparse
import shutil
import warnings
import csv
import requests


reload(sys)
sys.setdefaultencoding('utf8')

HOST = 'localhost'
USER = 'root'
PASSWD = 'root'

USE_TEST_DATA_GEN = False
TMP_DATA_PATH = None
TMP_DATABASE_NAME = 'databasetmp'
TIMEOUT = 1000 * 1000

COSETTE_API_KEY = '87e29bc3086947aac67af14b3df6a1e0'

MYSQL_DUMP_DIR = './dataset/mysql_dump/'
DATABASES = [
    'advising-querysplit',
    'advising-querysplit_rename',
    'advising-questionsplit',
    'advising-questionsplit_rename',
    'atis',
    'atis_rename',
    'geo',
    'geo_rename',
    'imdb',
    'imdb_rename',
    'mas',
    'patients',
    'patients_rename',
    'restaurant',
    'scholar',
    'wikisql',
    'wikitablequestions',
    'yelp',
    'yelp_rename',
]

GOLD_QUERY_DIR = './dataset_todo/gold_queries/'
LOG_DIR = './dataset_todo/logs/'
TEMPLATE_DIR = './cosette_template/'

MESSAGE_HLINE = '--------------------------------------------------------------------------------\n'
MESSAGE_BOLD = '\033[1m'
MESSAGE_RED = '\033[31m'
MESSAGE_GREEN = '\033[32m'
MESSAGE_YELLOW = '\033[33m'
MESSAGE_RESET = '\033[0m'
MESSAGE_DONE = '[' + MESSAGE_BOLD + MESSAGE_GREEN + 'DONE' + MESSAGE_RESET + ']'
MESSAGE_STOPPED = '[' + MESSAGE_BOLD + MESSAGE_YELLOW + 'STOPPED' + MESSAGE_RESET + ']'
MESSAGE_FAILED = '[' + MESSAGE_BOLD + MESSAGE_RED + 'FAILED' + MESSAGE_RESET + ']'

CREATE_BENCHMARK_TABLE_SQLS = None


def get_db_name(dbname):
    if USE_TEST_DATA_GEN:
       return TMP_DATABASE_NAME
    return dbname.replace('-', '')
def print_log_message(level, messages):

    timestamp = str(datetime.datetime.now())
    print timestamp + '\t' + level + '\t' + '\t'.join(messages)


def get_sqlite3_execute_result(query):
    spider_dbname = []
    query_toks = query.split()
    for idx, tok in enumerate(query_toks):
        if '__' in tok:
            dbname, table_name = tok.split('__')
            spider_dbname.append(dbname)
            query_toks[idx] = table_name
    query = ' '.join(query_toks)
    if len(query_toks) == 0:
        raise Exception("No query exists")
    elif len(spider_dbname) == 0:
        raise Exception("Cannot find spider dbname")
    elif len(set(spider_dbname)) > 1:
        raise Exception("Too many spider dbname")
    spider_dbname = spider_dbname[0]

    sqlite_db_dir = '/mnt/disk1/Benchmark_RA/in_progress/data/spider/origin/database/'
    for dirname in os.listdir(sqlite_db_dir):
        if dirname.lower() == spider_dbname:
            spider_dbname = dirname
            break
    sqlite_db_path = sqlite_db_dir + '{db}/{db}.sqlite'.format(db=spider_dbname)
    conn = sqlite3.connect(sqlite_db_path)
    conn.text_factory = lambda x: str(x).decode('utf8', errors='ignore')
    cu = conn.cursor()
    cu.execute(query)
    output = cu.fetchall()
    conn.close()
    return output


def create_benchmark_db(dbconn):
    print_log_message('INFO', ['create_benchmark_db', 'called'])
    num_created = 0
    if not CREATE_BENCHMARK_TABLE_SQLS:
        print_log_message('ERROR', ['create_benchmark_db', 'SQL does not exist'])
        return

    try:
        with closing(dbconn.cursor()) as cu:
            # cu.execute('''DROP DATABASE IF EXISTS benchmark''')
            cu.execute('''CREATE DATABASE benchmark''')
            cu.execute('''USE benchmark''')
            for sql in CREATE_BENCHMARK_TABLE_SQLS:
                cu.execute(sql)
                num_created += 1
    except Exception as e:
        print_log_message('ERROR', ['create_benchmark_db', str(e)])
        return
    print_log_message('INFO', ['create_benchmark_db', 'done', str(num_created) + ' table(s) created'])


def load_mysql_dump(dbconn, dbname, mysqldumpname, mysqldumppath, drop=False):
    try:
        with closing(dbconn.cursor()) as cu:
            # check if db exists
            cu.execute('''SHOW DATABASES LIKE "''' + dbname + '"')
            if cu.rowcount != 0:
                # drop the db
                # cu.execute('DROP DATABASE ' + db)

                # continue if db exists
                if drop:
                    if dbname in DATABASES:
                       raise Exception('You cannot drop '+dbname)
                    cu.execute('''DROP DATABASE ''' + dbname)
                else:
                    raise Exception(dbname + ' already exists')

            # create db
            cu.execute('''CREATE DATABASE ''' + dbname)

            # load db
            with open(mysqldumppath + mysqldumpname + '.mysqldump') as f:
               subprocess.call(['mysql', '-u', USER, '-p' + PASSWD, '-h', HOST, dbname], stdin=f)
    except Exception as e:
        print MESSAGE_FAILED
        print MESSAGE_RED + str(e) + MESSAGE_RESET
        return
    print MESSAGE_DONE

def load_mysql_dumps_to_tmpdb(dbconn):
    print_log_message('INFO', ['load_mysql_dumps_to_tmpdb', 'create database '+TMP_DATABASE_NAME])
    #load_mysql_dump(dbconn, TMP_DATABASE_NAME, 'schema', TMP_DATA_PATH, drop=True)

def load_mysql_dumps(dbconn):
    # TODO: outdated, slow
    print MESSAGE_HLINE + MESSAGE_BOLD + 'Loading MySQL dump files' + MESSAGE_RESET
    num_loaded = 0
    for db in DATABASES:
        db=db.replace('-', '')
        print 'Loading ' + db
        load_mysql_dump(dbconn, db, db, MYSQL_DUMP_DIR)
        num_loaded += 1
    print MESSAGE_BOLD + 'Loading MySQL dump files' + MESSAGE_RESET,
    print '(' + str(num_loaded) + ' file(s) loaded)'


def format_sql(sql):
    # aggregate functions
    agg_funcs = ['MAX', 'MIN', 'COUNT', 'AVG', 'SUM', 'max', 'min', 'count', 'avg', 'sum']
    for func in agg_funcs:
        sql = sql.replace(func + ' (', func + '(')

    # inequalities
    sql = sql.replace('< >', '<>')
    sql = sql.replace('> =', '>=')
    sql = sql.replace('< =', '<=')

    return sql


def load_gold_queries(dbconn):
    print_log_message('INFO', ['load_gold_queries', 'called'])
    num_inserted = 0
    num_updated = 0
    for filename in os.listdir(GOLD_QUERY_DIR):
        print_log_message('INFO', ['load_gold_queries', 'start loading ' + filename])
        if not filename.endswith('.tsv'):
            print_log_message('INFO', ['load_gold_queries', filename + ' not ends with \'.tsv\''])
            continue

        num_inserted_tmp = 0
        num_updated_tmp = 0
        try:
            category = filename.split('_')[1]
            if filename.endswith('_test.tsv'):
                split = "test"
            elif filename.endswith('_train.tsv'):
                split = "train"
            elif filename.endswith('_dev.tsv'):
                split = "dev"
            else:
                raise Exception('no split suffix: ' + filename)

            dbconn.begin()
            with closing(dbconn.cursor()) as cu:
                cu.execute('''USE benchmark''')

                with open(GOLD_QUERY_DIR + filename) as f:
                    next(f)  # ignore the first line
                    for line in f:
                        line_split = line.replace("\\", "\\\\").replace('\"','\\"').split('\t')

                        # check if there are five columns
                        if len(line_split) < 5:
                            raise Exception('invalid line: ' + line)

                        db = line_split[0].lower()
                        query_index = line_split[1]
                        nl = line_split[3]
                        gold_sql = format_sql(line_split[2])
                        gold_sql_cosette = format_sql(line_split[4])
                        gold_sql_cosette_rename = format_sql(line_split[5])
                        gold_sql_execute = format_sql(line_split[6])

                        # insert to query and query_split
                        cu.execute('''
                            SELECT S.query_id,
                                   Q.gold_sql_execute
                            FROM   query_split S,
                                   query Q
                            WHERE  S.db="%s"
                               AND S.query_id=Q.query_id
                               AND S.query_index=%s
                               AND S.category="%s"
                            ''' % (db, query_index, category))
                        if cu.rowcount == 0:
                            cu.execute('''
                                INSERT INTO query (db, query_index, nl, gold_sql, gold_sql_cosette, gold_sql_cosette_rename, gold_sql_execute)
                                VALUES ("%s", %s, "%s", "%s", "%s", "%s", "%s")
                                ''' % (db, query_index, nl, gold_sql, gold_sql_cosette, gold_sql_cosette_rename, gold_sql_execute))
                            query_id = cu.lastrowid

                            cu.execute('''
                                INSERT INTO query_split (query_id, db, query_index, split, is_simple, category)
                                VALUES (%s, "%s", %s, "%s", NULL, "%s")
                                ''' % (query_id, db, query_index, split, category))
                            num_inserted_tmp += 1
                        else:
                            query_id, old_gold_sql_execute = cu.fetchone()
                         #   show='''
                         #       UPDATE query
                         #       SET    db="%s",
                         #              query_index=%s,
                         #              nl="%s",
                         #              gold_sql="%s",
                         #              gold_sql_cosette="%s",
                         #              gold_sql_cosette_rename="%s",
                         #              gold_sql_execute="%s"
                         #       WHERE  query_id=%s''' % (db, query_index, nl, gold_sql, gold_sql_cosette, gold_sql_cosette_rename, gold_sql_execute, query_id)
                         #   print(show)
                            old_gold_sql_execute = old_gold_sql_execute.replace("\\", "\\\\").replace('\"','\\"')
                            if old_gold_sql_execute.strip() == gold_sql_execute.strip():
                                cu.execute('''
                                    UPDATE query
                                    SET    db="%s",
                                           query_index=%s,
                                           nl="%s",
                                           gold_sql="%s",
                                           gold_sql_cosette="%s",
                                           gold_sql_cosette_rename="%s",
                                           gold_sql_execute="%s"
                                    WHERE  query_id=%s
                                    ''' % (db, query_index, nl, gold_sql, gold_sql_cosette, gold_sql_cosette_rename, gold_sql_execute, query_id))
                            else:
                                cu.execute('''
                                    UPDATE query
                                    SET    db="%s",
                                           query_index=%s,
                                           nl="%s",
                                           gold_sql="%s",
                                           gold_sql_cosette="%s",
                                           gold_sql_cosette_rename="%s",
                                           gold_sql_execute="%s",
                                           sql_execution_result_id=null
                                    WHERE  query_id=%s
                                    ''' % (db, query_index, nl, gold_sql, gold_sql_cosette, gold_sql_cosette_rename, gold_sql_execute, query_id))
                            
                            #cu.execute('''
                            #    UPDATE query_split
                            #    SET    db="%s",
                            #           query_index=%s,
                            #           split="%s",
                            #           is_simple=NULL,
                            #           category="%s"
                            #    WHERE  query_id=%s
                            #    ''' % (db, query_index, split, category, query_id))
                            num_updated_tmp += 1

            dbconn.commit()
        except Exception as e:
            print_log_message(
                'ERROR', ['load_gold_queries', 'loading ' + filename + ' failed', str(e)])
            try:
                dbconn.rollback()
            except:
                pass
            continue
        print_log_message('INFO', ['load_gold_queries', 'loading ' + filename + ' done',
                                   str(num_inserted_tmp) + ' query/queries inserted',
                                   str(num_updated_tmp) + ' query/queries updated',
                                   ])
        num_inserted += num_inserted_tmp
        num_updated += num_updated_tmp
    print_log_message('INFO', ['load_gold_queries', 'loading done',
                               str(num_inserted) + ' query/queries inserted',
                               str(num_updated) + ' query/queries updated',
                               ])


def load_generated_queries(dbconn, filename):
    print_log_message('INFO', ['load_generated_queries', 'start loading ' + filename])
    num_inserted = 0
    num_updated = 0
    query_ids = []
    try:
        dbconn.begin()
        with closing(dbconn.cursor()) as cu:
            cu.execute('''USE benchmark''')
            with open(filename) as f:
                filename = filename.split('/')[-1]
                filename_split = filename.replace('.', '_').split('_')
                system = filename_split[0]
                training_data = filename_split[1]
                test_data = filename_split[2]
                category = filename_split[3]

                # insert (system, training_data, test_data, category) to experiment_info,
                # and get experiment_id
                cu.execute('''
                    SELECT experiment_id
                    FROM   experiment_info
                    WHERE  system="%s"
                       AND training_data="%s"
                       AND test_data="%s"
                       AND category="%s"
                    ''' % (system, training_data, test_data, category))
                if cu.rowcount == 0:
                    cu.execute('''
                        INSERT INTO experiment_info (system, training_data, test_data, category)
                        VALUES ("%s", "%s", "%s", "%s")
                        ''' % (system, training_data, test_data, category))
                    experiment_id = cu.lastrowid
                else:
                    experiment_id = cu.fetchone()[0]

                next(f)  # ignore the first line
                for line in f:
                    line_split = line.replace('\\', '\\\\').replace('"', '\\"').split('\t')

                    db = line_split[0].lower()
                    if db.endswith('_rename'):
                        db, db_suffix = db.split('_')
                    else:
                        db_suffix = ''
                    query_index = line_split[1]
                    gen_sql = format_sql(line_split[2] if len(line_split) >= 3 else '')
                    gen_sql_execute = format_sql(line_split[3] if len(line_split) >= 4 else '')
                    gen_sql_cosette = format_sql(
                        line_split[4] if len(line_split) >= 5 else gen_sql_execute)

                    # get query_id
                    cu.execute('''
                        SELECT query_id
                        FROM   query_split
                        WHERE  db="%s"
                           AND query_index=%s
                           AND category="%s"
                        ''' % (db, query_index, category))
                    if cu.rowcount == 0:
                        raise Exception('gold query (' + db + ', ' + query_index + ', ' + category + ') not found')
                    else: 
                        query_id = cu.fetchone()[0]

                    # insert (experiment_id, query_id, gen_sql, gen_sql_execute, db_suffix) to
                    # query_result
                    cu.execute('''
                        SELECT *
                        FROM   query_result
                        WHERE  experiment_id=%s
                           AND query_id=%s
                        ''' % (experiment_id, query_id))
                    if cu.rowcount == 0:
                        cu.execute('''
                            INSERT INTO query_result (experiment_id, query_id, gen_sql, gen_sql_execute, gen_sql_cosette, db_suffix)
                            VALUES (%s, %s, "%s", "%s", "%s", "%s")
                            ''' % (experiment_id, query_id, gen_sql, gen_sql_execute, gen_sql_cosette, db_suffix))
                        num_inserted += 1
                    else:
                        cu.execute('''
                            UPDATE query_result
                            SET    gen_sql="%s",
                                   gen_sql_execute="%s",
                                   gen_sql_cosette="%s",
                                   db_suffix="%s"
                            WHERE  experiment_id=%s
                               AND query_id=%s
                            ''' % (gen_sql, gen_sql_execute, gen_sql_cosette, db_suffix, experiment_id, query_id))
                        num_updated += 1
                    query_ids.append(query_id)
        dbconn.commit()
    except Exception as e:
        print_log_message('ERROR', ['load_generated_queries', 'loading failed', str(e)])
        try:
            dbconn.rollback()
        except:
            pass
    print_log_message('INFO', ['load_generated_queries', 'done',
                               str(num_inserted) + ' query/queries inserted',
                               str(num_updated) + ' query/queries updated',
                               ])
    return experiment_id, system, query_ids


def load_logs(dbconn):
    print MESSAGE_HLINE + MESSAGE_BOLD + 'Loading log files' + MESSAGE_RESET
    num_loaded = 0
    for filename in os.listdir(LOG_DIR):
        if not filename.endswith('.log'):
            continue

        print 'Loading ' + filename,
        is_loaded = False
        try:
            dbconn.begin()
            with closing(dbconn.cursor()) as cu:
                cu.execute('''USE benchmark''')

                filename_split = filename.replace('.', '_').split('_')
                system = filename_split[0]
                training_data = filename_split[1]
                test_data = filename_split[2]
                category = filename_split[3]

                # insert (system, training_data, test_data, category) to experiment_info, and get
                # experiment_id
                cu.execute('''
                    SELECT experiment_id
                    FROM   experiment_info
                    WHERE  system="%s"
                       AND training_data="%s"
                       AND test_data="%s"
                       AND category="%s"
                    ''' % (system, training_data, test_data, category))
                results = cu.fetchall()
                if results:
                    experiment_id = results[0][0]
                else:
                    cu.execute('''
                        INSERT INTO experiment_info (system, training_data, test_data, category)
                        VALUES ("%s", "%s", "%s", "%s")
                        ''' % (system, training_data, test_data, category))
                    experiment_id = cu.lastrowid

                # insert (experiment_id, log_path) to query_result
                cu.execute('''
                    SELECT experiment_id
                    FROM   log
                    WHERE  experiment_id=%s
                    ''' % (experiment_id))
                if cu.rowcount == 0:
                    cu.execute('''
                        INSERT INTO log (experiment_id, log_path)
                        VALUES (%s, "%s")
                        ''' % (experiment_id, LOG_DIR + filename))
                    is_loaded = True
                else:
                    cu.execute(
                        '''
                        UPDATE log
                        SET    log_path="%s"
                        WHERE  experiment_id=%s
                        ''' % (LOG_DIR + filename, experiment_id))
                    is_loaded = True
            dbconn.commit()
        except Exception as e:
            print MESSAGE_FAILED
            print MESSAGE_RED + str(e) + MESSAGE_RESET
            try:
                dbconn.rollback()
            except:
                pass
            continue
        print MESSAGE_DONE
        num_loaded += 1 if is_loaded else 0
    print MESSAGE_BOLD + 'Loading log files' + MESSAGE_RESET,
    print '(' + str(num_loaded) + ' file(s) loaded)'


def insert_accuracy(dbconn, filename):
    print_log_message('INFO', ['insert_accuracy', 'start loading ' + filename])
    num_inserted = 0
    num_updated = 0
    try:
        dbconn.begin()
        with closing(dbconn.cursor()) as cu:
            cu.execute('''USE benchmark''')
            with open(filename) as f:
                filename = filename.split('/')[-1]
                filename_split = filename.replace('.', '_').split('_')
                system = filename_split[0]
                training_data = filename_split[1]
                test_data = filename_split[2]
                category = filename_split[3]
                print("INFO: {}, {}, {}, {}".format(system, training_data, test_data, category))
               
                # get experiment_id
                cu.execute('''
                    SELECT experiment_id
                    FROM   experiment_info
                    WHERE  system="%s"
                       AND training_data="%s"
                       AND test_data="%s"
                       AND category="%s"
                    ''' % (system, training_data, test_data, category))
                results = cu.fetchall()
                if results:
                    experiment_id = results[0][0]
                else:
                    cu.execute('''
                        INSERT INTO experiment_info (system, training_data, test_data, category)
                        VALUES ("%s", "%s", "%s", "%s")
                        ''' % (system, training_data, test_data, category))
                    experiment_id = cu.lastrowid

                # get column info
                columns = next(f)[:-1].replace('correctness_', 'accurate_').split('\t')[2:]
                print("INFO: {}".format(columns))
                # insert rows
                for line in f:
                    line_split = line[:-1].split('\t')
                    db = line_split[0].lower()
                    if db.endswith('_rename'):
                        db = db.split('_')[0]
                    query_index = line_split[1]
                    accurates = line_split[2:]

                    # get query_id
                    cu.execute('''
                        SELECT query_id
                        FROM   query_split
                        WHERE  db="%s"
                           AND query_index=%s
                           AND category="%s"
                        ''' % (db, query_index, category))
                    results = cu.fetchall()
                    if results:
                        query_id = results[0][0]
                    else:
                        raise Exception(
                            'gold query (' + db + ', ' + query_index + ', ' + category + ') not found')
                    # insert a row into accuracy
                    cu.execute('''
                        SELECT experiment_id
                        FROM   accuracy
                        WHERE  experiment_id=%d AND query_id=%d
                        ''' % (experiment_id, query_id))
                    if cu.rowcount == 0:
                        column_names_str = ''.join(', ' + col for col in columns)
                        accurate_values_str = ''.join(', ' + acc for acc in accurates)
                        cu.execute('''
                            INSERT INTO accuracy (experiment_id, query_id''' + column_names_str + ''')
                            VALUES (''' + str(experiment_id) + ', ' + str(query_id) + accurate_values_str + ')')
                        num_inserted += 1
                    else:
                        sets = ',\n'.join(col + '=' + acc for col, acc in zip(columns, accurates))
                        print("INFO: {}, {}, {}".format(sets, experiment_id, query_id))
                        cu.execute('''
                            UPDATE accuracy
                            SET    ''' + sets + '''
                            WHERE  experiment_id=''' + str(experiment_id) + ''' AND query_id=''' + str(query_id))
                        num_updated += 1

        dbconn.commit()
    except Exception as e:
        print_log_message('ERROR', ['insert_accuracy', 'loading failed', str(e)])
        try:
            dbconn.rollback()
            cu.close()
        except:
            pass
    print_log_message('INFO', ['insert_accuracy', 'done',
                               str(num_inserted) + ' query/queries inserted',
                               str(num_updated) + ' query/queries updated',
                               ])


def compute_accuracy_qm_cosette(dbconn, experiment_id, target_query_ids=None, restore_prev=False, is_IRNet=False):
    num_computed = 0

    # select non-computed results
    compute_results = True
    print_log_message(
        'INFO', ['compute_accuracy_qm_cosette', 'start selecting (query_id, cosette_execution_result_id) pairs'])
    try:
        with closing(dbconn.cursor()) as cu:
            cu.execute('''USE benchmark''')

            # select (experiment_id, query_id) pairs to compute the accuracy
            if target_query_ids:
                select_results = []
                for target_query_id in target_query_ids:
                    cu.execute('''
                        SELECT qr.query_id,
                               a.cosette_execution_result_id
                        FROM   query_result qr
                        JOIN   accuracy a
                        ON     qr.query_id=a.query_id AND qr.experiment_id=a.experiment_id
                        JOIN   query q ON qr.query_id=q.query_id
                        JOIN   experiment_info E ON E.experiment_id=qr.experiment_id
                        JOIN   sql_execution_result SER_gold ON SER_gold.sql_execution_result_id=q.sql_execution_result_id
                        JOIN   sql_execution_result SER_gen ON SER_gen.sql_execution_result_id=qr.sql_execution_result_id
                        WHERE  qr.experiment_id=%d and (a.accurate_ex=1 OR a.accurate_ex_tmp_data=1 OR SER_gold.errno = -1 OR SER_gen.errno = -1) AND SER_gold.errno <= 0 AND SER_gen.errno <=0 AND qr.query_id=%d
                        ''' % (experiment_id, target_query_id))
                    select_results.extend(list(cu.fetchall()))
            else:
                cu.execute('''
                    SELECT qr.query_id,
                           a.cosette_execution_result_id
                    FROM   query_result qr
                    JOIN   accuracy a ON qr.query_id=a.query_id AND qr.experiment_id=a.experiment_id
                    JOIN   query q ON qr.query_id=q.query_id
                    JOIN   experiment_info E ON E.experiment_id=qr.experiment_id
                    JOIN   sql_execution_result SER_gold ON SER_gold.sql_execution_result_id=q.sql_execution_result_id
                    JOIN   sql_execution_result SER_gen ON SER_gen.sql_execution_result_id=qr.sql_execution_result_id
                    WHERE  qr.experiment_id=%d and (a.accurate_ex=1 OR a.accurate_ex_tmp_data=1 OR SER_gold.errno = -1 OR SER_gen.errno = -1) AND SER_gold.errno <= 0 AND SER_gen.errno <=0
                    ''' % (experiment_id))
                select_results = cu.fetchall()
    except MySQLdb.Error as e:
        print_log_message(
            'ERROR', ['compute_accuracy_qm_cosette', 'selecting (query_id, cosette_execution_result_id) pairs failed', str(e)])
        compute_results = False
        select_results = []
    print_log_message('INFO', ['compute_accuracy_qm_cosette', 'selecting (query_id, cosette_execution_result_id) pairs done', str(
        len(select_results)) + ' query/queries found', str(select_results)])

    # compute the results
    if compute_results:
        print_log_message('INFO', ['compute_accuracy_qm_cosette', 'start executing Cosette'])
        try:
            for select_result in select_results:
                query_id = select_result[0]
                prev_cosette_execution_result_id = select_result[1]
                if prev_cosette_execution_result_id and not restore_prev:
                    continue
                dbconn.begin()

                with closing(dbconn.cursor()) as cu:
                    cu.execute('''USE benchmark''')

                    # select db and queries
                    cu.execute('''
                        SELECT query.db,
                               query_result.db_suffix,
                               query.gold_sql_cosette,
                               query.gold_sql_cosette_rename,
                               query_result.gen_sql_cosette,
                               query_result.gen_sql_execute,
                               query.gold_sql_execute
                        FROM   query, query_result
                        WHERE  query.query_id=query_result.query_id
                           AND query_result.experiment_id=%d
                           AND query_result.query_id=%d
                        ''' % (experiment_id, query_id))
                    if cu.rowcount == 0:
                        raise Exception('query not found: (%d, %d)' % (experiment_id, query_id))
                    elif cu.rowcount > 1:
                        raise Exception('unexcpeted rows: (%d, %d)' % (experiment_id, query_id))
                    select_result = cu.fetchone()
                    db = select_result[0]
                    db_suffix = select_result[1]
                    gold_sql_cosette = select_result[2]
                    gold_sql_cosette_rename = select_result[3]
                    gen_sql_cosette = select_result[4]
                    gen_sql_execute = select_result[5]
                    gold_sql_execute = select_result[6]
                    if is_IRNet == True:
                        print_log_message('INFO', ['compute_accuracy_qm_cosette', 'IRNet cannot generate DISTINCT',
                                               str((experiment_id, query_id)),])
                        p = re.compile(" *distinct *", re.I)
                        gold_sql_cosette = p.sub(r' ', gold_sql_cosette)
                        gold_sql_cosette_rename = p.sub(r' ', gold_sql_cosette_rename)
                        gen_sql_cosette = p.sub(r' ', gen_sql_cosette)
                    
                    print_log_message('INFO', ['compute_accuracy_qm_cosette', 'SQLs to execute',
                                               str((experiment_id, query_id)),
                                               gold_sql_cosette_rename if db_suffix else gold_sql_cosette,
                                               gen_sql_cosette
                                               ])

                    # execute SQLs through dummy DB
                    if db_suffix and db != "wikisql":
                       db = db + '_' + db_suffix

                    if db == "wikisql":
                       dummydb = db
                    else:
                       dummydb = db + '_dummy'
                    if db.lower().startswith('wikitablequestionsnull'):
                       dummydb = 'wikitablequestions_rename_dummy'

                    try:
                        cu.execute('''USE ''' + dummydb.replace('-', ''))
                        if db_suffix:
                           cu.execute(gold_sql_cosette_rename)
                        else:
                           cu.execute(gold_sql_execute)
                        errno_gold = 0
                    except MySQLdb.Error as e:
                        errno_gold = e.args[0]
                        #raise Exception('gold query error: (%s, %s)' % (gold_sql_execute, e.args[1]))
                    try:
                        cu.execute('''USE ''' + dummydb.replace('-', ''))
                        cu.execute(gen_sql_execute)
                        errno_gen = 0
                    except MySQLdb.Error as e:
                        errno_gen = e.args[0]
                    cu.execute('''USE benchmark''')
                    if errno_gold != 0:
                        cos_output = 'Invalid gold SQL'
                        cos_result = 'Invalid gold SQL'
                    elif errno_gen != 0:
                        cos_output = 'Invalid generated SQL'
                        cos_result = 'Invalid generated SQL'
                    valid_sqls = errno_gold == 0 and errno_gen == 0

                    print_log_message(
                        'INFO', ['compute_accuracy_qm_cosette', 'executing SQLs over dummy DBs done',
                                 str((experiment_id, query_id)),
                                 str(errno_gold),
                                 str(errno_gen)
                                 ])

                    # run Cosette
                    if valid_sqls:
                        d = {
                            # [2018.8.14] Cosette cannot handle double quotation mark 
                            # [2018.8.14] Cosette also cannot handle quotation mark in value
                            'gold_sql': gold_sql_cosette_rename.strip().replace('"', "'") if db_suffix or db == "wikisql"  else gold_sql_cosette.decode('ascii', 'ignore').replace('"', "'"),
                            'gen_sql': gen_sql_cosette.strip().decode('ascii', 'ignore').replace('"', "'")}
                        print_log_message('INFO', ['compute_accuracy_qm_cosette', 'executing Cosette start',
                                                   str((experiment_id, query_id)),
                                                   str(d['gold_sql']), str(d['gen_sql'])
                                                   ])
                        # TODO: delete me when the issue solved
                        if db == 'mas_rename':
                            print 'mas_rename not found: (%d, %d)' % (experiment_id, query_id)
                            continue

                        try:
                            with open('./cosette_template/' + db + '.template.cos') as f:
                               template_string = f.read()
                            templates=template_string.split('\n')
                            templates_need=[]
                            for tem in templates:
                               if tem.startswith('schema '):
                                   schema_name=tem.split('schema schema_')[1].split('(')[0]
                                   if schema_name in d["gold_sql"].lower() or schema_name in d["gen_sql"].lower():
                                      templates_need.append(tem)
                               elif tem.startswith('table '):
                                   table_name=tem.split('table ')[1].split('(')[0]
                                   if table_name in d["gold_sql"].lower() or table_name in d["gen_sql"].lower():
                                      templates_need.append(tem)
                               else:
                                   templates_need.append(tem)
                            template_string='\n'.join(templates_need)
                        except IOError:
                            raise Exception(db + '.template.cos doesn\'t exist')
                        #except Exception:
                        #    raise Exception(db + '\'s running cosette fail: Another exception')
                              
                        cos_source = Template(template_string).substitute(d)
                        print(cos_source)
                        try:
                            #cos_output = solver.solve(cos_source, cos_folder="./Cosette")
                            cos_output = requests.post("https://demo.cosette.cs.washington.edu/solve", data={"api_key":COSETTE_API_KEY, "query":cos_source}, verify=False).text
                            cos_result = json.loads(cos_output)['result']
                        except Exception:
                            cos_output = 'Cosette Server Error'
                            cos_result = 'Cosette Server Error'
                            print_log_message('INFO', ['compute_accuracy_qm_cosette', 'executing Cosette Server Error',
                                                       str((experiment_id, query_id)),
                                                       cos_output
                                                       ])
                        else:
                            print_log_message('INFO', ['compute_accuracy_qm_cosette', 'executing Cosette done',
                                                       str((experiment_id, query_id)),
                                                       cos_output
                                                       ])

                    # delete previous cosette execution result
                    if prev_cosette_execution_result_id:
                        cu.execute('''
                           DELETE FROM cosette_execution_result
                           WHERE  cosette_execution_result_id=%d
                           ''' % (prev_cosette_execution_result_id))

                    # insert (cos_output, cos_result) to cosette_execution_result, and get
                    # cosette_execution_result_id
                    cu.execute('''
                        INSERT INTO cosette_execution_result (result, output)
                        VALUES ("%s", "%s")
                        ''' % (cos_result, cos_output.replace('\\', '\\\\').replace('"', '\\"')))
                    cosette_execution_result_id = cu.lastrowid

                    # update the corresponding (accurate_qm_cosette, cosette_execution_result_id) in
                    # accuracy
                    if cos_result == 'EQ':
                        accurate_qm_cosette = '1'
                    elif cos_result == 'NEQ':
                        accurate_qm_cosette = '0'
                    else:
                        accurate_qm_cosette = 'NULL'

                    print_log_message('INFO', ['compute_accuracy_qm_cosette', 'accurate_qm_cosette',
                                               str((experiment_id, query_id)),
                                               accurate_qm_cosette
                                               ])
                    cu.execute('''
                        UPDATE accuracy
                        SET accurate_qm_cosette=%s,
                            cosette_execution_result_id=%d
                        WHERE experiment_id=%d
                          AND query_id=%d
                        ''' % (accurate_qm_cosette, cosette_execution_result_id, experiment_id, query_id))
                    if cu.rowcount == 0:
                        cu.execute('''
                            INSERT INTO accuracy (experiment_id, query_id, accurate_qm_cosette, cosette_execution_result_id)
                            VALUES (%d, %d, %s, %d)
                            ''' % (experiment_id, query_id, accurate_qm_cosette, cosette_execution_result_id))
                dbconn.commit()
                print_log_message('INFO', ['compute_accuracy_qm_cosette', 'inserting accurate_qm_cosette done',
                                           str((experiment_id, query_id))
                                           ])
                num_computed += 1
            print_log_message('INFO', ['compute_accuracy_qm_cosette', 'executing Cosette done'])
        except Exception as e:
            print_log_message('ERROR', ['compute_accuracy_qm_cosette', 'executing Cosette failed', str(e)])
            try:
                dbconn.rollback()
            except:
                pass
    print_log_message('INFO', ['compute_accuracy_qm_cosette', 'done',
                               str(num_computed) + ' query/queries computed'
                               ])


def insert_from_local_file(dbconn, command, dbname):
   with closing(dbconn.cursor()) as cu:
      print_log_message('INFO', ['compute_accuracy_ex', command])
      while 1:
         cu.execute(command)
         cu.execute("show warnings")
         warnings=cu.fetchall()

         # handle foreign key exception
         foreign_keys={}
         need_to_solve_foreign_key=False
         for msg in warnings:
            print_log_message('INFO', ['compute_accuracy_ex', 'WARNING {}'.format(msg[2])])
            if msg[2].find('Cannot add or update') is not -1 and msg[2].find('REFERENCES') is not -1:
               foreign_tablename=msg[2][msg[2].find('REFERENCES')+11:msg[2].find('(')-1]
               foreign_schema=msg[2][msg[2].find('REFERENCES')+11:]
               foreign_tablename=foreign_schema[1:foreign_schema.find('(')].strip()[:-1]
               foreign_columnname=foreign_schema[foreign_schema.find('(')+2:foreign_schema.find(')')-1].strip()
               my_columnname=msg[2][msg[2].find('FOREIGN KEY')+14:].split(')')[0][:-1]

               if foreign_tablename in foreign_keys or foreign_columnname.replace(' ', '').find("`,") is not -1:
                  print_log_message('INFO', ['compute_accuracy_ex', 'WARNING {}, {} - Multiple foreign keys from one table currently not supported'.format(foreign_tablename, foreign_columnname)])
                  need_to_solve_foreign_key=False
                  break
               keyMap=[my_columnname,foreign_columnname]
               foreign_keys[foreign_tablename]=keyMap
               need_to_solve_foreign_key=True
         if not need_to_solve_foreign_key:
            break
         else:
            for table, [my_column, foreign_column] in foreign_keys.items():
               # GET VALUES from tsv file
               filename=command[:command.find("tsv")+3]
               filename=filename[filename.find("INFILE")+8:]
               my_column_list=command[command.find("\\n")+5:].strip()[:-2].lower().split(',')
               my_index=my_column_list.index(my_column.lower())
               with open(filename) as tsvfile:
                  tsvreader = csv.reader(tsvfile, delimiter="\t")
                  valueSet=set() 
                  for line in tsvreader:
                     try:
                        valueSet.add(int(line[my_index]))
                     except:
                        raise Exception("String key currently not supported")
               for value in valueSet:
                  cu.execute('''USE '''+dbname)
                  cu.execute('''
                      INSERT INTO %s (%s)
                      VALUES (%d) ''' % (table, foreign_column, value))
                  print('''
                      INSERT INTO %s (%s)
                      VALUES (%d) ''' % (table, foreign_column, value))
         
## TODO: REWRITE ##
def compute_accuracy_lf(dbconn, experiment_id, restore_prev=False, target_query_ids=None):
    num_computed = 0

    # select non-computed results
    compute_results = True
    print_log_message('INFO', ['compute_accuracy_lf', 'start selecting query_id\'s'])
    try:
        with closing(dbconn.cursor()) as cu:
            cu.execute('''USE benchmark''')

            # select (experiment_id, query_id) pairs to compute the accuracy
            if target_query_ids:
                select_results = [(query_id,) for query_id in target_query_ids]
            else:
                cu.execute('''
                    SELECT qr.query_id
                    FROM   query_result qr
                    WHERE  qr.experiment_id=%d
                    ''' % (experiment_id))
                select_results = cu.fetchall()
    except MySQLdb.Error as e:
        print_log_message('ERROR', ['compute_accuracy_lf', 'selecting query_id\'s failed', str(e)])
        compute_results = False
        select_results = []
    print_log_message(
        'INFO', ['compute_accuracy_lf', 'selecting query_id\'s done', str(len(select_results)) + ' query/queries found'])

    # compute the results
    if compute_results:
        print_log_message('INFO', ['compute_accuracy_lf', 'start comparing SQLs'])
        print 'Computing results'
        try:
            for select_result in select_results:
                query_id = select_result[0]
                dbconn.begin()

                with closing(dbconn.cursor()) as cu:
                    cu.execute('''USE benchmark''')

                    # select db and queries
                    cu.execute('''
                        SELECT query.db,
                               query_result.db_suffix,
                               query.gold_sql_cosette,
                               query.gold_sql_cosette_rename,
                               query_result.gen_sql_cosette,
                               accuracy.accurate_lf
                        FROM   query, query_result, accuracy
                        WHERE  query.query_id=query_result.query_id
                           AND accuracy.query_id=query.query_id
                           AND accuracy.experiment_id=query_result.experiment_id
                           AND query_result.experiment_id=%d
                           AND query_result.query_id=%d
                        ''' % (experiment_id, query_id))
                    if cu.rowcount == 0:
                        raise Exception('query not found: (%d, %d)' % (experiment_id, query_id))
                    elif cu.rowcount > 1:
                        raise Exception('unexpected rows: (%d, %d)' % (experiment_id, query_id))
                    select_result = cu.fetchone()
                    db = select_result[0]
                    db_suffix = select_result[1]
                    gold_sql_cosette = select_result[2]
                    gold_sql_cosette_rename = select_result[3]
                    gen_sql_cosette = select_result[4]
                    accurate_lf = select_result[5]
                    if accurate_lf is not None and not restore_prev:
                       dbconn.commit()
                       continue
                    if db_suffix or db == "wikisql":
                       gold_sql_cosette=gold_sql_cosette_rename
                    # Rewrite
                    print_log_message('INFO', ['compute_accuracy_lf', 'SQLs to execute',
                                               str((experiment_id, query_id)),
                                               gold_sql_cosette.encode('utf8'),
                                               gen_sql_cosette.encode('utf8')
                                               ])
                    gold_sql_string=gold_sql_cosette.strip().lower().replace(' ','')
                    gen_sql_string=gen_sql_cosette.strip().lower().replace(' ','')
                    # compute accurate_lf
                    if gold_sql_string == gen_sql_string:
                        accurate_lf = '1'
                    else:
                        accurate_lf = '0' 

                    print_log_message('INFO', ['compute_accuracy_lf', 'accurate_lf',
                                               str((experiment_id, query_id)),
                                               accurate_lf
                                               ])

                    # update accuracy
                    cu.execute('''USE benchmark''')
                    cu.execute('''
                        SELECT * FROM accuracy
                        WHERE  experiment_id=%d
                           AND query_id=%d
                        ''' % (experiment_id, query_id))
                    
                    if cu.rowcount == 0:
                        cu.execute('''
                            INSERT INTO accuracy (experiment_id, query_id, accurate_lf)
                            VALUES (%d, %d, %s)
                            ''' % (experiment_id, query_id, accurate_lf))
                    else:
                        cu.execute('''
                        UPDATE accuracy
                        SET    accurate_lf=%s
                        WHERE  experiment_id=%d
                           AND query_id=%d
                        ''' % (accurate_lf, experiment_id, query_id))

                    print_log_message(
                        'INFO', ['compute_accuracy_lf', 'updating accuracy_lf done',
                                 str((experiment_id, query_id))
                                 ])

                    dbconn.commit()
                    num_computed += 1
            print_log_message('INFO', ['compute_accuracy_lf', 'comparing SQLs done'])
        except Exception as e:
            print_log_message('ERROR', ['compute_accuracy_lf', 'comparing SQLs failed', str(e)])
            try:
                dbconn.rollback()
            except:
                pass
    print_log_message('INFO', ['compute_accuracy_lf', 'done',
                               str(num_computed) + ' query/queries computed'
                               ])



def compute_accuracy_ex(dbconn, experiment_id, target_query_ids=None, restore_prev=False, is_IRNet=False):
    num_computed = 0

    # select non-computed results
    compute_results = True
    print_log_message('INFO', ['compute_accuracy_ex', 'start selecting query_id\'s'])
    try:
        with closing(dbconn.cursor()) as cu:
            cu.execute('''USE benchmark''')

            # select (experiment_id, query_id) pairs to compute the accuracy
            if target_query_ids:
                select_results = [(query_id,) for query_id in target_query_ids]
            else:
                cu.execute('''
                    SELECT qr.query_id
                    FROM   query_result qr
                    WHERE  qr.experiment_id=%d
                    ''' % (experiment_id))
                select_results = cu.fetchall()
    except MySQLdb.Error as e:
        print_log_message('ERROR', ['compute_accuracy_ex', 'selecting query_id\'s failed', str(e)])
        compute_results = False
        select_results = []
    print_log_message(
        'INFO', ['compute_accuracy_ex', 'selecting query_id\'s done', str(len(select_results)) + ' query/queries found'])

    # compute the results
    if compute_results:
        print_log_message('INFO', ['compute_accuracy_ex', 'start executing SQLs'])
        print 'Computing results'
        try:
            for select_result in select_results:
                query_id = select_result[0]
                if USE_TEST_DATA_GEN:
                    current_insert_dir=TMP_DATA_PATH+"Q"+str(query_id)
                    if not os.path.isdir(current_insert_dir):
                        print_log_message('INFO', ['compute_accuracy_ex', 'Directory {} DOES NOT EXIST'.format(query_id)])
                        continue
                    else:
                        if os.path.isdir('./tmp'):
                            shutil.rmtree('./tmp')
                        shutil.copytree(current_insert_dir, './tmp')
                        
                        current_insert_file="./tmp/INSERTQUERY.sql"
                        if not os.path.exists(current_insert_file):
                            print_log_message('INFO', ['compute_accuracy_ex', 'File {} DOES NOT EXIST'.format(current_insert_file)])
                            continue

                        # load tmp db
                        load_mysql_dumps_to_tmpdb(dbconn)

                        # insert rows
                        for line in open(current_insert_file): 
                           insert_from_local_file(dbconn, line.strip(), TMP_DATABASE_NAME)
                            
                dbconn.begin()

                with closing(dbconn.cursor()) as cu:
                    cu.execute('''USE benchmark''')

                    # select db and queries
                    cu.execute('''
                        SELECT query.db,
                               query_result.db_suffix,
                               query.gold_sql_execute,
                               query_result.gen_sql_execute,
                               query.sql_execution_result_id,
                               query_result.sql_execution_result_id,
                               query.sql_execution_result_id_tmp_data,
                               query_result.sql_execution_result_id_tmp_data
                        FROM   query, query_result
                        WHERE  query.query_id=query_result.query_id
                           AND query_result.experiment_id=%d
                           AND query_result.query_id=%d
                        ''' % (experiment_id, query_id))
                    if cu.rowcount == 0:
                        raise Exception('query not found: (%d, %d)' % (experiment_id, query_id))
                    elif cu.rowcount > 1:
                        raise Exception('unexcpeted rows: (%d, %d)' % (experiment_id, query_id))
                    select_result = cu.fetchone()
                    db = select_result[0]
                    db_suffix = select_result[1]
                    gold_sql_execute = select_result[2]
                    gen_sql_execute = select_result[3]
                    gold_sql_execution_result_id = select_result[4]
                    gen_sql_execution_result_id = select_result[5]
                    gold_sql_execution_result_id_tmp_data = select_result[6]
                    gen_sql_execution_result_id_tmp_data = select_result[7]

                    # time limit
                    if db.startswith('spider'):
                        gold_sql_execute = gold_sql_execute.encode('utf8')
                        gen_sql_execute = gen_sql_execute.encode('utf8')
                    else:
                        gold_sql_execute = re.sub(
                            '(SELECT |select )', 'SELECT /*+ MAX_EXECUTION_TIME(' + str(TIMEOUT) + ') */ ', gold_sql_execute, count=1)
                        gen_sql_execute = re.sub(
                            '(SELECT |select )', 'SELECT /*+ MAX_EXECUTION_TIME(' + str(TIMEOUT) + ') */ ', gen_sql_execute, count=1)

                    if is_IRNet == True:
                        p = re.compile(" *distinct *", re.I)
                        gold_sql_execute = p.sub(r' ', gold_sql_execute)
                        gen_sql_execute = p.sub(r' ', gen_sql_execute)
                    print_log_message('INFO', ['compute_accuracy_ex', 'SQLs to execute',
                                               str((experiment_id, query_id)),
                                               gold_sql_execute.encode('utf8'),
                                               gen_sql_execute.encode('utf8')
                                               ])

                    gold_output_too_large = False
                    gold_errno = 0
                    if gold_sql_execution_result_id and not USE_TEST_DATA_GEN:
                        # retrieve the gold query execution result
                        cu.execute('''USE benchmark''')
                        cu.execute('''
                            SELECT output,
                                   errno
                            FROM   sql_execution_result
                            WHERE  sql_execution_result_id=%d
                            ''' % (gold_sql_execution_result_id,))
                        gold_output_json, gold_errno = cu.fetchone()
                        gold_output_too_large = gold_output_json == 'Too large'
                        if not gold_output_too_large:
                            try:
                                gold_output = json.loads(
                                    gold_output_json, use_decimal=True, encoding='utf-8')
                            except:
                                gold_errno=1000
                    if (not gold_sql_execution_result_id) or USE_TEST_DATA_GEN or gold_output_too_large or gold_errno != 0 or gold_output == [] or gold_output == [[0]] or restore_prev:
                        # execute the gold query
                        gold_output_too_large=False
                        if db.startswith('spider'):
                            gold_output = get_sqlite3_execute_result(gold_sql_execute)
                            gold_rowcount = len(gold_output)
                            if gold_rowcount > 500:
                                gold_output_too_large=True
                            gold_output_json = json.dumps(
                                gold_output, use_decimal=True, encoding='utf-8', default=str)
                            gold_errno = 0
                        else:
                            if db.lower().startswith('wikitablequestionsnull'):
                                db = 'wikitablequestionsnull_rename'
                            try:
                                cu.execute('''USE ''' + get_db_name(db))
                                cu.execute("SET NAMES utf8")
                                cu.execute("SET collation_connection = 'utf8_general_ci'")
                                cu.execute(gold_sql_execute)
                                gold_output = cu.fetchall()
                                gold_rowcount = cu.rowcount
                                if gold_rowcount > 500:
                                   gold_output_too_large=True
                                gold_output_json = json.dumps(
                                    gold_output, use_decimal=True, encoding='utf-8', default=str)
                                gold_errno = 0
                            except MySQLdb.Error as e:
                                gold_rowcount = 0
                                gold_errno = e.args[0]
                                gold_output = e.args[1]
                                gold_output_json = json.dumps(
                                    gold_output, use_decimal=True, encoding='utf-8', default=str)
                            if gold_errno == 1065:
                                print_log_message('INFO', ['compute_accraucy_ex', 'executing gold SQL wrong', str(gold_sql_execute)])
                    print_log_message('INFO', ['compute_accuracy_ex', 'executing gold SQL done against {}'.format(get_db_name(db)),
                                               str((experiment_id, query_id)),
                                               str(gold_errno)
                                               ])

                    # renamed DB
                    if db_suffix:
                        db_rename = db + '_' + db_suffix
                    else:
                        db_rename = db
                    if db.lower().startswith('wikitablequestionsnull'):
                        db_rename = 'wikitablequestionsnull_rename'

                    gen_errno = 0
                    gen_output_too_large = False
                    if gen_sql_execution_result_id and not USE_TEST_DATA_GEN:
                        # retrieve the gen query execution result
                        cu.execute('''USE benchmark''')
                        cu.execute('''
                            SELECT output,
                                   errno
                            FROM   sql_execution_result
                            WHERE  sql_execution_result_id=%d
                            ''' % (gen_sql_execution_result_id,))
                        gen_output_json, gen_errno = cu.fetchone()
                        gen_output_too_large = gen_output_json == 'Too large'
                        if not gen_output_too_large:
                            try:
                                gen_output = json.loads(
                                gen_output_json, use_decimal=True, encoding='utf-8')
                            except:
                                gen_errno=1000
                    if (not gen_sql_execution_result_id) or USE_TEST_DATA_GEN or gen_output_too_large or gen_errno != 0 or gen_output==[] or gen_output==[[0]] or restore_prev:
                        # execute generated query
                        gen_output_too_large=False
                        if db.startswith('spider'):
                            try:
                                gen_output = get_sqlite3_execute_result(gen_sql_execute)
                                gen_rowcount = len(gen_output)
                                if gen_rowcount > 500:
                                    gen_output_too_large=True
                                gen_output_json = json.dumps(
                                    gen_output, use_decimal=True, encoding='utf-8', default=str)
                                gen_errno = 0
                            #except sqlite3.ProgrammingError as e:
                            except Exception as e:
                                #print(type(e))
                                #print(e)
                                gen_rowcount = 0
                                gen_errno = 9999
                                gen_output = e.args[0]
                                gen_output_json = json.dumps(
                                    gen_output, use_decimal=True, encoding='utf-8', default=str)
                        else:
                            try:
                                cu.execute('''USE ''' + get_db_name(db_rename))
                                cu.execute("SET NAMES utf8")
                                cu.execute("SET collation_connection = 'utf8_general_ci'")
                                cu.execute(gen_sql_execute)
                                gen_rowcount = cu.rowcount
                                if gen_rowcount > 500:
                                   gen_output_too_large=True
                                gen_output = cu.fetchall()
                                gen_output_json = json.dumps(
                                    gen_output, use_decimal=True, encoding='utf-8', default=str)
                                gen_errno = 0
                            except MySQLdb.Error as e:
                                gen_rowcount = 0
                                gen_errno = e.args[0]
                                gen_output = e.args[1]
                                gen_output_json = json.dumps(
                                    gen_output, use_decimal=True, encoding='utf-8', default=str)

                    print_log_message(
                        'INFO', ['compute_accuracy_ex', 'executing generated SQL done against {}'.format(get_db_name(db_rename)),
                                 str((experiment_id, query_id)),
                                 str((experiment_id, query_id)),
                                 str(gen_errno)
                                 ])

                    # check if the run failed
                    sql_error = gold_errno != 0 or gen_errno != 0

                    # check 'ORDER BY'
                    ordered_result = any('order by' in sql.lower()
                                         for sql in [gold_sql_execute, gen_sql_execute])

                    # format output
                    def lower_str(row):
                        return [v.lower() if isinstance(v, unicode) or isinstance(v, str) else v for v in row]
                    def pack_column(table, col_index): 
                        return zip(*table)[col_index]
                    gold_output = tuple(tuple(lower_str(row)) for row in gold_output)
                    gen_output = tuple(tuple(lower_str(row)) for row in gen_output)

                    # compute accurate_ex
                    if gold_errno != 0 and gen_errno <= 0:
                        accurate_ex = 'NULL'  # wrong gold SQL
                    elif gen_errno > 0:
                        accurate_ex = '0'  # wrong generated SQL
                    elif len(gold_output) == 0 and len(gen_output) == 0:
                        accurate_ex = '1'  # empty results
                    elif len(gold_output) == 0 or len(gen_output) == 0:
                        accurate_ex = '0'  # wrong execution results
                    elif ordered_result:
                        print_log_message(
                            'INFO', ['compute_accuracy_ex', 'SQLs contain \'order by\'',
                                     str((experiment_id, query_id))
                                     ])
                        gold_cols=tuple(tuple(pack_column(gold_output, i) for i in range(len(gold_output[0]))))
                        gen_cols=tuple(tuple(pack_column(gen_output, i) for i in range(len(gen_output[0]))))
                        print('{}, {}'.format(Counter(gold_cols), Counter(gen_cols)))
                        accurate_ex = '1' if Counter(gold_cols) == Counter(gen_cols) else '0'
                         
                       # accurate_ex = '1' if gold_output == gen_output else '0'  # compare results as sequence
                    else:
                        # print('{}\n\n{}'.format(gold_output, gen_output)) 
                        gold_cols=tuple(pack_column(gold_output, i) for i in range(len(gold_output[0])))
                        gen_cols=tuple(pack_column(gen_output, i) for i in range(len(gen_output[0])))
                        if len(gold_cols) != len(gen_cols) or len(gold_output) != len(gen_output):
                            accurate_ex = '0'
                        elif len(gold_cols) == 0:
                            accurate_ex = '1'
                        else:
                            row_maps=[[] for i in range(len(gold_cols))] 
                            for i in range(len(gen_cols)):
                                for j in range(len(gold_cols)):
                                    if Counter(gen_cols[i]) == Counter(gold_cols[j]):
                                        row_maps[j].append(i)
                            permus=[ permu for permu in permutations([i for i in range(len(gold_cols))], len(gold_cols)) ]
                            mappings=list(mapping for mapping in product(*row_maps) if mapping in permus)
                            accurate_ex = '0'
                            for mapping in mappings:
                                gen_cols_permus=tuple(gen_cols[i] for i in mapping)
                                gen_output_permus=tuple(pack_column(gen_cols_permus, i) for i in range(len(gen_cols_permus[0])))
                                if Counter(gen_output_permus) == Counter(gold_output):
                                    accurate_ex='1'
                                    break    

                     #   accurate_ex = '1' if Counter(gold_output) == Counter(
                     #       gen_output) else '0'  # compare results as bag

                    print_log_message('INFO', ['compute_accuracy_ex', 'accurate_ex',
                                               str((experiment_id, query_id)),
                                               accurate_ex
                                               ])
                    accurate_ex_col_name="accurate_ex"
                    sql_execution_result_id_col_name="sql_execution_result_id"
                    if USE_TEST_DATA_GEN:
                       accurate_ex_col_name="accurate_ex_tmp_data"
                       sql_execution_result_id_col_name="sql_execution_result_id_tmp_data"

                    # update accuracy
                    cu.execute('''USE benchmark''')
                    cu.execute('''
                        SELECT * FROM accuracy
                        WHERE  experiment_id=%d
                           AND query_id=%d
                        ''' % (experiment_id, query_id))
                    
                    if cu.rowcount == 0:
                        cu.execute('''
                            INSERT INTO accuracy (experiment_id, query_id, %s)
                            VALUES (%d, %d, %s)
                            ''' % (accurate_ex_col_name, experiment_id, query_id, accurate_ex))
                    else:
                        cu.execute('''
                        UPDATE accuracy
                        SET    %s=%s
                        WHERE  experiment_id=%d
                           AND query_id=%d
                        ''' % (accurate_ex_col_name, accurate_ex, experiment_id, query_id))

                    print_log_message(
                        'INFO', ['compute_accuracy_ex', 'updating accuracy_ex done',
                                 str((experiment_id, query_id))
                                 ])
                    # delete previous gold sql execution result
                    if gold_sql_execution_result_id and not USE_TEST_DATA_GEN:
                        cu.execute('''
                           DELETE FROM sql_execution_result
                           WHERE  sql_execution_result_id=%d
                           ''' % (gold_sql_execution_result_id,))
                    if gold_sql_execution_result_id_tmp_data and USE_TEST_DATA_GEN:
                        cu.execute('''
                           DELETE FROM sql_execution_result
                           WHERE  sql_execution_result_id=%d
                           ''' % (gold_sql_execution_result_id_tmp_data,))

                    # insert (gold_output, gold_errno) to sql_execution_result, and get
                    # sql_execution_result_id
                    if gold_output_too_large:
                        gold_output_json = "Too large"
                    cu.execute('''
                        INSERT INTO sql_execution_result (output, errno)
                        VALUES ("%s", %d)
                        ''' % (gold_output_json.replace('\\', '\\\\').replace('"', '\\"'), gold_errno))
                    gold_sql_execution_result_id = cu.lastrowid

                    # update the corresponding executed_result_id in query_result
                    cu.execute('''
                        UPDATE query
                        SET %s=%d
                        WHERE query_id=%d
                        ''' % (sql_execution_result_id_col_name, gold_sql_execution_result_id, query_id))


                    # delete previous generated sql execution result
                    if gen_sql_execution_result_id and not USE_TEST_DATA_GEN:
                        cu.execute('''
                           DELETE FROM sql_execution_result
                           WHERE  sql_execution_result_id=%d
                           ''' % (gen_sql_execution_result_id,))

                    if gen_sql_execution_result_id_tmp_data and USE_TEST_DATA_GEN:
                        cu.execute('''
                           DELETE FROM sql_execution_result
                           WHERE  sql_execution_result_id=%d
                           ''' % (gen_sql_execution_result_id_tmp_data,))

                    # insert (gen_output, gen_errno) to sql_execution_result, and get
                    # sql_execution_result_id
                    if gen_output_too_large:
                        gen_output_json = "Too large"
                    cu.execute('''
                        INSERT INTO sql_execution_result (output, errno)
                        VALUES ("%s", %d)
                        ''' % (gen_output_json.replace('\\', '\\\\').replace('"', '\\"'), gen_errno))
                    gen_sql_execution_result_id = cu.lastrowid

                    # update the corresponding executed_result_id in query_result
                    cu.execute('''
                        UPDATE query_result
                        SET %s=%d
                        WHERE experiment_id=%d AND query_id=%d
                        ''' % (sql_execution_result_id_col_name, gen_sql_execution_result_id, experiment_id, query_id))

                    dbconn.commit()
                    print_log_message('INFO', ['compute_accuracy_ex', 'inserting SQL results done',
                                               str((experiment_id, query_id))
                                               ])
                    num_computed += 1
            print_log_message('INFO', ['compute_accuracy_ex', 'executing SQLs done'])
        except Exception as e:
            print_log_message('ERROR', ['compute_accuracy_ex', 'executing SQLs failed', str(e)])
            try:
                dbconn.rollback()
            except:
                pass
    print_log_message('INFO', ['compute_accuracy_ex', 'done',
                               str(num_computed) + ' query/queries computed'
                               ])


def generate_cosette_template(dbconn):
    print MESSAGE_HLINE + MESSAGE_BOLD + 'Generating Cosette templates' + MESSAGE_RESET
    num_created = 0
    try:
        cu = dbconn.cursor()
        for db in DATABASES:
            filename = db + '.template.cos'
            print 'Generating ' + filename,
            with open(TEMPLATE_DIR + filename, 'w') as f:
                schemas = ''
                tables = ''

                cu.execute('USE ' + db.replace('-', ''))
                cu.execute('SHOW TABLES')
                result_tables = cu.fetchall()
                for result_table in result_tables:
                    table = result_table[0]
                    cu.execute('DESC ' + table)
                    field_results = cu.fetchall()

                    fields = []
                    for field_result in field_results:
                        field_name = field_result[0].lower()
                        field_type = field_result[1].lower()
                        if 'int' in field_type:
                            field_type = 'int'
                        elif 'varchar' in field_type:
                            field_type = 'text'
                        elif 'text' in field_type:
                            field_type = 'text'
                        # TODO: verify
                        elif 'decimal' in field_type:
                            field_type = 'int'
                        elif 'double' in field_type:
                            field_type = 'int'
                        elif 'float' in field_type:
                            field_type = 'int'
                        else:
                            field_type = 'int'

                        fields.append((field_name, field_type))

                    schema = 'schema schema_' + table + '('
                    for (field_name, field_type) in fields:
                        schema += field_name + ':' + field_type + ', '
                    schema = schema[:-2] + ');\n'
                    schemas += schema
                    tables += 'table ' + table + '(schema_' + table + ');\n'

                f.write(schemas + '\n')
                f.write(tables + '\n')
                f.write('query q1 \n')
                f.write('`${gold_sql}`;\n\n')
                f.write('query q2 \n')
                f.write('`${gen_sql}`;\n\n')
                f.write('verify q1 q2;')
            print MESSAGE_DONE
            num_created += 1
        cu.close()
    except Exception, e:
        print 'Creating benchmark database ' + MESSAGE_FAILED
        print MESSAGE_RED + str(e) + MESSAGE_RESET
        return
    print MESSAGE_BOLD + 'Creating benchmark database' + MESSAGE_RESET,
    print '(' + str(num_created) + ' template(s) created)'


# ------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_file', help='NEED for load_generated_queries/insert_accuracy', default='')
    parser.add_argument('--training_data', help='NEED for compute_accuracy_XY', default='')
    parser.add_argument('--test_data', help='NEED for compute_accuracy_XY', default='')
    parser.add_argument('--category', help='NEED for compute_accuracy_XY', default='')
    parser.add_argument('--system', help='NEED for compute_accuracy_XY', default='')
    parser.add_argument('--compute_qm_cosette', help='NEED for compute_accuracy_XY', default='False')
    parser.add_argument('--compute_ex', help='NEED for compute_accuracy_XY', default='False')
    parser.add_argument('functions', help='name of functions to execute')
    parser.add_argument('--tmp_data_path', help='NEED for generated data', default=None)
    args = parser.parse_args()
    if args.tmp_data_path is not None:
        USE_TEST_DATA_GEN=True
        TMP_DATA_PATH=args.tmp_data_path
        if not os.path.isdir(TMP_DATA_PATH):
            print("TMP_DATA_PATH: {} WRONG".format(TMP_DATA_PATH))
            exit(1)
        if not  os.path.exists(TMP_DATA_PATH+"/schema.mysqldump"):
            print("TMP_DATA_PATH: {} WRONG - schema.mysqldump does not exist".format(TMP_DATA_PATH))
            exit(1)
    with closing(MySQLdb.connect(host=HOST, user=USER, passwd=PASSWD, charset='utf8', local_infile=1)) as dbconn:
        if args.functions == "load_mysql_dumps":
            load_mysql_dumps(dbconn)
        elif args.functions == "generate_cosette_template":
            generate_cosette_template(dbconn)
        elif args.functions == "create_benchmark_db":
            create_benchmark_db(dbconn)
        elif args.functions == "load_gold_queries":
            load_gold_queries(dbconn)
        elif args.functions == "load_generated_queries":
            if args.input_file == '':
                print("python main.py load_generated_queries --input_file [INPUT_FILE_NAME]")
                exit(1)
            experiment_id, system, query_ids = load_generated_queries(dbconn, args.input_file)
            if args.compute_ex == 'True':
                compute_accuracy_ex(dbconn, experiment_id, target_query_ids=query_ids, restore_prev=True, is_IRNet=(system.lower() == 'irnet'))
            if args.compute_qm_cosette == 'True':
                compute_accuracy_qm_cosette(dbconn, experiment_id, target_query_ids=query_ids, restore_prev=True, is_IRNet=(system.lower() == 'irnet'))
        elif args.functions == "load_logs":
            load_logs(dbconn)
        elif args.functions == "insert_accuracy":
            if args.input_file == '':
                print("python main.py insert_accuracy --input_file [INPUT_FILE_NAME]")
                exit(1)
            insert_accuracy(dbconn, args.input_file)
        elif args.functions == "compute_accuracy_ex" or args.functions == "compute_accuracy_qm_cosette" or args.functions == "compute_accuracy_lf":
            if args.training_data == '' or args.test_data == '' or args.system == '' or args.category == '':
                print(
                    "python main.py compute_accuracy_qm_cosette(or compute_accuracy_ex) --category [CATEGORY] --training_data [TRAINING_DATA] --test_data [TEST_DATA] --system [SYSTEM_NAME] {--tmp_data_path [Path to generated data]}")
                exit(1)
            category = args.category
            training_data = args.training_data
            test_data = args.test_data
            system = args.system
            experiment_id = None
            try:
               dbconn.begin()
               with closing(dbconn.cursor()) as cu:
                  cu.execute('''USE benchmark''')
                  cu.execute('''
                    SELECT experiment_id
                    FROM   experiment_info
                    WHERE  system="%s"
                       AND training_data="%s"
                       AND test_data="%s"
                       AND category="%s"
                    ''' % (system, training_data, test_data, category))
                  if cu.rowcount == 0:
                    print("NO EXPERIMENT {}, {}, {}, {}".format(system, training_data, test_data, category))
                    exit(1)
                  else:
                    experiment_id = cu.fetchone()[0]

               dbconn.commit()
            except:
               try:
                  dbconn.rollback()
               except:
                  pass
               print("ERROR {}, {}, {}, {}".format(system, training_data, test_data, category))
               exit(1)
            if experiment_id is None:
               print("ERROR {}, {}, {}, {}".format(system, training_data, test_data, category))
               exit(1)
            if args.functions == "compute_accuracy_ex":
               compute_accuracy_ex(dbconn, experiment_id, restore_prev=True, is_IRNet=(system.lower() == 'irnet'))
            elif args.functions == "compute_accuracy_qm_cosette":
               compute_accuracy_qm_cosette(dbconn, experiment_id, restore_prev=True, target_query_ids=None, is_IRNet=(system.lower() == 'irnet'))
            elif args.functions == "compute_accuracy_lf":
               compute_accuracy_lf(dbconn, experiment_id, restore_prev=True, target_query_ids=None)

        pass
