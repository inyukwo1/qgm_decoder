# -*- coding: utf-8 -*-
import os
import json

# Correct Info
db_ids = [ 'scholar', 'store_1', 'formula_1']

db_with_error = [{'db_id': 'scholar', 'correct_table_names': ["venue", "author", "dataset", "journal", "key phrase", "paper", "cite", "paper dataset", "paper key phrase", "writes"]},
		{'db_id': 'store_1', 'correct_table_names': ['artists', 'sqlite sequence', 'albums', 'employees', 'customers', 'genres', 'invoices', 'media types', 'tracks', 'invoice lines', 'playlists', 'playlist tracks']},
		{'db_id': 'formula_1', 'correct_table_names': ['circuits', 'races', 'drivers', 'status', 'seasons', 'constructors', 'constructor standings', 'results', 'driver standings', 'constructor results', 'qualifying', 'pitstops', 'laptimes']}]

path = '../data/'
# File paths
infile_path = 'tables_original.json'
outfile_path = 'tables.json'

dbs = json.load(open(os.path.join(path, infile_path)))

for error_info in db_with_error:
    correct_table_names = error_info['correct_table_names']
    for db in dbs:
        if db['db_id'] == error_info['db_id']:
            print('Editing {}...'.format(db['db_id']))

            table_names = db['table_names']
            column_names = db['column_names']
            # Check if no typo in my answer
            assert len(table_names) == len(correct_table_names)
            assert set([str(table_name) for table_name in table_names]) == set([str(table_name) for table_name in correct_table_names])
            old = {str(table_name): idx for idx, table_name in enumerate(table_names)}
            new = {str(table_name): idx for idx, table_name in enumerate(correct_table_names)}
            ref = {new[table_name] : old[table_name] for table_name in correct_table_names}

            # Get old list
            save = [[] for _ in range(len(table_names)+1)]
            for item in column_names:
                save[item[0]+1] += [item[1]]

            # Create new list
            new_column_names = [[-1,"*"]]
            for idx in range(len(table_names)):
                for item in save[ref[idx]+1]:
                    new_column_names += [[idx, item]]

            # Check
            assert len(new_column_names) == len(column_names)
            for idx in range(len(column_names)):
                assert new_column_names[idx][0] == db['column_names_original'][idx][0], '{} : {} : {}'.format(idx, new_column_names[idx], db['column_names_original'][idx])
 
            # Change
            db['table_names'] = correct_table_names
            db['column_names'] = new_column_names

            print('Done editing!\n')

# Write
with open(os.path.join(path, outfile_path), 'w') as f:
   json.dump(dbs, f)
print('Editing tables.json Done!')


# Fix dev_gold.sql
print('\nNow fix dev_gold.sql')
# Path
in_path = 'dev_gold_original.sql'
out_path = 'dev_gold.sql'
# Info
ori = 'SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id INTERSECT SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.liked_id  =  T2.id\tnetwork_1\n'
new = 'SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id INTERSECT SELECT T2.name FROM Likes AS T3 JOIN Highschooler AS T2 ON T3.liked_id  =  T2.id\tnetwork_1\n'
# Read in
lines = open(os.path.join(path, in_path), 'r').readlines()
# Change
new_lines = [new if line == ori else line for line in lines]
# Save
with open(os.path.join(path, out_path), 'w') as f:
    for line in new_lines:
        f.write(line)
print('Editing dev_gol.sql Done!')


'''
car_1
What are the different models for the cards produced after 1980?",
"What is the average miles per gallon of all the cards with 4 cylinders?"
"What is the average horsepower for all cards produced before 1980?",
'''
