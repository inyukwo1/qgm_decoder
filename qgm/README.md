download spider dataset
combine train_spider.json and train_others.json as train_original.json


Rename as follows: 
 - train_real_origianl.json
 - dev_real_origianl.json
 - tables_origianl.json

# Fix erros in tables.json
run fix_data.py to fix erros in tables_original.json and create tables.json

# Append 'sql' info in the dictionary
run parse_data.py to fix dev_real_origianl.json and create dev_original.json
run parse_data.py to fix train_real_original.json and create train_original.json

# Append 'qgm' info in the dictionary
run sql2qgm.py to append qgm to dev_original.json and create dev.json
run sql2qgm.py to append qgm to train_origianl.json and create train.json



