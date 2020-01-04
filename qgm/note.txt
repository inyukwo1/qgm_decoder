download spider dataset
combine train_spider.json and train_others.json as train_original.json

# Fix erros in tables.json
run fix_data.py to fix erros in tables.json

# Append 'sql' info in the dictionary
run parse_data.py to create dev.json
run parse_data.py to create train.json

# Append 'qgm' info in the dictionary
run sql2qgm.py to append qgm to dev.json
run sql2qgm.py to append qgm to train.json
