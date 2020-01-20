# Spider
echo("convert spider dev")
python sql2qgm.py --db ../data/spider/tables.json --source ../data/spider/dev_original.json --destin ../data/spider/dev.json
echo("convert spider train")
python sql2qgm.py --db ../data/spider/tables.json --source ../data/spider/train_original.json --destin ../data/spider/train.json

# Patients
echo("convert patient dev")
python sql2qgm.py --db ../data/patients/tables.json --source ../data/patients/dev_original.json --destin ../data/patients/dev.json
echo("convert patient train")
python sql2qgm.py --db ../data/patients/tables.json --source ../data/patients/train_original.json --destin ../dat/patients/train.json

# Wikisql
echo("convert wikisql dev")
python sql2qgm.py --db ../data/wikisql/tables.json --source ../data/wikisql/dev_original.json --destin ../data/wikisql/dev.json
echo("convert wikisql train")
python sql2qgm.py --db ../data/wikisql/tables.json --source ../data/wikisql/train_original.json --destin ../data/wikisql/train.json

# WikiTableQuestions
echo("convert wikitablequestions")
python sql2qgm.py --db ../data/wikitablequestions/tables.json --source ../data/wikitablequestions/dev_original.json --destin ../data/wikitablequestions/dev.json
echo("convert wikitablequestions")
python sql2qgm.py --db ../data/wikitablequestions/tables.json --source ../data/wikitablequestions/train_original.json --destin ../data/wikitablequestions/train.json

