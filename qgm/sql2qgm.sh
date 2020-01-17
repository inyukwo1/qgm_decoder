echo("convert dev")
python sql2qgm.py --db ../data/tables.json --source ../data/dev_original.json --destin ../data/dev.json
echo("convert train")
python sql2qgm.py --db ../data/tables.json --source ../data/train_original.json --destin ../data/train.json
