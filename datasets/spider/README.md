# Spider Dataset Loader


#### Fixed bugs for the Spider Dataset
- Order difference b/w ("column names", "column names original") and ("table names", "table names original") for db_id: "scholar", "store_1", and "formula_1" in tables.json
- Non-existing "Ref_Company_Types" table being used for db_id: assets_maintenance in train.json and dev.json
- "Nested query in from clause" bug. (About 7 queries are erased)
- "syntaxsqlnet bug - parsing bug" bug. (About 50 queries are erased)


### Requirments
- Download modified Spider dataset from [Spider](https://drive.google.com/file/d/1TsekxtgIUum4xa6WRGFUGS_jpPWhvamL/view?usp=sharing) and put it under `datasets/spider/data/`