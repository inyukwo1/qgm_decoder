# Preprocess

This directory contains the preprocess for origin Spider dataset

#### Data preparation
 
* Download [ConceptNet](https://drive.google.com/open?id=1cCOmsI8fG-euIOSGgFrGnKPoZLg2PcDN) and put it in `./conceptNet/` directory

#### Generating Train Data

`sh run_me.sh ../data/spider/train.json ../data/spider/tables.json ../data/spider/train.json`

#### Generating Dev Data

`sh run_me.sh ../data/spider/dev.json ../data/spider/tables.json ../data/spider/dev.json`

The script first performs schema linking over the dataset and schema and then generate SemQL base on the SQL query. 
Note that some From causes(self-join, subquery in From cause) and GroupBy causes(not groupby the primary key) can not be inferred from SemQL, we exclude them from our [processed dataset](https://drive.google.com/open?id=1YFV1GoLivOMlmunKW0nkzefKULO4wtrn).

