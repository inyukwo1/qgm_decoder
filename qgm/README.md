Step 1.  Download [spider dataset](https://drive.google.com/a/dblab.postech.ac.kr/uc?export=download&confirm=96wR&id=11icoH_EA-NYb0OrPTdehRWm_d7-DIzWX). And put it under ../data/spider/  
Step 2.  Naming files  
 - Combine train_spider.json and train_others.json as train_original.json  
 - Rename tables.json as tables_origianl.json  
 - Rename dev.json as dev_original.json  
 - Rename train.json as train_original.json    
  
Step 3. Fix errors
 - run fix_data.py to fix erros in tables.json  

Step 4. Parse SQL
 - run parse_data.py to create dev.json  
 - run parse_data.py to create train.json  

step 5. Parse QGM
 - run sql2qgm.py to append QGM to dev.json  
 - run sql2qgm.py to append QGM to train.json  

step 6. Parse SemQL
 - run ../preprocess/run_me.sh to semQL to dev.json  
 - run ../preprocess/run_me.sh to semQL to train.json   
 