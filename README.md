# Text-to-SQL Development Environment

Wrapper for tranining and testing Text-to-SQL model, built on Pytorch.

## Description  
Build your own Text-to-SQL Model.  
Training and Testing is done in the following sequence.  

1. DataLoader loads data (preprocessing data if necessary)  
2. Text-to-SQL model receives data from  DataLoader and preprocess in the batch unit  
3. Text-to-SQL model predicts SQL  
4. Text-to-SQL model evaluates the result  

## Requirments
#### DataLoader Methods:
1. \_\_init__(H_PARAMS: Dict)
2. load_data(mode: Text, load_option: Dict)
3. shuffle() : None
4. get_train() : List
5. get_train_len() : Int
6. get_eval() : List
7. get_eval_len() : Int

#### Text-to-SQL Model Methods:
1. preprocess(batch: ) : tuple(input_data, gt_data)
2. forward(input_data: List) : List
3. loss(score: List, gt_data: List) : List
4. evaluate(score: List, gt_data: List, batch: List) : List
5. gen_sql(score:List, gt_data: List) : List
5. acc_num <- variable 

#### Config File for training/testing
1. model: str (model module path)
2. dataloader: str (dataloader module path)
3. batch_size: int
4. lr: int
5. weight_decay: int
6. epoch: int
7. eval_freq: int

### Training and Testing
``` python train.py --train_config={train_config_path}```  
``` python test.py --test_config={test_config_path}```


### Implemented
1.Models

- SyntaxSQL  
- TypeSQL  
- SQLNet  
- From Predictor

2.Dataset  

- Spider 

### To-Do
- Add more datasets
- Add more models
- Generalize train and test script
