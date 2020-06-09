## Environment Setup

* `Ubuntu 16.04`
* `Python3.6`
* `Pytorch 1.4.0`

Install Python dependency: `pip install -r requirements.txt`
Install radam from https://github.com/LiyuanLucasLiu/RAdam. I.e.,
```bash
git clone https://github.com/LiyuanLucasLiu/RAdam.git
python setup.py build
python setup.py install
```


## Running Code

#### Data preparation

* Download [Glove Embedding](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) and put `glove.42B.300d` under `./data/` directory  
* run `download_spider_dataset.py` under `./preprocess` directory to download spider dataset
* run `parse_spider_dataset.py` under `./preprocess` directory to fix errors in spider dataset
* run `create_semql_qgm_for_all_dataset.sh` under `./preprocess/` directory

#### Data path hierarchy
```
data
|
|-- spider
|   |
|   |-- tables.json
|   |-- train.json
|   |-- dev.json
|   |-- database
|       |
|       |...
|
|-- (another dataset)
|
...
```


##### Arguments

Now support hydra to pass arguments.
config design is as below
```
config
 |
 |-- dataset
 |   |
 |   |-- spider
 |   |-- patients
 |   |-- wikisql
 |   |-- wikitablequestions
 |
 |-- model
 |   | 
 |   |-- semql
 |   |-- qgm
 |   |-- qgm_transformer
 |
 |-- train    
 |   |
 |   |-- train_config1
 |   |-- train_config2
 |   |-- etc..
 |
 |--hydra
     |
     |-- output
     |-- job_logging
```

#### Training
* You can select config file or change any argument with the command line `python train.py cuda=1 model=qgm dataset=wikisql`
* Or create your own config file under *config/train/* for convenience.

##### Run `train.py` to train the model.


* ra-transformer encoder with qgm decoder:
```bash
python train.py encoder=ra_transformer decoder=transformer batch_size=8 optimize_freq=16 tag=${something}
```
* lstm encoder with qgm decoder:
```bash
python train.py encoder=lstm decoder=transformer batch_size=8 optimize_freq=16 tag=${something}
```
* bert encoder with qgm decoder:
```bash
python train.py encoder=bert decoder=transformer batch_size=2 optimize_freq=4 tag=${something}
```

#### Testing

* Eval trained model:
```bash
python eval.py encoder=ra_transformer decoder=transformer batch_size=1 load_model=logs/${date}/${something}/model/best_model.pt
```
* For down sizing schema set argument use_down_schema=True
```bash
python eval.py use_down_schema=true encoder=ra_transformer decoder=transformer batch_size=1 load_model=logs/${date}/${something}/model/best_model.pt
```

* Detailed results will be recorded at `logs/${data}/${something}/dev.result`
* For parsed grammar, check `rule/noqgm/noqgm.manifesto`

##### Arguments

* `--train_config:` Path to jsonnet file that contains training settings  
* `--cuda:` GPU number  
* `--load_model` Path to trained model  

#### tensorboard

* After running train.py, run `tensorboard --logdir ./logs --bind_all --port [port_num]` in terminal
* check http://[ip]:6006

#### tensorboard plugin

* To install tensorboard, run  `cd ./tf_board_plugin && sh ./setup.sh`.
* after running eval.py, run tensorboard for the analysis.

#### SQL Support

* tested with simple query + join query.
* other complex queries are not supported yet..
