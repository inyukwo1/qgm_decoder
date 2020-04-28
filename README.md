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

* ra-transformer encoder with qgm decoder:
```bash
python train.py encoder=ra_transformer decoder=transformer query_type=all batch_size=8 optimize_freq=16 tag=${something}
```
* lstm encoder with qgm decoder:
```bash
python train.py encoder=lstm decoder=transformer query_type=all batch_size=8 optimize_freq=16 tag=${something}
```
* 

#### Data preparation

* Download [Glove Embedding](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) and put `glove.42B.300d` under `./data/` directory  
* run `create_semql_qgm_for_all_dataset.sh` under `./preprocess/` directiory

#### Data path hierarchy

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


#### Training

Run `train.py` to train the model.

##### Arguments

Now support hydra to pass arguments.
config design is as below

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


* You can select config file or change any argument with the command line `python train.py cuda=1 model=qgm dataset=wikisql`
* Or create your own config file under *config/train/* for convenience.

#### Testing

Run `eval.py` to evaluate the model

##### Arguments

* `--train_config:` Path to jsonnet file that contains training settings  
* `--cuda:` GPU number  
* `--load_model` Path to trained model  

#### tensorboard

* After running train.py, run `tensorboard --logdir ./logs --bind_all --port [port_num]` in terminal
* check http://[ip]:6006

