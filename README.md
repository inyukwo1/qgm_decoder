## Environment Setup

* `Ubuntu 16.04`
* `Python3.6`
* `Pytorch 1.4.0` or higher

Install Python dependency: `pip install -r requirements.txt`

## Running Code

#### Data preparation

* Download [Glove Embedding](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) and put `glove.42B.300d` under `./data/` directory  
* Follow instructions under `./qgm/` directiory

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

