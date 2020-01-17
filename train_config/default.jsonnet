{
"seed": 90,

# Path
"data_path": "./data",
"glove_embed_path": "./data/glove.42B.300d.txt",
"log_path": "./logs/",
"log_key": "testing",

# Hyper-parameters
"beam_size": 5,
"embed_size": 300,
"col_embed_size": 300,
"action_embed_size": 128,
"type_embed_size": 128,
"hidden_size": 300,
"att_vec_size": 300,
"dropout": 0.3,
"word_dropout": 0.2,
"loss_epoch_threshold": 50,
"sketch_loss_coefficient": 1.0,
"decode_max_time_step": 40,
"clip_grad": 5,
"lr_scheduler_gamma": 5e-1,
"milestones": [40, 70],

# Options
"lr_scheduler": true,
"sentence_features": true,
"column_pointer": true,
"query_vec_to_action_diff_map": false,
"no_query_vec_to_action_map": false,

"model_name": "rnn", # choices=['transformer', 'rnn', 'table', 'sketch']
"lstm": "lstm", # choices=['lstm', 'lstm_with_dropout', 'parent_feed']
"optimizer": "Adam",
"readout": "identity", #choices=['identity', 'non_linear']
"column_att": "affine", #choices=['dot_prod', 'affine']

# Training settings
"toy": false,
"lr": 1e-3,
"bert_lr": 1e-4,
"bert": -1, # -1 is no bert.
"batch_size": 128,
"max_epoch": 100,
"eval_freq": 1
}

