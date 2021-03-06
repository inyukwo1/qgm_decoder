import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from commons.utils import run_lstm, seq_conditional_weighted_num, SIZE_CHECK
from models.syntaxsql.net_utils import to_batch_seq


class KeyWordPredictor(nn.Module):
    '''Predict if the next token is (SQL key words):
        WHERE, GROUP BY, ORDER BY. excluding SELECT (it is a must)'''
    def __init__(self, H_PARAM, embed_layer, bert=None):
        super(KeyWordPredictor, self).__init__()
        self.N_word = H_PARAM['N_WORD']
        self.N_depth = H_PARAM['N_depth']
        self.N_h = H_PARAM['N_h']
        self.gpu = H_PARAM['gpu']
        self.use_hs = H_PARAM['use_hs']
        self.table_type = H_PARAM['table_type']

        self.acc_num = 3
        self.embed_layer = embed_layer

        self.use_bert = True if bert else False
        if bert:
            self.q_bert = bert
            encoded_num = 768
        else:
            self.q_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
            encoded_num = self.N_h

        self.hs_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.kw_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_num_att = nn.Linear(encoded_num, self.N_h)
        self.hs_num_att = nn.Linear(self.N_h, self.N_h)
        self.kw_num_out_q = nn.Linear(encoded_num, self.N_h)
        self.kw_num_out_hs = nn.Linear(self.N_h, self.N_h)
        self.kw_num_out = nn.Sequential(nn.Tanh(), nn.Linear(self.N_h, 4)) # num of key words: 0-3

        self.q_att = nn.Linear(encoded_num, self.N_h)
        self.hs_att = nn.Linear(self.N_h, self.N_h)
        self.kw_out_q = nn.Linear(encoded_num, self.N_h)
        self.kw_out_hs = nn.Linear(self.N_h, self.N_h)
        self.kw_out_kw = nn.Linear(self.N_h, self.N_h)
        self.kw_out = nn.Sequential(nn.Tanh(), nn.Linear(self.N_h, 1))

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if self.gpu:
            self.cuda()

    def forward(self, input_data):
        q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len = input_data

        B = len(q_len)
        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        kw_enc, _ = run_lstm(self.kw_lstm, kw_emb_var, kw_len)

        # Predict key words number: 0-3
        q_weighted_num = seq_conditional_weighted_num(self.q_num_att, q_enc, q_len, kw_enc).sum(1)
        # Same as the above, compute SQL history embedding weighted by key words attentions
        hs_weighted_num = seq_conditional_weighted_num(self.hs_num_att, hs_enc, hs_len, kw_enc).sum(1)
        # Compute prediction scores
        kw_num_score = self.kw_num_out(self.kw_num_out_q(q_weighted_num) + int(self.use_hs)* self.kw_num_out_hs(hs_weighted_num))
        SIZE_CHECK(kw_num_score, [B, 4])

        # Predict key words: WHERE, GROUP BY, ORDER BY.
        q_weighted = seq_conditional_weighted_num(self.q_att, q_enc, q_len, kw_enc)
        SIZE_CHECK(q_weighted, [B, 3, self.N_h])

        # Same as the above, compute SQL history embedding weighted by key words attentions
        hs_weighted = seq_conditional_weighted_num(self.hs_att, hs_enc, hs_len, kw_enc)
        # Compute prediction scores
        kw_score = self.kw_out(self.kw_out_q(q_weighted) + int(self.use_hs)* self.kw_out_hs(hs_weighted) + self.kw_out_kw(kw_enc)).view(B,3)

        score = (kw_num_score, kw_score)

        return score

    def loss(self, score, truth):
        loss = 0
        B = len(truth)
        kw_num_score, kw_score = score
        #loss for the key word number
        truth_num = [len(t) for t in truth] # double check to exclude select
        data = torch.from_numpy(np.array(truth_num))
        if self.gpu:
            data = data.cuda()
        truth_num_var = Variable(data)
        loss += self.CE(kw_num_score, truth_num_var)
        #loss for the key words
        T = len(kw_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth[b]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            data = data.cuda()
        truth_var = Variable(data)
        #loss += self.mlsml(kw_score, truth_var)
        #loss += self.bce_logit(kw_score, truth_var) # double check no sigmoid for kw
        pred_prob = self.sigm(kw_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss

    def evaluate(self, score, gt_data):
        return self.check_acc(score, gt_data)

    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        if self.gpu:
            kw_num_score, kw_score = [x.data.cpu().numpy() for x in score]
        else:
            kw_num_score, kw_score = [x.data.numpy() for x in score]

        for b in range(B):
            cur_pred = {}
            kw_num = np.argmax(kw_num_score[b])
            cur_pred['kw_num'] = kw_num
            cur_pred['kw'] = np.argsort(-kw_score[b])[:kw_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            kw_num, kw = p['kw_num'], p['kw']
            flag = True
            if kw_num != len(t): # double check to excluding select
                num_err += 1
                flag = False
            if flag and set(kw) != set(t):
                err += 1
                flag = False
            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))

    def preprocess(self, batch):
        q_seq, history, label = to_batch_seq(batch)
        q_emb_var, q_len = self.embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)

        kw_emb_var = self.embed_layer.gen_word_list_embedding(["where", "group by", "order by"], len(batch))
        mkw_len = np.full(q_len.shape, 3, dtype=np.int64)

        input_data = q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, mkw_len
        gt_data = label

        return input_data, gt_data

    def save_weights(self, save_dir):
        print('Saving model...')
        torch.save(self.state_dict(), os.path.join(save_dir, "keyword_models.dump"))