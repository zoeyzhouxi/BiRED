# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:09:02 2020

@author: zhouxi
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_function = nn.MSELoss()

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    if weight is not None:
        loss = loss * weight
    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size):
        super(TemporalDecay, self).__init__()
        self.W = Parameter(torch.Tensor(output_size, input_size).to(device))
        self.b = Parameter(torch.Tensor(output_size).to(device))
        self.zeros = Variable(torch.zeros(output_size).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):        
        gamma = torch.abs(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()

        self.W = Parameter(torch.Tensor(input_size, input_size).to(device))
        self.b = Parameter(torch.Tensor(input_size).to(device))

        m = torch.ones(input_size, input_size).to(device) - torch.eye(input_size, input_size).to(device)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

def RNN_cell(type, feature_dim, hidden_size):
	if type == 'LSTM':
		cell = nn.LSTMCell(feature_dim, hidden_size)
	elif type == 'GRU':
		cell = nn.GRUCell(feature_dim, hidden_size)
	return cell

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, x_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = x_dim
        self.state_dim = x_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(self.state_dim, self.hidden_dim, self.num_layers)
        self.dense1 = nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim)
        self.dense2 = nn.Linear(self.hidden_dim, 1)
            
    def forward(self, x, a, h=None):  # x: seq * batch * 10, a: seq * batch * 10
        p, hidden = self.gru(x, h)   # p: seq * batch * 10
        p = torch.cat([p, a], 2)   # p: seq * batch * 20
        prob = F.sigmoid(self.dense2(F.relu(self.dense1(p))))    # prob: seq * batch * 1
        return prob

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))
    
class Encoder(nn.Module):
    def __init__(self, rnn_hid_size, feature_dim, seq_len, typ):
        super(Encoder, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.typ = typ
        
        self.feat = FeatureRegression(feature_dim)
        self.temp = FeatureRegression(seq_len)
        self.temp_decay_h = TemporalDecay(input_size=self.feature_dim, output_size=self.rnn_hid_size)
        self.rnn_cell = RNN_cell(typ, feature_dim * 2, rnn_hid_size)
        self.out = nn.Linear(rnn_hid_size, feature_dim)
       
    def forward(self, data, direct):
        values = data[direct]['values']  # [batch_size, 48, 35]
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        
        values_hat = self.feat(values) + self.temp(values.permute(0, 2, 1)).permute(0, 2, 1)
        values_hat = values * masks + values_hat * (1-masks)

        h = Variable(torch.zeros([values.size()[0], self.rnn_hid_size], device=device))
        c = Variable(torch.zeros([values.size()[0], self.rnn_hid_size], device=device))
        loss = 0.0
        for i in range(values.size()[1]-1):

            x_hat = values_hat[:, i, :]
            x = values[:, i, :]
            m = masks[:, i, :]
            d = deltas[:, i, :]
            
            gamma_h = self.temp_decay_h(d)
            h = h * gamma_h
            inputs = torch.cat([x_hat, m], dim=1)
            if self.typ == 'LSTM':
                h, c = self.rnn_cell(inputs, (h, c))
            else:
                h = self.rnn_cell(inputs, h)
                
            if i<(self.seq_len-1):
                out = self.out(h)
                loss += torch.sum(torch.abs(values[:, i+1, :] - out) * masks[:, i+1, :]) / (
                        torch.sum(masks[:, i+1, :]) + 1e-5)
            else:
                loss += 0.0
                
        if self.typ == 'LSTM':
            return h, c, loss
        else:
            return h, loss   
      
class Decoder(nn.Module):
    def __init__(self, rnn_hid_size, feature_dim, typ):
        super(Decoder, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.feature_dim = feature_dim
        self.typ = typ
        
        self.rnn_cell = RNN_cell(typ, feature_dim * 2, rnn_hid_size)
        self.temp_decay_h = TemporalDecay(input_size=self.feature_dim, output_size=self.rnn_hid_size)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.feature_dim)
        self.feat_reg = FeatureRegression(self.feature_dim)

        self.weight_combine = nn.Linear(self.feature_dim*2, self.feature_dim)
        self.merge = nn.Linear(self.feature_dim*2, self.feature_dim)
        self.out = nn.Linear(self.rnn_hid_size, 1)
       
    def forward(self, data, direct, h, c=None):
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']
        
        SEQ_LEN = values.size()[1]
        
        x_loss = 0.0
        imputations = []
        loss=[]
        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            
            gamma_h = self.temp_decay_h(d)  # [batch_size, hid_size]
            h = h * gamma_h
            
            x_h = self.hist_reg(h)
            x_loss += torch.sum((torch.abs(x - x_h)) * m) / (torch.sum(m) + 1e-5)

            x_c = m * x + (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            x_loss += torch.sum((torch.abs(x - z_h)) * m) / (torch.sum(m) + 1e-5)

            alpha = F.sigmoid(self.weight_combine(torch.cat([d, m], dim=1)))  # [batch,feature]
        
            c_h = alpha * z_h + (1 - alpha) * x_h
            l = torch.sum((torch.abs(x - c_h)) * m) / (torch.sum(m) + 1e-5)
            x_loss += l

            c_c = m * x + (1 - m) * c_h
            
            inputs = torch.cat([c_c, m], dim=1)
            if self.typ == 'LSTM':
                h, c = self.rnn_cell(inputs, (h, c))
            else:
                h = self.rnn_cell(inputs, h)
            
            imputations.append(c_c.unsqueeze(dim=1))
            loss.append(l.unsqueeze(dim=0))

        imputations = torch.cat(imputations, dim=1)
        loss = torch.cat(loss, dim=0)
        
        if self.typ == 'LSTM':
            return {'loss': x_loss, 'x_loss': loss, 'imputations': imputations, 'evals': evals, 'eval_masks': eval_masks, 'masks': masks}, h,c
        else:
            return {'loss': x_loss, 'x_loss': loss, 'imputations': imputations, 'evals': evals, 'eval_masks': eval_masks, 'masks': masks}, h


class Model(nn.Module):
    def __init__(self, rnn_hid_size, feature_dim, seq_len, output_size, impute_weight, typ):
        super(Model, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.seq_len = seq_len
        self.typ = typ
        self.encoder = Encoder(rnn_hid_size, feature_dim, seq_len, typ)
        
        # self.encoder = Encoder(rnn_hid_size, feature_dim, seq_len, typ)
        self.decoder_f = Decoder(rnn_hid_size, feature_dim, typ)
        self.decoder_b = Decoder(rnn_hid_size, feature_dim, typ)
        self.out = nn.Linear(self.rnn_hid_size, output_size)

    def forward(self, data):
        labels = data['labels']
        if self.typ == 'LSTM':
            hidden, cell, _ = self.encoder(data, 'forward')
            ret_b, hidden, cell = self.decoder_b(data, 'backward', hidden, cell)
            ret_b = self.reverse(ret_b)
            ret_f, hidden, cell = self.decoder_f(data, 'forward', hidden, cell)
        else:
            hidden, _ = self.encoder(data, 'forward')
            ret_b, hidden = self.decoder_b(data, 'backward', hidden)
            ret_b = self.reverse(ret_b)
            ret_f, hidden = self.decoder_f(data, 'forward', hidden)
            
        y_h = self.out(hidden)
        
        if self.output_size==1 and self.seq_len==48:
            y_loss = binary_cross_entropy_with_logits(y_h.squeeze(dim=1), labels)
            y_h = torch.sigmoid(y_h)
        else:
            y_loss = loss_function(y_h, labels)
        ret_f['predictions'] = y_h
        ret_f['labels'] = labels

        ret_f = self.merge_ret(ret_f, ret_b, y_loss)

        return ret_f

    def merge_ret(self, ret_f, ret_b, y_loss):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']

        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        x_loss = loss_f + loss_b + loss_c

        loss = self.impute_weight * x_loss + (1-self.impute_weight) * y_loss
        
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2
        l = (ret_f['x_loss'] + ret_b['x_loss']) / 2
        
        ret_f['loss'] = loss
        ret_f['x_loss'] = l
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
      
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret