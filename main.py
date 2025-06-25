from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import numpy as np
import torch
from torch import nn
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F

print(torch.__version__)
import enum
import matplotlib.pyplot as plt
from d2l import torch as d2l
from dtaidistance import dtw
# import pyttsx3
import lmmdv
from torch.autograd import Function

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.logits = nn.Linear(128, 5)

    def forward(self, input):
        logits = self.logits(input)
        return logits


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(128 * 1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input):
        out = self.layer(input)
        return out


label_path = "BS+DDAST\\Train-1"
def val_self_training(model, valid_dl, device, src_id, trg_id, round_idx):
    from sklearn.metrics import accuracy_score

    home_path = r'E:\jtlv\Python\DDAST\ResultsMASS'
    if round_idx == 0:
        model[0].eval()
        model[1][0].eval()
    else:
        PATH_Acc = f'E:\jtlv\Python\DDAST\ResultsMASS/model_param/{fold_feature}_{round_idx-1}feature_extractor.pth'
        save_model = torch.load(PATH_Acc, map_location=device)
        model[0].load_state_dict(save_model)
        model[0].eval()

        PATH_Acc = f'E:\jtlv\Python\DDAST\ResultsMASS/model_param/{fold_feature}_{round_idx-1}classifier_1.pth'
        save_model = torch.load(PATH_Acc, map_location=device)
        model[1][0].load_state_dict(save_model)
        model[1][0].eval()

    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []
    all_dtw = []

    with torch.no_grad():
        for data, labels, dtw in valid_dl:
            data = data.float().to(device)
            dtw = dtw.float().to(device)
            labels = torch.argmax(labels, axis=1)
            labels = labels.view((-1)).to(device)

            # forward pass
            out = model[0](data, dtw,0)
            # features = model[1](out)
            features = out
            predictions = model[1][0](features)


            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())
            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)
            all_dtw.append(dtw)

    print("agreement of labels: ", accuracy_score(all_labels, all_pseudo_labels))
    all_data = torch.cat(all_data, dim=0)
    all_dtw = torch.cat(all_dtw, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["dtw"] = all_dtw
    data_save["labels"] = torch.tensor(torch.from_numpy(all_pseudo_labels))
    file_name = f"pseudo_train_{src_id}_to_{trg_id}_round_{fold_feature}_{round_idx}.pt"
    os.makedirs(os.path.join(home_path, label_path), exist_ok=True)
    torch.save(data_save, os.path.join(home_path, label_path, file_name))
    return accuracy_score(all_labels, all_pseudo_labels)


def model_evaluate(model, valid_dl, device):
    if type(model) == tuple:
        #         if(round_idx==0):
        model[0].eval()
        # model[1].eval()
        model[1][0].eval()

    else:
        model.eval()
    total_loss = []
    total_acc = []
    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, dtw in valid_dl:
            data = data.float().to(device)
            dtw = dtw.float().to(device)
            labels = torch.argmax(labels, axis=1)
            labels = labels.view((-1)).to(device)

            # forward pass
            out = model[0](data, dtw,0)

            # features = model[1](out)
            features = out
            predictions = model[1][0](features)


            # compute loss
            loss = criterion(predictions, labels)
            total_loss.append(loss.item())
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs

class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.w1 = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float32), requires_grad=True)
        
        self.sigmoid = nn.Sigmoid()
        

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, domain_label):

        if domain_label == 0:
            
            original_mean_t = self.bns[0].running_mean.clone().detach()
            original_mean_s = self.bns[1].running_mean.clone().detach()
            self.bns[0].running_mean.data.mul_(self.sigmoid(self.w1))
            self.bns[0].running_mean.data.add_((1-self.sigmoid(self.w1)) * self.bns[1].running_mean.data)
            #计算var部分
            t_mean_diff_square = torch.square(original_mean_t-self.bns[0].running_mean.data)
            s_mean_diff_square = torch.square(original_mean_s-self.bns[0].running_mean.data)

            final_add = self.sigmoid(self.w1) * t_mean_diff_square + (1-self.sigmoid(self.w1)) * (self.bns[1].running_var.data + s_mean_diff_square)

            self.bns[0].running_var.data.mul_(self.sigmoid(self.w1))
            self.bns[0].running_var.data.add_(final_add)
        bn = self.bns[domain_label]


        return bn(x)



class _DomainSpecificBatchNorm1d(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm1d, self).__init__()
        
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats) for _ in
             range(num_classes)])
        self.sigmoid = nn.Sigmoid()
        self.w1 = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float32), requires_grad=True)
        

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, domain_label):

        if domain_label == 0:
            original_mean_t = self.bns[0].running_mean.clone().detach()
            original_mean_s = self.bns[1].running_mean.clone().detach()
            self.bns[0].running_mean.data.mul_(self.sigmoid(self.w1))
            self.bns[0].running_mean.data.add_((1 - self.sigmoid(self.w1)) * self.bns[1].running_mean.data)
            # 计算var部分
            t_mean_diff_square = torch.square(original_mean_t - self.bns[0].running_mean.data)
            s_mean_diff_square = torch.square(original_mean_s - self.bns[0].running_mean.data)

            final_add = self.sigmoid(self.w1) * t_mean_diff_square + (1 - self.sigmoid(self.w1)) * (
                        self.bns[1].running_var.data + s_mean_diff_square)

            self.bns[0].running_var.data.mul_(self.sigmoid(self.w1))
            self.bns[0].running_var.data.add_(final_add)
        bn = self.bns[domain_label]

        return bn(x)

import torch.nn as nn
import torch
from torch.nn import functional as F


class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = _DomainSpecificBatchNorm(self.channels,2, affine=True)

    def forward(self, x,domain_label):
        residual = x

        x = self.bn2(x,domain_label)
        weight_bn = self.bn2.bns[domain_label].weight.data.abs() / torch.sum(self.bn2.bns[domain_label].weight.data.abs())
        #         print(weight_bn.shape)
        #         print(x.shape)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class Spaital_Att(nn.Module):
    def __init__(self, H, W, t=16):
        super(Spaital_Att, self).__init__()
        self.H = H
        self.W = W
        self.channels = H * W

        self.bn2 = _DomainSpecificBatchNorm1d(self.channels,2, affine=True)

    def forward(self, x,domain_label):
        residual = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(x.shape[0], -1, x.shape[3])
        x = self.bn2(x,domain_label)
        weight_bn = self.bn2.bns[domain_label].weight.data.abs() / torch.sum(self.bn2.bns[domain_label].weight.data.abs())
        #         print(weight_bn.shape)
        #         print(x.shape)
        weight_bn = weight_bn.unsqueeze(1).repeat(x.shape[0], 1, x.shape[2])
        #         print(weight_bn.shape)

        x = torch.mul(weight_bn, x)
        x = x.reshape(x.shape[0], self.H, self.W, x.shape[2])
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class Att(nn.Module):
    def __init__(self, channels, shape, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1


# In[ ]:

#设备
dev = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# In[ ]:


class HeadAttention(nn.Module):
    def __init__(self, input_shape, ratio=2):
        super(HeadAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(input_shape, input_shape // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(input_shape // ratio, input_shape, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class AdditiveMatric(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveMatric, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.mean(self.attention_weights, dim=0)
        



# In[ ]:


# @save
# @save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            
        else:
            valid_lens = valid_lens.reshape(-1)
            
            
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,  # 作用：去除其中一个维度
                              value=-1e6)
        
        return nn.functional.softmax(X.reshape(shape),
                                     dim=-1)  # 参考注意力分数矩阵的形状是n*m 相当于一行对应一个query所以是对每行softmax 一些列mask掉舍去没用的填充


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        return torch.bmm(self.dropout(self.attention_weights), values)



class DTWAttention(nn.Module):
    """加性注意力"""

    def __init__(self, num_vec, head, dropout, **kwargs):
        super(DTWAttention, self).__init__(**kwargs)
        self.head = head

        self.w_v = nn.Linear(num_vec, num_vec * head, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, valid_lens):
        norm_y = torch.max(y, axis=-1)[0].unsqueeze(-1) - y

        A_Timenew = self.w_v(norm_y).reshape(y.shape[0], num_vec, self.head, num_vec)
        A_Timenew = A_Timenew.permute(0, 2, 1, 3)
        A_Timenew = A_Timenew.reshape(-1, num_vec, num_vec)
        self.attention_weights = masked_softmax(A_Timenew, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), x)


# In[ ]:

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, timestep, in_dim, dropout=0.3, out_dim=12, residual_channels=32, blocks=2,
                 layers=2):
        # self.gwnet = gwnet(dev, nums_vec, gw_time, in_dim) 25 5 32
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.attention = AdditiveMatric(in_dim, in_dim, in_dim, 0.5)

        self.end_conv_2 = nn.Conv2d(in_channels=residual_channels, out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        
        for b in range(blocks):
            for i in range(layers):
                self.bn.append(_DomainSpecificBatchNorm(residual_channels,2))
                self.gconv.append(gcn(residual_channels, residual_channels, dropout, support_len=1))

        self.fc1 = nn.Linear(timestep * out_dim * num_nodes, 64)

    def forward(self, input,domain_label):
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        index = in_len // 2
        query = input[:, :, :, index].permute(0, 2, 1)
        adp = self.attention(query, query, query, None)
        
        new_supports = [] + [adp]
        x = self.start_conv(input)

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x,domain_label)
            

        block_value = self.end_conv_2(x)
        
        block_value = torch.flatten(block_value, 1)
        
        block_value = self.fc1(block_value)

        return block_value


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, nums_vec, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        #         self.attention = AdditiveAttention(int(num_hiddens/num_heads), int(num_hiddens/num_heads),num_hiddens,dropout)
        self.attention = DTWAttention(nums_vec, num_heads, dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.head_attention = HeadAttention(num_heads)

    def forward(self, queries, keys, values, y, valid_lens):
        
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        values = values.reshape(-1, values.shape[2], values.shape[3])

        if valid_lens is not None:
            
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
            
        output = self.attention(values, y, valid_lens)

        
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)




# @save
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X


# @save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


# @save
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


# @save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """transformer编码器块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    """transformer编码器"""

    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        
        X = self.pos_encoding(X * math.sqrt(self.num_hiddens))
        
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X




def LayerNorm(x):
    '''
    Apply relu and layer normalization
    '''
    x_residual, time_conv_output = x
    relu_x = torch.nn.functional.elu(x_residual + time_conv_output)
    temp = nn.LayerNorm(relu_x.shape[3])
    temp = temp.to(dev)
    ln = temp(relu_x)
    return ln


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c,_ = x.size()
        # y = x
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# In[ ]:


class GraphSleepBlock(nn.Module):
    def __init__(self, input_shape, nums_vec, num_hidden, time_conv_strides, head, **kwargs):
        super(GraphSleepBlock, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.filter = num_hidden * head
        
        self.gaan = MultiHeadAttention(nums_vec, input_shape[3], input_shape[3], input_shape[3],
                                       self.filter, head, 0.5)


        self.namChannel = Channel_Att(input_shape[1])
        self.namSpaial = Spaital_Att(input_shape[2], num_hidden * head)
        
        self.lstm = TransformerEncoder(self.filter * input_shape[2], self.filter * input_shape[2],
                                       self.filter * input_shape[2], self.filter * input_shape[2],
                                       [input_shape[1], self.filter * input_shape[2]], self.filter * input_shape[2],
                                       4 * self.filter * input_shape[2], head, 2, 0.1)

        self.conv2 = nn.Conv2d(self.input_shape[3], self.filter, kernel_size=(1, 1), stride=(1, time_conv_strides))

    def forward(self, x, y,domain_label):  # x_before

        for timestep in range(x.shape[1]):
            if timestep == 0:
                spatial_gcn = self.gaan(x[:, timestep, :, :], x[:, timestep, :, :], x[:, timestep, :, :],
                                        y[:, timestep, :, :], None).unsqueeze(1)
            else:
                spatial_gcn = torch.cat((spatial_gcn,
                                         self.gaan(x[:, timestep, :, :], x[:, timestep, :, :], x[:, timestep, :, :],
                                                   y[:, timestep, :, :], None).unsqueeze(1)), axis=1)
                
        spatial_gcn = self.namSpaial(spatial_gcn,domain_label)

        # 使用lstm的部分:
        spatial_gcn = spatial_gcn.reshape(x.shape[0], x.shape[1], -1)
        spatial_lstm = self.lstm(spatial_gcn, None)
        time_conv_output = spatial_lstm.reshape(x.shape[0], x.shape[1], x.shape[2],
                                                self.filter)  
        
        time_conv_output = self.namChannel(time_conv_output,domain_label)

        x = x.permute(0, 3, 1, 2)
        x_residual = self.conv2(x)
        x_residual = x_residual.permute(0, 2, 3, 1)
        end_output = LayerNorm([x_residual, time_conv_output])
        return end_output


# In[ ]:

class GraphSleepNet(nn.Module):
    
    def __init__(self, input_shape, nums_vec, gw_time, in_dim, num_hidden, time_conv_strides, head, **kwargs):
        
        super(GraphSleepNet, self).__init__(**kwargs)
        self.block_out = GraphSleepBlock(input_shape, nums_vec, num_hidden, time_conv_strides, head)
        self.gwnet = gwnet(dev, nums_vec, gw_time, in_dim)
        
        self.fc1 = nn.Linear(num_hidden * head * input_shape[1] * input_shape[2], 64)
        
        self.fc2 = nn.Linear(128, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x, y,domain_label):
        block_out = self.block_out(x, y,domain_label)
        gw_value = self.gwnet(x,domain_label)
        
        block_out = torch.flatten(block_out, 1)
        block_out = self.fc1(block_out)
        block_out = torch.cat((block_out, gw_value), axis=1)
        

        return block_out



# In[ ]:


def AddContext(x, context, label=False, dtype=float):
    ret = []
    assert context % 2 == 1, "context value error."

    cut = int(context / 2)
    if label:
        for p in range(10):
            tData = x[p][cut:x[p].shape[0] - cut]
            ret.append(tData)
            # print(tData.shape)
    else:
        for p in range(10):
            tData = np.zeros([x[p].shape[0] - 2 * cut, context, x[p].shape[1], x[p].shape[2]], dtype=dtype)
            for i in range(cut, x[p].shape[0] - cut):
                tData[i - cut] = x[p][i - cut:i + cut + 1]
            # print(tData.shape)
            ret.append(tData)
    return ret



class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1  # the fold number
    x_list = []  # x list with length=k
    y_list = []  # x list with length=k

    # Initializate
    def __init__(self, k, x, y, z):
        if len(x) != k or len(y) != k:
            # print(len(x),len(y),len(z))
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = k
        self.x_list = x
        self.y_list = y
        self.z_list = z

    #     Get i-th fold
    def getFold(self, i):
        isFirst = True
        for p in range(self.k):
            if p != i:
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    train_dtw = self.z_list[p]
                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
                    train_dtw = np.concatenate((train_dtw, self.z_list[p]))
            else:
                val_data = self.x_list[p]
                val_targets = self.y_list[p]
                val_dtw = self.z_list[p]
        return train_data, train_targets, train_dtw, val_data, val_targets, val_dtw

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1, self.k):
            All_X = np.append(All_X, self.x_list[i], axis=0)
        return All_X

    # Get all label y
    def getY(self):
        All_Y = self.y_list[0]
        for i in range(1, self.k):
            All_Y = np.append(All_Y, self.y_list[i], axis=0)
        return All_Y

    # Get all label y
    def getY_one_hot(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)




from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

metric_func = lambda y_pred, y_true: accuracy_score(y_true.data.cpu().numpy(), y_pred.data.cpu().numpy())
metric_name = 'accuracy'
import scipy.io as scio
def updateBN(model, s):
    model.block_out.namChannel.bn2.bns[0].weight.grad.data.add_(
        s * torch.sign(model.block_out.namChannel.bn2.bns[0].weight.grad.data))
    
    model.block_out.namSpaial.bn2.bns[0].weight.grad.data.add_(s * torch.sign(model.block_out.namSpaial.bn2.bns[0].weight.grad.data))
    


def softmax(f):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f)  # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer


bs = 64
device = 'cuda'
torch.manual_seed(42)
torch.cuda.manual_seed(42)

for fold_feature in range(10):
    
    save_dir = r'E:\jtlv\Python\DDAST\ISRUC-S3\Feature_' + str(fold_feature) + '.npz'

    ReadList = np.load(save_dir, allow_pickle=True)
    context = 5
    Fold_Data = ReadList['Fold_Data']
    Fold_Label = ReadList['Fold_Label']
    DTW = ReadList['DTW']
    print("Read data successfully")
    Fold_Data = AddContext(Fold_Data, context)
    DTW = AddContext(DTW, context)
    Fold_Label = AddContext(Fold_Label, context, label=True)
    print('Context added successfully.')
    
    DataGenerator = kFoldGenerator(10, Fold_Data, Fold_Label, DTW)

    print('Fold #', fold_feature)
    num_vec = 12 # ISRUC-S3 12 MASS-SS3 25

    feature_extractor = GraphSleepNet((64, context, num_vec, 32), num_vec, context, 32, 10, 1, 3).float().to(device)
    classifier_1 = Classifier().float().to(device)

    feature_discriminator = Discriminator().to(device)



    # loss functions
    disc_criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    lmmd_loss = lmmdv.LMMD_loss(class_num=5).to(device)

    # optimizer.
    optimizer_encoder = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(classifier_1.parameters()), lr=1e-3, betas=(0.5, 0.99),
        weight_decay=3e-4)
    optimizer_disc = torch.optim.Adam(feature_discriminator.parameters(), lr=1e-3,
                                      betas=(0.5, 0.99), weight_decay=3e-4)
    train_data, train_targets, train_dtw, val_data, val_targets, val_dtw = DataGenerator.getFold(fold_feature)
    print(train_data.shape)
    print('train_data.shape:', train_data.shape)
    print('val_data.shape:', val_data.shape)
    print('dtw.shape:', train_dtw.shape)
    train_data = torch.tensor(train_data)  # .to(dev)
    train_targets = torch.tensor(train_targets)  # .to(dev)
    train_dtw = torch.tensor(train_dtw)  # .to(dev)
    val_data = torch.tensor(val_data)  # .to(dev)
    val_targets = torch.tensor(val_targets)  # .to(dev)
    val_dtw = torch.tensor(val_dtw)  # .to(dev)

    train_ds = TensorDataset(train_data, train_targets, train_dtw)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_ds = TensorDataset(val_data, val_targets, val_dtw)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)

    src_train_dl, src_valid_dl = train_dl,valid_dl
    trg_train_dl, trg_valid_dl = valid_dl,valid_dl
    len_dataloader = min(len(train_dl), len(valid_dl))
    running_loss = 0.0
    min_loss = 100000  # 随便设置一个比较大的数
    max_epoch = 100
    max_acc = 0
    best_label = []
    n_epoch = 30
    for round_idx in range(10):

        scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=30,
                                                            gamma=0.1)
        if round_idx == 0:
            trg_clf_wt = 0
            src_clf_wt = 1
        else:
            src_clf_wt = 1 * 0.1
            trg_clf_wt = 1e-2

        # generate pseudo labels
        train_acc = val_self_training((feature_extractor, (classifier_1,)), trg_train_dl, device,
                                      '10人',
                                      '10人', round_idx)

        if round_idx == 0:
            self_train_acc = 100 * train_acc
        else:
            self_train_acc = np.append(self_train_acc, 100 * train_acc)

            PATH_Acc = f'E:\jtlv\Python\DDAST\ResultsMASS/model_param/{fold_feature}_{round_idx-1}feature_discriminator.pth'
            save_model = torch.load(PATH_Acc, map_location=device)
            feature_discriminator.load_state_dict(save_model)


            PATH_Acc = f'E:\jtlv\Python\DDAST\ResultsMASS/model_param/{fold_feature}_{round_idx-1}feature_extractor.pth'
            save_model = torch.load(PATH_Acc, map_location=device)
            feature_extractor.load_state_dict(save_model)


            PATH_Acc = f'E:\jtlv\Python\DDAST\ResultsMASS/model_param/{fold_feature}_{round_idx-1}classifier_1.pth'
            save_model = torch.load(PATH_Acc, map_location=device)
            classifier_1.load_state_dict(save_model)


        home_path = r'E:\jtlv\Python\DDAST\ResultsMASS'
        model_type = f'{fold_feature}'
        train_mode = f'{round_idx}'
        

        file_name = f"pseudo_train_10人_to_10人_round_{fold_feature}_{round_idx}.pt"
        home_path = 'E:\jtlv\Python\DDAST\ResultsMASS'
        pseudo_trg_train_dataset1 = torch.load(os.path.join(home_path, label_path, file_name))

        # Loading datasets
        pseudo_trg_train_dataset = TensorDataset(pseudo_trg_train_dataset1["samples"],
                                                 pseudo_trg_train_dataset1["labels"], pseudo_trg_train_dataset1["dtw"])

        # Dataloader for target pseudo labels
        pseudo_trg_train_dl = torch.utils.data.DataLoader(dataset=pseudo_trg_train_dataset,
                                                          batch_size=bs,
                                                          shuffle=True, drop_last=True,
                                                          num_workers=0)

        # training..
        max_acc = 0.0
        max_epoch = 0
        for epoch in range(0, 15 + 1):
            joint_loaders = enumerate(zip(src_train_dl, pseudo_trg_train_dl))
            feature_extractor.train()
            classifier_1.train()
            # trg_att.train()
            feature_discriminator.train()
            # src_att.train()

            for step, ((src_data, src_labels, src_dtw), (trg_data, pseudo_trg_labels, trg_dtw)) in joint_loaders:
                src_data, src_labels, src_dtw, trg_data, pseudo_trg_labels, trg_dtw = src_data.to(torch.float32).to(
                    device), src_labels.to(torch.float32).to(device), src_dtw.to(torch.float32).to(device), trg_data.to(
                    torch.float32).to(device), pseudo_trg_labels.to(device), trg_dtw.to(torch.float32).to(device)



                # pass data through the source model network.
                src_feat = feature_extractor(src_data, src_dtw,1)
                
                src_pred = classifier_1(src_feat)

                # pass data through the target model network.
                trg_feat = feature_extractor(trg_data, trg_dtw,0)
                
                trg_pred = classifier_1(trg_feat)






                # Compute Source Classification Loss

                src_clf_loss = criterion(src_pred, src_labels)


                # Compute target classification loss
                trg_clf_loss = criterion(trg_pred, pseudo_trg_labels.long())
                
                # 计算dsan的loss
                loss_lmmd = lmmd_loss.get_loss(src_feat, trg_feat, torch.argmax(src_labels, axis=1),
                                               torch.nn.functional.softmax(trg_pred, dim=1))
                lambd = 2 / (1 + math.exp(-10 * (epoch) / 15)) - 1



            # total loss calucalations
                total_loss =  lambd * loss_lmmd + \
                             src_clf_wt * src_clf_loss + \
                              trg_clf_wt * trg_clf_loss
                             # 1 * 1e-2 * lambd * loss_lmmd

                #######################################################
                optimizer_encoder.zero_grad()
                total_loss.backward()
                updateBN(feature_extractor, 0.0001)
                optimizer_encoder.step()

            if round_idx in range(2):
                scheduler_encoder.step()

            # to print learning rate every epoch
            for param_group in optimizer_encoder.param_groups:
                print(round_idx, ':', param_group['lr'])

            if epoch % 1 == 0:
                target_loss, target_score, outs, trgs = model_evaluate(
                    (feature_extractor, (classifier_1,)), trg_valid_dl, device)
            if epoch == 0:
                val_acc = 100 * target_score
            else:
                val_acc = np.append(val_acc, 100 * target_score)
            model_list = [('feature_extractor', feature_extractor),
                          ('classifier_1', classifier_1),
                          ('feature_discriminator', feature_discriminator)]
            if target_score > max_acc:
                max_acc = target_score
                max_epoch = epoch
                print("save acc model")
                for net in model_list:
                    PATH_Acc = f'E:\jtlv\Python\DDAST\ResultsMASS/model_param/{fold_feature}_{round_idx}{net[0]}.pth'
                    torch.save(net[1].state_dict(), PATH_Acc)
        print('最高的测试集准确率是第{}个epoch的{}'.format(max_epoch, max_acc))
        os.makedirs(os.path.join(home_path + '\predict_label_dtw', label_path), exist_ok=True)
        val_acc_path = home_path + '\predict_label_dtw' + '/' + label_path + '/' + str(fold_feature) + '_' + str(
            round_idx) + '-val_acc.mat'
        scio.savemat(val_acc_path, {'val_acc': val_acc})

        # _plot_umap(model, src_dl, trg_dl, device, save_dir, model_type,train_mode)

    train_acc_path = home_path + '\predict_label_dtw' + '/' + label_path + '/' + str(fold_feature)+'self_train_acc.mat'
    scio.savemat(train_acc_path, {'self_train_acc': self_train_acc})



