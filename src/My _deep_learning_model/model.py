import copy
import math
import numpy as np
import random
import torch
import torch.nn as nn

add_sum=0
multi_sum=0
is_sum=0
c = 1
    
class CNN(nn.Module):
  def __init__(self,config):
    super(CNN,self).__init__()
    
    self.config = config

    history_length = config['history_lengths'][0]
    self.history_length = history_length
    conv_filters = config['conv_filters'][0]
    conv_width = config['conv_widths'][0]
    self.pooling_width = config['pooling_widths'][0]
    embedding_dims = config['embedding_dims']
    pc_hash_bits = config['pc_hash_bits']
    hash_dir_with_pc = config['hash_dir_with_pc']

    self.conv_width = conv_width
    self.config = config

    index_width = pc_hash_bits + 1

    self.embedding_table = nn.Embedding(2 ** index_width, embedding_dims)
    self.conv = nn.Conv2d(embedding_dims, conv_filters, (conv_width,1))
    self.batchnorm_CON = nn.BatchNorm2d(conv_filters)
    #self.convolution_activation = nn.ReLU(inplace=True)#x = torch.clamp(x,0,1)
    #self.pooling = nn.AvgPool2d((self.pooling_width,1), padding=0)
    #self.sumpooling_activation = nn.BatchNorm2d(conv_filters)
    if self.pooling_width > 0: 
      conv_output_size = (history_length - conv_width + 1)
      pooling_output_size = conv_output_size // self.pooling_width
    else:
      pooling_output_size = (history_length +c*( -conv_width + 1))
    self.total_output_size = pooling_output_size * conv_filters

    input_dim1 = self.total_output_size
    output_dim1 = config['hidden_neurons'][0]
    input_dim2 = output_dim1
    output_dim2 = config['hidden_neurons'][1]

    self.input_dim1 = input_dim1
    self.output_dim1 = output_dim1


    
    #self.activation_layer_fc1 = nn.ReLU(inplace=True)
    #self.activation_layer_fc2 = nn.ReLU(inplace=True)
    
    self.fc1 = nn.Linear(input_dim1, output_dim1)
    self.batchnorm_fc1 = nn.BatchNorm1d(output_dim1)
    #self.fc2 = nn.Linear(input_dim2, output_dim2)
    #self.batchnorm_fc2 = nn.BatchNorm1d(output_dim2)
    # last FC layer
    self.fc3 = nn.Linear(output_dim1, 1)
    

  def forward(self, x):
    
    x = x[:,-self.history_length:]
    x = self.embedding_table(x)
    x = torch.transpose(x, 1, 2)
    x = x.unsqueeze(-1)
    for _ in range(c):
      x = self.conv(x)
      x = self.batchnorm_CON(x)
      x = torch.clamp(x,0,1)
    global add_sum,multi_sum
    if is_sum==1:
      add_sum+=x.size(0)*x.size(1)*x.size(2)*self.conv_width
    #x = self.convolution_activation(x)
    if self.pooling_width>0:
      x = self.pooling(x) * self.pooling_width
      x = self.sumpooling_activation(x)

    x = x.view(-1, self.total_output_size)
    
    x = self.fc1(x)
    x = self.batchnorm_fc1(x) 
    x = torch.clamp(x,0,1)
    if is_sum==1:
      add_sum+=x.size(0)*self.input_dim1*self.output_dim1
    # x = self.activation_layer_fc1(x)
    
    # x = self.fc2(x)
    # x = self.batchnorm_fc2(x) 
    # x = torch.clamp(x,0,1)
    # x = self.activation_layer_fc2(x)

    x = self.fc3(x)
    # x = torch.sigmoid(x)
    if is_sum==1:
      add_sum+=x.size(0)*self.output_dim1
      multi_sum=add_sum
    x = x.squeeze(dim=1)

    return x

class CNN_no_bn(nn.Module):
  def __init__(self,config):
    super(CNN_no_bn,self).__init__()
    
    self.config = config

    history_length = config['history_lengths'][0]
    self.history_length = history_length
    conv_filters = config['conv_filters'][0]
    conv_width = config['conv_widths'][0]
    self.pooling_width = config['pooling_widths'][0]
    embedding_dims = config['embedding_dims']
    pc_hash_bits = config['pc_hash_bits']
    hash_dir_with_pc = config['hash_dir_with_pc']
    self.config = config

    index_width = pc_hash_bits + 1

    self.embedding_table = nn.Embedding(2 ** index_width, embedding_dims)
    self.conv = nn.Conv2d(embedding_dims, conv_filters, (conv_width,1))

    if self.pooling_width > 0: 
      conv_output_size = (history_length - conv_width + 1)
      pooling_output_size = conv_output_size // self.pooling_width
    else:
      pooling_output_size = (history_length +c* (-conv_width + 1))
    self.total_output_size = pooling_output_size * conv_filters

    input_dim1 = self.total_output_size
    output_dim1 = config['hidden_neurons'][0]
    input_dim2 = output_dim1
    output_dim2 = config['hidden_neurons'][1]


    self.fc1 = nn.Linear(input_dim1, output_dim1)
    #self.fc2 = nn.Linear(input_dim2, output_dim2)

    # last FC layer
    self.fc3 = nn.Linear(output_dim1, 1)
    

  def forward(self, x):
    
    x = x[:,-self.history_length:]
    x = self.embedding_table(x)
    x = torch.transpose(x, 1, 2)
    x = x.unsqueeze(-1)

    for _ in range(c):
      x = self.conv(x)
      x = torch.clamp(x,0,1)

    #x = self.convolution_activation(x)
    if self.pooling_width>0:
      x = self.pooling(x) * self.pooling_width
      x = self.sumpooling_activation(x)

    x = x.view(-1, self.total_output_size)
      
    x = self.fc1(x)

    x = torch.clamp(x,0,1)
    # x = self.activation_layer_fc1(x)
    
    # x = self.fc2(x)

    # x = torch.clamp(x,0,1)
    # x = self.activation_layer_fc2(x)

    x = self.fc3(x)
    # x = torch.sigmoid(x)

    x = x.squeeze(dim=1)

    return x

from catSNN import spikeLayer, transfer_model, SpikeDataset ,load_model, fuse_module
import catCuda,catCpp
add_sum=0
class SNN(nn.Module):
  def __init__(self,config):
      super(SNN, self).__init__()
      
      self.config = config
      history_length = config['history_lengths'][0]
      self.history_length = history_length
      conv_filters = config['conv_filters'][0]
      conv_width = config['conv_widths'][0]
      self.pooling_width = config['pooling_widths'][0]
      embedding_dims = config['embedding_dims']
      pc_hash_bits = config['pc_hash_bits']
      hash_dir_with_pc = config['hash_dir_with_pc']
      self.config = config
      self.conv_width = conv_width

      index_width = pc_hash_bits + 1

      if self.pooling_width > 0: 
        conv_output_size = (history_length - conv_width + 1)
        pooling_output_size = conv_output_size // self.pooling_width
      else:
        pooling_output_size = (history_length +c* (-conv_width + 1))
      self.total_output_size = pooling_output_size * conv_filters

      input_dim1 = self.total_output_size
      output_dim1 = config['hidden_neurons'][0]
      input_dim2 = output_dim1
      output_dim2 = config['hidden_neurons'][1]
      self.input_dim1 = input_dim1
      self.output_dim1 = output_dim1
      
      
      self.T = config['time_step']
      snn = spikeLayer(self.T)
      self.embedding_table = nn.Embedding(2 ** index_width, embedding_dims)
      self.snn=snn
      self.conv = snn.conv(embedding_dims, conv_filters, kernelSize=(conv_width,1),stride=1,padding=0)
      #self.pooling = snn.pool((pooling_width,1))

      self.fc1 = nn.Linear(input_dim1, output_dim1)
      #self.fc2 = snn.dense(input_dim2, output_dim2)
      self.fc3 = snn.dense((1,1,output_dim1), 1)
      #self.fc2 = snn.dense(128, 10, bias=True)


  def forward(self, x):
      x = x[:,-self.history_length:]
      x = self.embedding_table(x)
      x = torch.transpose(x, 1, 2)
      x = [x for _ in range(self.T)]
      x = torch.stack(x, dim=-1)  #float
      x = catCuda.getSpikes(x, 0.999)        
      

      
      global add_sum
      rate = torch.sum(x).item() / (x.size(0)*x.size(1)*x.size(2)*self.T)
      x = x.unsqueeze(-2)
      #print('conv',self.conv.weight,self.conv.bias)
      for _ in range(c):
        x = self.snn.spike(self.conv(x),theta=0.999)
      add_sum+=x.size(0)*x.size(1)*x.size(2)*self.T*self.conv_width*rate
      #print('111',add_sum)
      rate = torch.sum(x).item() / (x.size(0)*x.size(1)*x.size(2)*self.T)

      x = x.view(-1, self.total_output_size,self.T)
      
  
      # x = self.fc1(x)


      # #generate spike train
      # spikes_data = [x for _ in range(Quantized_activation_level)]
      # out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
      # out = catCuda.getSpikes(out, 0.999)
      # out = out.cpu()
      # #reshape
      # out_ = out.clone()
      # out_ = out_.reshape(out_.shape[0],out_.shape[2],out_.shape[1])
      # for i in range(out.shape[0]):
      #     out_[i] = out[i].mT

      # out_s = self.fc2(out_[0][0].reshape(1,10))
      # for i in range(Quantized_activation_level-1):
      #     out_s+=self.fc2(out_[0][i+1])
      # out_s = catCuda.getSpikes(out_sx, 1)/T
      # return out_s
      #print('fc1',self.fc1.weight,self.fc1.bias)
      x_list = []
      x_ = torch.split(x,1,2)
      
      for x_s in x_:
        x_s = self.fc1(x_s.squeeze(-1))
        x_list.append(x_s)
      x = torch.stack(x_list, dim=-1)
      x = catCuda.getSpikes(x, 1)

      x = x.unsqueeze(-2)
      x = x.unsqueeze(-2)

      # add_sum+=x.size(0)*self.input_dim1*self.output_dim1*self.T*rate
      # rate = torch.sum(x).item() / (x.size(0)*x.size(1)*self.T)
      # x = self.snn.spike(self.fc2(x))
      #print('fc3',self.fc3.weight,self.fc3.bias)
      x = self.fc3(x)
      # x = torch.sigmoid(x)
      add_sum+=x.size(0)*self.input_dim1*self.T*rate

      outs=self.snn.sum_spikes(x)/self.T
      outs=outs.view(-1)
      return outs

import copy
def bnfuse_blocks(src_dict):
    bnfuse_dict = {}
    prefix = []
    before_norm = []
    stk = 'conv.weight'
    for k in src_dict:
        if 'batchnorm' in k and k[0:14] not in prefix:
            prefix.append(k[0:14]) # 每一个bn层的前缀，拼接bias等即可得到bn层
            before_norm.append(stk[0:-4]) #下一层是bn的层
        elif 'fc3' in k :#and k[0:14] not in prefix
            bnfuse_dict[k] = src_dict[k]
        stk = k

    eps = 1e-3
    for i in range(0,len(prefix)):
        mu = src_dict[prefix[i]+'running_mean']
        var = src_dict[prefix[i]+'running_var']
        gamma = src_dict[prefix[i]+'weight']
        beta = src_dict[prefix[i]+'bias']
        W = src_dict[before_norm[i]+'weight']
        bias = src_dict[before_norm[i]+'bias'] # 有5个bias
    
        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        bnfuse_dict[before_norm[i]+'weight'] = torch.nn.Parameter(W)
        bnfuse_dict[before_norm[i]+'bias'] = torch.nn.Parameter(bias)

    bnfuse_dict['embedding_table.weight'] = src_dict['embedding_table.weight']
    return bnfuse_dict