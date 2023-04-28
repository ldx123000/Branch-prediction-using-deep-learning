import argparse
import copy
import os
import numpy as np
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import h5py
from model import CNN,CNN_no_bn,SNN,bnfuse_blocks
from dataset_loader import BranchDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

config_file = os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'

print(torch.cuda.device_count())

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config

__args__ = {"validation_traces": None, "log_validation": True}

class LossLogger():
  def __init__(self):
    self.training_loss = []
    self.group_lasso_loss = []
    self.fc_reg_loss = []
    self.learning_rate = []
    self.validation_steps = []
    self.validation_loss = []
    self.validation_accuracy = []

  def log_training(self, prediction_loss, group_lasso_loss,
                   fc_reg_loss, learning_rate):
    self.training_loss.append(prediction_loss)
    self.group_lasso_loss.append(group_lasso_loss)
    self.fc_reg_loss.append(fc_reg_loss)
    self.learning_rate.append(learning_rate)

  def log_validation(self, validation_loss, validation_accuracy):
    self.validation_steps.append(len(self.training_loss) - 1)
    self.validation_loss.append(validation_loss)
    self.validation_accuracy.append(validation_accuracy)

  def validation_is_behind(self):
    return ((len(self.validation_steps) == 0) or
            (self.validation_steps[-1] < len(self.training_loss) - 1))

  def plot_loss(self, filename):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,20))
    ax1.plot(self.training_loss, label='Training loss')
    ax1.plot(self.validation_steps, self.validation_loss, label='Validation loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(self.validation_steps, self.validation_accuracy)
    ax2.set_title('Validation Accuracy')
    ax1.set_ylim(0, 1)
    ax3.plot(self.group_lasso_loss, label='Group Lasso Regulariozation')
    ax3.plot(self.fc_reg_loss, label='Fully-connected Regulariozation')
    ax3.set_title('Regularization Losses')
    ax3.legend()
    ax4.plot(self.learning_rate)
    ax4.set_title('Learning Rate')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

class ModelWrapper():
  """A wrapper around a branch predictor model for training and evaluation.
  Attributes:
    model: A Pytorch neural network module.
    training_traces: Paths of the traces used for the training set.
    validation_traces: Paths of the traces used for the validation set.
    test_traces: Paths of the traces used for the test set.
    br_pc: the PC of the target branch (as an int).
    branchnet_config: branchnet configuration dictionary.
    batch_size: Training and Inference batch size.
    dataloader: The active pytorch DataLoader for branchnet (could be None).
    dataloader_traces: The set of branch traces in the active dataloader.
  """
  def __init__(self, model, *,
               br_pc, branchnet_config, batch_size, cuda_device, log_progress):
    """Simply initializes class attributes based on the constructor arguments
    """
    self.model = model
    self.br_pc = br_pc
    self.branchnet_config = branchnet_config
    self.batch_size = batch_size
    self.device = torch.device('cpu') if cuda_device == -1 else torch.device('cuda:'+str(cuda_device))#torch.device('cuda')
    self.logger = LossLogger() if log_progress else None
    # if torch.cuda.device_count() > 1:
    #         #print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         model = nn.DataParallel(model)
    self.model.to(self.device)
    self.dataloader_dict = {}
    self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

  def get_dataloader(self, traces, eval=False):
    dict_key = ' '.join(traces)
    if dict_key in self.dataloader_dict:
      return self.dataloader_dict[dict_key]
    
    history_length = max(self.branchnet_config['history_lengths'])

    # # need to have enough history for the largest slice
    # history_length = max(self.branchnet_config['history_lengths'])
    # if any(self.branchnet_config['shifting_pooling']):
    #   # need to add extra history to support random shifting
    #   history_length += max(self.branchnet_config['pooling_widths'])

    dataset = BranchDataset(
        traces,
        br_pc=self.br_pc,
        history_length=history_length,
        pc_bits=self.branchnet_config['pc_bits'],
        pc_hash_bits=self.branchnet_config['pc_hash_bits'],
        hash_dir_with_pc=self.branchnet_config['hash_dir_with_pc'])
    if len(dataset) > 0:
      
      dataloader = DataLoader(
          dataset, batch_size=self.batch_size,
          shuffle=not eval, num_workers=6, pin_memory=True,prefetch_factor=64,
                                 persistent_workers=True) #edited here change num_workers from 6 to 2
    else:
      dataloader = None

    self.dataloader_dict[dict_key] = dataloader
    return dataloader
    
  def get_loss_values(self, outs, labels, group_lasso_coeff,
                      fc_regularization_coeff):
    loss = self.criterion(outs, labels)
    prediction_loss_value = loss.item()

    group_lasso_loss_value = 0

    if group_lasso_coeff > 0:
      group_lasso_loss = group_lasso_coeff * self.model.group_lasso_loss()
      group_lasso_loss_value = group_lasso_loss.item()
      loss += group_lasso_loss

    fc_reg_loss_value = 0
    if fc_regularization_coeff > 0:
      fc_reg_loss = fc_regularization_coeff * self.model.fc_weights_l1_loss()
      fc_reg_loss_value = fc_reg_loss.item()
      loss += fc_reg_loss

    return (loss, prediction_loss_value,
            group_lasso_loss_value, fc_reg_loss_value)


  def train(self, training_traces, training_steps, learning_rate,
            group_lasso_coeff, fc_regularization_coeff):
    """Train one epoch using the training set"""
    self.model.train()
    
    training_set_loader = self.get_dataloader(training_traces)
    try:
      optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
      scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)
      #print('model')
      #print(self.model)
      
      for num_steps in training_steps:
        # print('Training for {} steps with learning rate {}'.format(
        #     num_steps, scheduler.get_lr()[0]))
        step = 0
        while step < num_steps:
          for inps, labels in training_set_loader:
            outs = self.model(inps.to(self.device))
            
            # if step == 0:
            #   print('inps size at ' + str(step))
            #   print(inps.size())
            #   print('outs')
            #   print(outs)
            #   print('labels')
            #   print(labels)
            (loss, prediction_loss_value,
            group_losso_loss_value, fc_reg_loss_value) = self.get_loss_values(
                outs, labels.to(self.device), group_lasso_coeff,
                fc_regularization_coeff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if self.logger is not None:
            #   self.logger.log_training(prediction_loss_value, 
            #                            group_losso_loss_value,
            #                            fc_reg_loss_value,
            #                            scheduler.get_lr()[0])

            #   if (step + 1) % 500 == 0:
            #     #print('Evaluating the validation set')
            #     #if __args__.validation_traces:
            #     #  corrects, total, loss = self.eval(__args__.validation_traces)
            #     #  self.model.train()
            #     #  self.logger.log_validation(loss, corrects / total * 100)
            #     #else:
            #     #  self.logger.log_validation(0, 0)
            #     self.logger.plot_loss('visual_log_{}.pdf'.format(hex(self.br_pc)))

            step += 1
            if step >= num_steps:
              break

        scheduler.step()
        if self.logger is not None:
          if self.logger.validation_is_behind():
            #edited here
            if __args__['validation_traces'] and __args__['log_validation']:
              print('Evaluating the validation set')
              #edited here
              corrects, total, loss = self.eval(__args__['validation_traces'])
              self.model.train()
              self.logger.log_validation(loss, corrects / total * 100)
            else:
              self.logger.log_validation(0, 0)
          #edited here
          #self.logger.plot_loss('/home2/dongxu/BranchNet/test/1/{}.pdf'.format(hex(self.br_pc)))
    except Exception as e:
      print(e)

  def eval(self, trace_list):
    """Evaluate the predictor on the traces passed.
    """

    self.model.eval()
    test_set_loader = self.get_dataloader(trace_list, eval=True)
    #loader could be None if no branches of br_pc is found in the traces
    corrects = 0
    total = 0
    total_loss = 0
    if test_set_loader is not None: 
      for inps, labels in test_set_loader:
        inps = inps.to(self.device)
        labels = labels.to(self.device)
        outs = self.model(inps)
        _, loss_value, _, _ = self.get_loss_values(
            outs, labels, 0.0, 0.0)
        predictions = outs > 0
        targets = labels > 0.5
        corrects += (predictions == targets).sum().item()
        total += len(predictions)
        total_loss += loss_value
      total_loss = total_loss / len(test_set_loader)
    return corrects, total, total_loss
    

  def load_checkpoint(self, checkpoint_path):
    """ Loads a checkpoint file to initialize the model state
    """
    self.model.load_state_dict(torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage))

  def save_checkpoint(self, checkpoint_path):
    """ Saves the model state into a checkpoint file
    """
    torch.save(self.model.state_dict(), checkpoint_path)

def run_on(branches):
  global add_sum,multi_sum,is_sum
  add_sum=0
  multi_sum=0
  count = 0
  cor=0
  tot=0
  f = open(res_path, "w")

  for branch in branches:
    test_traces = test_path
    torch.cuda.empty_cache()
    is_sum=0
    count += 1

    try:
      brpc = int(branch.replace('br_indices_', ''), 16)
      print(str(hex(brpc)) + ' ' + str(count) + '/' + str(len(branches)))
    except ValueError as e:
      print()

    lst = []
    for _ in range(10):
      model = CNN(config)
      test_model = CNN_no_bn(config)
      snn_model = SNN(config)
      try:
        cf = config
        bs = 512*4
        cuda = -1
        if torch.cuda.device_count() > 0:
          cuda = 0
        lp = True
        model_wrapper = ModelWrapper(model,
              br_pc= brpc,
              branchnet_config=cf,
              batch_size= bs,
              cuda_device = cuda,
              log_progress=lp)
        
        training_traces = [train_path]
        training_steps = [100,100,100]
        learning_rate = 0.1
        fc_regularization_coeff = 0.0
        model_wrapper.train(training_traces,
                            training_steps,
                            learning_rate,
                            0.0,
                            fc_regularization_coeff)
        
        dst_dict = copy.deepcopy(snn_model.state_dict())
        src_dict = copy.deepcopy(model.state_dict())
        f_model = open('dstname1.txt','w')
        for i in dst_dict.keys():
            f_model.write(str(i)+'\n')
        f_model.close()
        f_model = open('srcname1.txt','w')
        for i in src_dict.keys():
            f_model.write(str(i)+'\n')
        f_model.close()
        
        src_dict = bnfuse_blocks(src_dict)

        reshape_dict = {}
        print(src_dict.keys())
        for (k, v) in src_dict.items():
              # if k=="stem.0.weight":
                  # reshape_dict['stem.weight'] = nn.Parameter(v.reshape(dst_dict['stem.weight'].shape)*6)   
              # if k=="head.0.weight":
                  # reshape_dict['head.weight'] = nn.Parameter(v.reshape(dst_dict['head.weight'].shape)*6)
              if k in dst_dict.keys():
                  dst_dict[k] = torch.tensor(dst_dict[k], dtype=torch.float32)
                  v = torch.tensor(v, dtype=torch.float32)
                  # print('not miss', k, dst_dict[k].shape, v.reshape(dst_dict[k].shape).shape)
                  if 'weight' in k:
                      reshape_dict[k] = nn.Parameter(v.reshape(dst_dict[k].shape))
                  else:
                      reshape_dict[k] = nn.Parameter(v.reshape(dst_dict[k].shape))
                  #reshape_dict['embedding_table.weight'] = src_dict['embedding_table.weight']

        snn_model.load_state_dict(reshape_dict, strict=False)

        dst_dict = copy.deepcopy(test_model.state_dict())
        for (k, v) in src_dict.items():
              # if k=="stem.0.weight":
                  # reshape_dict['stem.weight'] = nn.Parameter(v.reshape(dst_dict['stem.weight'].shape)*6)   
              # if k=="head.0.weight":
                  # reshape_dict['head.weight'] = nn.Parameter(v.reshape(dst_dict['head.weight'].shape)*6)
              if k in dst_dict.keys():
                  dst_dict[k] = torch.tensor(dst_dict[k], dtype=torch.float32)
                  v = torch.tensor(v, dtype=torch.float32)
                  # print('not miss', k, dst_dict[k].shape, v.reshape(dst_dict[k].shape).shape)
                  if 'weight' in k:
                      reshape_dict[k] = nn.Parameter(v.reshape(dst_dict[k].shape))
                  else:
                      reshape_dict[k] = nn.Parameter(v.reshape(dst_dict[k].shape))
                  #reshape_dict['embedding_table.weight'] = src_dict['embedding_table.weight']
        test_model.load_state_dict(reshape_dict, strict=False)
        #fuse_module(model)
        # for param_tensor in model.state_dict():
        #   print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        # print(".......................")
        # for param_tensor in snn_model.state_dict():
        #   print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())
        #transfer_model(model, snn_model)
        print('SNN')
      
        model_wrapper = ModelWrapper(snn_model,
              br_pc= brpc,
              branchnet_config=cf,
              batch_size= bs,
              cuda_device = cuda,
              log_progress=lp)
        
        model_wrapper_2 = ModelWrapper(test_model,
              br_pc= brpc,
              branchnet_config=cf,
              batch_size= bs,
              cuda_device = cuda,
              log_progress=lp)
        
        model_wrapper_1 = ModelWrapper(model,
              br_pc= brpc,
              branchnet_config=cf,
              batch_size= bs,
              cuda_device = cuda,
              log_progress=lp)

        is_sum=1
        
        corrects, total, _ = model_wrapper.eval([test_traces])
        lst.append((corrects,total))
        accuracy_SNN = 0 if total == 0 else corrects/total

        print('{}/{} accuracy of {}/{}: {} out of {} ({}%)'.format(
            str(count) , str(len(branches)),test_traces, str(hex(brpc)), corrects, total, accuracy_SNN*100.0))

        corrects, total, _ = model_wrapper_2.eval([test_traces])
        accuracy = 0 if total == 0 else corrects/total
        print('{}/{} accuracy of {}/{}: {} out of {} ({}%)'.format(
            str(count) , str(len(branches)),test_traces, str(hex(brpc)), corrects, total, accuracy*100.0))
        corrects, total, _ = model_wrapper_1.eval([test_traces])
        accuracy = 0 if total == 0 else corrects/total
        print('{}/{} accuracy of {}/{}: {} out of {} ({}%)'.format(
            str(count) , str(len(branches)),test_traces, str(hex(brpc)), corrects, total, accuracy*100.0))
        
        if accuracy - accuracy_SNN < 0.05:
            break
      except Exception as e:
        print(e)

    corrects,total = max(lst, key=lambda x: x[0])
    cor+=corrects
    tot+=total
    accuracy = 0 if total == 0 else corrects/total
    print('final: {}/{} accuracy of {}/{}: {} out of {} ({}%)'.format(
                  str(count) , str(len(branches)),test_traces, str(hex(brpc)), corrects, total, accuracy*100.0))
    print('{}/{} accuracy of {}/{}: {} out of {} ({}%)'.format(
                  str(count) , str(len(branches)),test_traces, str(hex(brpc)), corrects, total, accuracy*100.0),file=f)

  acc = 0 if tot == 0 else cor/tot
  print('total accuracy: {} out of {} ({}%)'.format(
                 cor, tot, acc*100.0),file=f)
  print('Adders:',int(add_sum),file=f)
  print('Multipliers:',int(multi_sum),file=f)
  f.close()

# traces = [eval_path] + [test_path]
TARGET_BENCHMARKS = os.path.dirname(os.path.abspath(__file__)) + '/../../dataset_good'
TRAIN_FILE = TARGET_BENCHMARKS+'/training'
EVAL_FILE = TARGET_BENCHMARKS+'/training'
OUTPUT_FILE = os.path.dirname(os.path.abspath(__file__)) + '/../../result_SNN'
# list1 = [42, 78, 150, 294, 400, 582 ,650 ,800 ,900 ,1000]
# for h in list1:
for trace in os.listdir(TRAIN_FILE):
  # if trace == 'SHORT_MOBILE-9_dataset.hdf5':
    train_path = TRAIN_FILE + "/" + trace
    test_path = EVAL_FILE + "/" + trace
    os.makedirs(OUTPUT_FILE, exist_ok=True)
    res_path = OUTPUT_FILE + "/" + trace[:-13] + ".out"

    train_file = h5py.File(train_path, 'r')
    test_file = h5py.File(test_path, 'r')#eval_path

    branches = set(train_file.keys()).intersection(set(test_file.keys()))
    branches.discard('history')
    print(branches)
    run_on(branches)