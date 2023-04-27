#!/usr/bin/env python3

from collections import namedtuple
import glob
import os
import shutil
import numpy as np

import common
from common import PATHS, BENCHMARKS_INFO, ML_INPUT_PARTIONS

Job = namedtuple('Job', ['benchmark', 'experiment_name', 'config_file', 'training_mode'])
JOBS = [
    Job('SHORT_SERVER-1', 'testrun1', 'mini_250', 'mini'),
    Job('SHORT_MOBILE-107', 'testrun1', 'mini_250', 'mini')
]

config_mode_tuples = [
    ('mini_1k', 'mini'),
    ('mini_2k', 'mini'),
    ('mini_250', 'mini'),
    ('mini_500', 'mini'),
    ('big', 'float'),
    ('tarsa', 'tarsa'),
]

BATCH_SIZE = 2048
TRAINING_STEPS = [100, 100, 100]
FINE_TUNING_STEPS = [50, 50, 50]
LEARNING_RATE = 0.1
LASSO_COEFFICIENT = 0.0
REGULARIZATION_COEFFICIENT = 0.0
CUDA_DEVICE = 0
LOG_VALIDATION = False

CREATE_WORKDIRS = True
WORKDIRS_OVERRIDE_OK = True

EXPERIMENTS = "/home2/dongxu/BranchNet/experiments"
CSVS = "/home2/dongxu/BranchNet/traces_csv"
TRACES_ROOT = "/home2/dongxu/BranchNet/traces_csv/training"
DATASETS_DIR = "/home2/dongxu/BranchNet/dataset"

def get_br_pcs(trace_path):
    print('getting ' + trace_path)
    struct_type = [
        ('br_pc', np.uint64),
        ('target', np.uint64),
        ('dir', np.uint8),
        ('type', np.uint8)]
    record_dtype = np.dtype(struct_type, align=False)

    x = np.genfromtxt(trace_path,delimiter=',',dtype=record_dtype)
    
    return set(x['br_pc'])

def all_same(pcs1, pcs2):
    print(len(pcs1))
    print(len(pcs2))
    # if len(pcs1) != len(pcs2):
    #     return False
    for pc in pcs1:
        if pc not in pcs2:
            return False
    return True

def has_same_branches(trace1, trace2):
    return all_same(get_br_pcs(trace1), get_br_pcs(trace2))

def no_same(pcs1, pcs2):
    longer = pcs1
    shorter = pcs2
    if (len(pcs2) > len(pcs1)):
        longer = pcs2
        shorter = pcs1
    for pc in longer:
        if pc in shorter:
            print(pc)
            return False
    return True
    
def has_same_pc(trace1, trace2):
    return no_same(get_br_pcs(trace1), get_br_pcs(trace2))

def main():
    print(has_same_pc('/home2/dongxu/BranchNet/traces_csv/training/SHORT_MOBILE-1_.csv', '/home2/dongxu/BranchNet/traces_csv/training/SHORT_MOBILE-2_.csv'))


if __name__ == '__main__':
    main()