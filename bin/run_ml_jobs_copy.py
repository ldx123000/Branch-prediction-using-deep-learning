#!/usr/bin/env python3
# set thread limit
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from collections import namedtuple
import glob
# import os
import shutil
import numpy as np
#from numba import jit

import common
from common import PATHS, BENCHMARKS_INFO, ML_INPUT_PARTIONS

import h5py

import subprocess


Job = namedtuple('Job', ['benchmark', 'experiment_name', 'config_file', 'training_mode'])
JOBS = [
    # Job('LONG_MOBILE-1', 'testrun1', 'big', 'float'),
    # Job('SHORT_SERVER-139', 'testrun1', 'mini_250', 'mini'),
    Job('SHORT_MOBILE-5', 'testrun1', 'big', 'float'),
]

config_mode_tuples = [
    ('mini_1k', 'mini'),
    ('mini_2k', 'mini'),
    ('mini_250', 'mini'),
    ('mini_500', 'mini'),
    ('big', 'float'),
    ('tarsa', 'tarsa'),
]

BATCH_SIZE = 512
TRAINING_STEPS = [100, 100, 100]
FINE_TUNING_STEPS = [50, 50, 50]
LEARNING_RATE = 0.1
LASSO_COEFFICIENT = 0.0
REGULARIZATION_COEFFICIENT = 0.0
CUDA_DEVICE = 0
LOG_VALIDATION = True

CREATE_WORKDIRS = True
WORKDIRS_OVERRIDE_OK = True

EXPERIMENTS = "/home2/dongxu/BranchNet/experiment"
CSVS = "/home2/dongxu/BranchNet/traces_csv"
TRACES_ROOT = "/home2/dongxu/BranchNet/traces_csv/training"
DATASETS_DIR = "/home2/dongxu/BranchNet/dataset"



def get_br_pcs(trace_path):
    print('getting br pcs from ' + trace_path)
    struct_type = [
        ('br_pc', np.uint64),
        ('target', np.uint64),
        ('dir', np.uint8),
        ('type', np.uint8)]
    record_dtype = np.dtype(struct_type, align=False)

    x = np.genfromtxt(trace_path,delimiter=',',dtype=record_dtype)
    print('len: ' + str(len(set(x['br_pc']))))
    # print(set(x['br_pc']))
    return set(x['br_pc'])

def get_br_pcs_hdf5(trace_path):
    file_ptr = h5py.File(trace_path, 'r')
    print('len: ' + str(len(file_ptr.keys())))
    return file_ptr.keys()

def create_run_command(workdir, training_datasets, evaluation_datasets,
                       validation_datasets, br_pc, training_mode):
    return ('cd {workdir}; python3 run.py '
            '-trtr {tr} -evtr {ev} -vvtr {vv} --br_pc {pc} --batch_size {batch} '
            '-bsteps {bsteps} -fsteps {fsteps} --log_progress {log_validation} '
            '-lr {lr} -gcoeff {gcoeff} -rcoeff {rcoeff} -mode {mode} '
            '-c config.yaml --cuda_device {cuda} &> run_logs/{pc}.out'.format(
                workdir=workdir,
                tr=' '.join(training_datasets),
                ev=' '.join(evaluation_datasets),
                vv=' '.join(validation_datasets),
                pc=hex(br_pc),
                batch=BATCH_SIZE,
                bsteps=' '.join(map(str, TRAINING_STEPS)),
                fsteps=' '.join(map(str, FINE_TUNING_STEPS)),
                log_validation='--log_validation' if LOG_VALIDATION else '',
                lr=LEARNING_RATE,
                gcoeff=LASSO_COEFFICIENT,
                rcoeff=REGULARIZATION_COEFFICIENT,
                mode=training_mode,
                cuda=CUDA_DEVICE,
            ))

def get_workdir(job):
    return '{}/{}/{}'.format(EXPERIMENTS, job.experiment_name, job.benchmark)

def create_workdirs():
    if not WORKDIRS_OVERRIDE_OK:
        for job in JOBS:
            workdir = get_workdir(job)
            assert not os.path.exists(workdir), 'Experiment already exists at {}'.format(workdir)

    for job in JOBS:
        workdir = get_workdir(job)
        branchnet_dir = '/home2/dongxu/BranchNet/src/branchnet'
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(workdir + '/run_logs', exist_ok=True)
        os.makedirs(workdir + '/visual_logs', exist_ok=True)
        for filename in ['run.py', 'dataset_loader.py', 'model.py']:
            shutil.copy(branchnet_dir + '/' + filename, workdir)
        shutil.copyfile('{}/configs/{}.yaml'.format(branchnet_dir, job.config_file), workdir + '/config.yaml')

def create_workdirs_with_config(config):
    if not WORKDIRS_OVERRIDE_OK:
        for job in JOBS:
            workdir = get_workdir(job)
            assert not os.path.exists(workdir), 'Experiment already exists at {}'.format(workdir)

    for job in JOBS:
        workdir = get_workdir(job)
        branchnet_dir = os.path.dirname(os.path.abspath(__file__)) + '/../src/branchnet'
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(workdir + '/run_logs', exist_ok=True)
        os.makedirs(workdir + '/visual_logs', exist_ok=True)
        for filename in ['run.py', 'dataset_loader.py', 'model.py']:
            shutil.copy(branchnet_dir + '/' + filename, workdir)
        shutil.copyfile('{}/configs/{}.yaml'.format(branchnet_dir, config), workdir + '/config.yaml')


def create_job_commands():
    cmds = []
    for job in JOBS:
        workdir = get_workdir(job)
        brs = get_br_pcs(TRACES_ROOT + "/" + job.benchmark + "_.csv")
        # brs = get_br_pcs(DATASETS_DIR + "/" + job.benchmark + "_dataset.hdf5")
        # datasets_dir = '{}/{}'.format(DATASETS_DIR, job.benchmark)
        training_datasets = [DATASETS_DIR + "/training/" + job.benchmark + "_dataset.hdf5"]
        evaluation_datasets = [DATASETS_DIR + "/eval/" + job.benchmark + "_dataset.hdf5"]
        validation_datasets = [DATASETS_DIR + "/test/" + job.benchmark + "_dataset.hdf5"]

        for br in brs:
            cmd = create_run_command(workdir, training_datasets, evaluation_datasets,
                                     validation_datasets, br, job.training_mode)
            cmds.append(cmd)
    return cmds

#undone
def create_job_commands_all_config(benchmark):
    cmds = []
    for job in JOBS:
        workdir = get_workdir(job)
        brs = get_br_pcs(TRACES_ROOT + "/" + job.benchmark + "_dataset.hdf5")
        # datasets_dir = '{}/{}'.format(DATASETS_DIR, job.benchmark)
        training_datasets = [DATASETS_DIR + "/training/" + job.benchmark + "_dataset.hdf5"]
        evaluation_datasets = [DATASETS_DIR + "/eval/" + job.benchmark + "_dataset.hdf5"]
        validation_datasets = [DATASETS_DIR + "/test/" + job.benchmark + "_dataset.hdf5"]

        for br in brs:
            cmd = create_run_command(workdir, training_datasets, evaluation_datasets,
                                     validation_datasets, br, job.training_mode)
            cmds.append(cmd)
    return cmds

# @jit
def main():
    print(CREATE_WORKDIRS)
    if CREATE_WORKDIRS:
        create_workdirs()
    print('creating job commands')
    cmds = create_job_commands()
    batches = []
    max_batch_size = 1
    if len(cmds) <= max_batch_size:
        batches.append(cmds)
    else:
        for i in np.reshape(cmds[:int(len(cmds)/max_batch_size)*max_batch_size], (int(len(cmds)/max_batch_size), max_batch_size)):
            batches.append(i.tolist())
        batches.append(cmds[-(len(cmds) % max_batch_size):])
    index = 0
    for batch in batches:
        # print(batch[0])
        # return
        # common.run_parallel_commands_local(batch, 1)
        for cmd in batch:
            continue
            # print(index)
            # index += 1
            # print(cmd)
            # subprocess.call(cmd, shell=True)
# pip3 install torch==1.11.0+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# pip3 install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
