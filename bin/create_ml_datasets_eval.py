#!/usr/bin/env python3

import h5py
import multiprocessing
import numpy as np
import os
import struct

import common
# from common import PATHS, BENCHMARKS_INFO


# TARGET_BENCHMARKS = ['leela']
# HARD_BRS_FILE = 'top100'
NUM_THREADS = 32
PC_BITS = 30


def read_branch_trace(trace_path):
    struct_type = [
        ('br_pc', np.uint64),
        ('target', np.uint64),
        ('dir', np.uint8),
        ('type', np.uint8)]
    record_dtype = np.dtype(struct_type, align=False)

    x = np.genfromtxt(trace_path,delimiter=',',dtype=record_dtype)

    return x['br_pc'].copy(), x['dir'].copy()


def create_new_dataset(dataset_path, pcs, directions):
    '''
    Create a new hdf5 file and copy over the history to the file.
    Branch PCs and directions are concatenated. Only the least significant bits of PC
    (controlled by PC_BITS) are stored.
    '''
    stew_bits = PC_BITS + 1
    if stew_bits < 8:
        stew_dtype = np.uint8
    elif stew_bits < 16:
        stew_dtype = np.uint16
    elif stew_bits < 32:
        stew_dtype = np.uint32
    elif stew_bits < 64:
        stew_dtype = np.uint64
    else:
        assert False, 'History elements of larger than 64 bits are not supported'

    pc_mask = (1 << PC_BITS) - 1
    fptr = h5py.File(dataset_path, 'w-')
    processed_history = ((pcs & pc_mask) << 1) | directions
    processed_history = processed_history.astype(stew_dtype)
    fptr.attrs['pc_bits'] = PC_BITS
    fptr.create_dataset(
        "history",
        data=processed_history,
        compression='gzip',
        compression_opts=9,
    )
    return fptr

def get_work_items(TRACES_DIR, DATASETS_DIR):
    work_items = []    
    os.makedirs(DATASETS_DIR, exist_ok=True)
    for trace in os.listdir(TRACES_DIR):
        trace_path = TRACES_DIR + "/" + trace
        dataset_path = '{}/{}_dataset.hdf5'.format(DATASETS_DIR, trace[:-5]) 
        work_items.append((trace_path, dataset_path))

    return work_items


def gen_dataset(trace_path, dataset_path):
    processed = os.path.exists(dataset_path) 
    if processed:
        return
    print('reading file', trace_path)
    pcs, directions = read_branch_trace(trace_path)

    print('Creating output file', dataset_path)
    fptr = create_new_dataset(dataset_path, pcs, directions)

    print(trace_path + " " + dataset_path)

    for br_pc in pcs:
        print('processing branch {}'.format(hex(br_pc)))
        #find indicies of hard branches
        trace_br_indices = np.argwhere(pcs == br_pc).squeeze(axis=1)
        fptr.create_dataset(
            'br_indices_{}'.format(hex(br_pc)),
            data=trace_br_indices,
            compression='gzip',
            compression_opts=9,
        )
        num_taken = np.count_nonzero(
            np.bitwise_and(pcs == br_pc, directions == 1))
        num_not_taken = np.count_nonzero(
            np.bitwise_and(pcs == br_pc, directions == 0))
        fptr.attrs['num_taken_{}'.format(hex(br_pc))] = num_taken
        fptr.attrs['num_not_taken_{}'.format(hex(br_pc))] = num_not_taken


def main():
    TRAINING_TRACES_DIR = '/home2/dongxu/BranchNet/traces_csv/training'
    TRAINING_DATASETS_DIR = '/home2/dongxu/BranchNet/dataset/training'
    EVAL_TRACES_DIR = '/home2/dongxu/BranchNet/traces_csv/eval'
    EVAL_DATASETS_DIR = '/home2/dongxu/BranchNet/dataset/eval'
    TEST_TRACES_DIR = '/home2/dongxu/BranchNet/traces_csv/test'
    TEST_DATASETS_DIR = '/home2/dongxu/BranchNet/dataset/test'  
    work_items = get_work_items(EVAL_TRACES_DIR, EVAL_DATASETS_DIR)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        pool.starmap(gen_dataset, work_items)


if __name__ == '__main__':
    main()
