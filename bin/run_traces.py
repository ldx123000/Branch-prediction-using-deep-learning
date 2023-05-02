#!/usr/bin/env python3

import common
from common import PATHS
import os

BINARY_NAME = 'tagescl64'
CONFIG_NAME = 'tagescl64'
NUM_THREADS = 32
TRAIN_TRACES_DIR = os.path.dirname(os.path.abspath(__file__)) + PATHS['train_traces_dir']
TEST_TRACES_DIR = os.path.dirname(os.path.abspath(__file__)) + PATHS['test_traces_dir']

OUT_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__)) + PATHS['output_traces_dir'] + '/training'
OUT_TEST_DIR = os.path.dirname(os.path.abspath(__file__)) + PATHS['output_traces_dir'] + '/test'
FILE_SUFFIX = '.csv' 

def main():
    tage_binary = os.path.dirname(os.path.abspath(__file__)) + '/../build/tracer/' + BINARY_NAME
    assert os.path.exists(tage_binary), 'Could not find the TAGE binary at ' + tage_binary
    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUT_TEST_DIR, exist_ok=True)

    cmds = []
    for trace in os.listdir(TRAIN_TRACES_DIR):
        trace_path = os.path.join(TRAIN_TRACES_DIR,trace)
        if os.path.isfile(trace_path) and '.bt9.trace.gz' in trace:
            out_file = ('{}/{}_{}').format(OUT_TRAIN_DIR, trace[:-13],FILE_SUFFIX)
            processed = os.path.exists(out_file) 
            if processed:
                continue
            cmd = '{} {} {}'.format(tage_binary, trace_path, out_file)
            cmds.append(cmd)
    for trace in os.listdir(TEST_TRACES_DIR):
        trace_path = os.path.join(TEST_TRACES_DIR,trace)
        if os.path.isfile(trace_path) and '.bt9.trace.gz' in trace:
            out_file = ('{}/{}_{}').format(OUT_TEST_DIR, trace[:-13],FILE_SUFFIX)
            processed = os.path.exists(out_file) 
            if processed:
                continue
            cmd = '{} {} {}'.format(tage_binary, trace_path, out_file)
            cmds.append(cmd)
    if len(cmds) == 0:
        print("all traces done")    
    common.run_parallel_commands_local(cmds, NUM_THREADS)


if __name__ == '__main__':
    main()