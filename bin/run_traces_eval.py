#!/usr/bin/env python3

import common
# from common import PATHS, BENCHMARKS_INFO
import os

BINARY_NAME = 'tagescl64'
CONFIG_NAME = 'tagescl64'
NUM_THREADS = 32
TRACES_DIR = '/home2/dongxu/cbp2016.eval/traces/evaluationTraces'
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../traces/eval'
FILE_SUFFIX = '.csv' 

def main():
    tage_binary = os.path.dirname(os.path.abspath(__file__)) + '/../build/tracer/' + BINARY_NAME
    assert os.path.exists(tage_binary), 'Could not find the TAGE binary at ' + tage_binary
    os.makedirs(OUT_DIR, exist_ok=True)
    # os.makedirs(STATS_DIR, exist_ok=True)

    cmds = []
    for trace in os.listdir(TRACES_DIR):
        trace_path = os.path.join(TRACES_DIR,trace)
        if os.path.isfile(trace_path) and '.bt9.trace.gz' in trace:
            out_file = ('{}/{}_{}').format(OUT_DIR, trace[:-13],FILE_SUFFIX)
            # stats_file = ('{}/{}_{}').format(STATS_DIR, trace[:-13], FILE_SUFFIX)
            processed = os.path.exists(out_file) 
            if processed:
                continue
            # cmd = '{} {} {}'.format(tage_binary, trace_path, stats_file, out_file)
            cmd = '{} {} {}'.format(tage_binary, trace_path, out_file)
            cmds.append(cmd)
    if len(cmds) == 0:
        print("all traces done")    
    common.run_parallel_commands_local(cmds, NUM_THREADS)


if __name__ == '__main__':
    main()