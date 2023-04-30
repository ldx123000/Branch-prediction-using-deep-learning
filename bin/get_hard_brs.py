#!/usr/bin/env python3

import os

import common
from common import PATHS

TARGET_BENCHMARKS = os.path.dirname(os.path.abspath(__file__)) + PATHS['tage_stats_dir']
HARD_BR_DIR = os.path.dirname(os.path.abspath(__file__)) + PATHS['hard_br_dir']
HARD_BR_ACC_DIR = os.path.dirname(os.path.abspath(__file__)) + PATHS['hard_br_acc_dir']
TAGE_CONFIG_NAME = 'tagescl64'
NUM_BRS_TO_PRINT = 100
PRODUCE_HARD_BR_FILES = True
HARD_BR_FILE_NAME = 'top50'
TOP = 50

import pandas as pd

def main():
    for trace in os.listdir(TARGET_BENCHMARKS):
        brs=[]
        trace_path = TARGET_BENCHMARKS + "/" + trace
        data = pd.read_csv(trace_path, usecols=['Branch PC', 'Mispredictions'])
        data = data.head(TOP+1)
        
        for index,row in data.iterrows():
           if index>0:
              brs.append(int(row['Branch PC'],16))

        if PRODUCE_HARD_BR_FILES:
            os.makedirs(HARD_BR_DIR, exist_ok=True)
            filepath = '{}/{}_top{}'.format(
                HARD_BR_DIR, trace[:-14], TOP)
            with open(filepath, 'w') as f:
                for br in brs:
                    f.write(hex(br) + '\n')

        data = pd.read_csv(trace_path, usecols=['Branch PC', 'Correct Predictions', 'Total'])
        data = data.head(TOP+1)
        # sorted_data = data.sort_values(by='Accuracy')
        correct,total = 0,0
        for index,row in data.iterrows():
           if index>0:
              correct+=row['Correct Predictions']
              total+=row['Total']

        if PRODUCE_HARD_BR_FILES:
            os.makedirs(HARD_BR_ACC_DIR, exist_ok=True)
            filepath = '{}/{}_top{}'.format(
                HARD_BR_ACC_DIR, trace[:-14], TOP)
            with open(filepath, 'w') as f:
                print('{} total accuracy: {} out of {} ({}%)'.format(trace[:-14],correct,total, 1.0*correct/total*100.0),file=f)

if __name__ == '__main__':
    main()
