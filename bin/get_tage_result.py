#!/usr/bin/env python3

import os

import common
from common import PATHS, BENCHMARKS_INFO, ML_INPUT_PARTIONS

TARGET_BENCHMARKS = '/home2/dongxu/BranchNet/tage_stats'
HARD_BR_DIR = '/home2/dongxu/BranchNet/hard_br'
TAGE_CONFIG_NAME = 'tagescl64'
NUM_BRS_TO_PRINT = 100
PRODUCE_HARD_BR_FILES = True
HARD_BR_FILE_NAME = 'top50'
TOP = 50

BRANCH_STATS_TEMPLATE = (
    "*****    {line_name: <15.15}{weighted_specifier:<12.12}"
    "{weighted_accuracy:>10.10}, {weighted_mpki:>10.10}, "
    "{weighted_mpki_ratio:>10.10}, {weighted_total:>12.12} || "
    "{unweighted_specifier:<12.12}{unweighted_accuracy:>10.10}, "
    "{unweighted_mpki:>10.10}, {unweighted_mpki_ratio:>10.10}, "
    "{unweighted_total:>12.12}    *****")

def print_benchmark_header(benchmark, inp):
  """Prints a line with the benchmark and input names."""
  padded_benchmark = '  {}  '.format(benchmark)
  padded_inp = '  {}  '.format(inp)
  print('{:=>156}'.format(''))
  print('{benchmark:=>77.77}, {input_name:=<77.77}'.format(
      benchmark=padded_benchmark, input_name=padded_inp))
  print('{:=>156}'.format(''))

def print_branch_stats_header():
  """Prints a line describing the fields in the following stats line."""
  print(BRANCH_STATS_TEMPLATE.format(
      line_name='Line Name:',
      weighted_specifier='Weighted',
      weighted_accuracy='Accuracy',
      weighted_mpki='MPKI',
      weighted_mpki_ratio='MPKI Ratio',
      weighted_total='Total',
      unweighted_specifier='Unweighted',
      unweighted_accuracy='Accuracy',
      unweighted_mpki='MPKI',
      unweighted_mpki_ratio='MPKI Ratio',
      unweighted_total='Total',
  ))

def print_branch_stats_line(line_name, aggregate_branch_stats,
                            weighted_total_mpki, unweighted_total_mpki):
  """Prints a line for branch prediction stats for a given branch."""
  print(BRANCH_STATS_TEMPLATE.format(
      line_name=line_name,
      weighted_specifier='',
      weighted_accuracy='{:.2%}'.format(
          aggregate_branch_stats.weighted_stats.accuracy),
      weighted_mpki='{:.3f}'.format(
          aggregate_branch_stats.weighted_stats.mpki),
      weighted_mpki_ratio='{:.2%}'.format(
          aggregate_branch_stats.weighted_stats.mpki/ weighted_total_mpki),
      weighted_total='{:.0f}'.format(
        aggregate_branch_stats.weighted_stats.total),
      unweighted_specifier='',
      unweighted_accuracy='{:.2%}'.format(
          aggregate_branch_stats.unweighted_stats.accuracy),
      unweighted_mpki='{:.3f}'.format(
          aggregate_branch_stats.unweighted_stats.mpki),
      unweighted_mpki_ratio='{:.2%}'.format(
          aggregate_branch_stats.unweighted_stats.mpki/ unweighted_total_mpki),
      unweighted_total='{:.0f}'.format(
        aggregate_branch_stats.unweighted_stats.total),
  ))


class BenchmarkInputStats:
  """Encapsulate per input branch statistics for a benchmark."""

  def __init__(self, inp):
    """Read and preprocess all Tage statistics during construction."""
    self.inp = inp
    self.stats = common.read_tage_stats(TAGE_CONFIG_NAME)

  def get_mpki_dict(self):
    br_mpki = {}
    for br, stats in self.stats.items():
      if br == -1: continue
      br_mpki[br] = stats.weighted_stats.mpki
    return br_mpki

  def print_detailed_summary(self, brs=None):
    """Print the summary of the top mispredicting branches of the benchmark"""
    print_branch_stats_header()
    weighted_total_mpki = self.stats[-1].weighted_stats.mpki
    unweighted_total_mpki = self.stats[-1].unweighted_stats.mpki
    print_branch_stats_line("Total:", self.stats[-1], weighted_total_mpki,
                            unweighted_total_mpki)


    selected_mpki_weighted = 0
    selected_mpki_unweighted = 0
    for br in brs:
      if br in self.stats:
        print_branch_stats_line(hex(br) + ' ', self.stats[br],
                                weighted_total_mpki, unweighted_total_mpki)
        selected_mpki_weighted += self.stats[br].weighted_stats.mpki
        selected_mpki_unweighted += self.stats[br].unweighted_stats.mpki
      else:
        print(BRANCH_STATS_TEMPLATE.format(
            line_name=hex(br),
            weighted_specifier='',
            weighted_accuracy='---',
            weighted_mpki='---',
            weighted_mpki_ratio='---',
            weighted_total='---',
            unweighted_specifier='',
            unweighted_accuracy='---',
            unweighted_mpki='---',
            unweighted_mpki_ratio='---',
            unweighted_total='---',))
    print(BRANCH_STATS_TEMPLATE.format(
        line_name='Selected Brs',
        weighted_specifier='',
        weighted_accuracy='N/A',
        weighted_mpki='{:.3f}'.format(selected_mpki_weighted),
        weighted_mpki_ratio='{:.2%}'.format(
            selected_mpki_weighted / weighted_total_mpki),
        weighted_total='---',
        unweighted_specifier='',
        unweighted_accuracy='N/A',
        unweighted_mpki='{:.3f}'.format(selected_mpki_unweighted),
        unweighted_mpki_ratio='{:.2%}'.format(
            selected_mpki_unweighted / unweighted_total_mpki),
        unweighted_total='---',))

    #print('Number of selected branches:', num_brs)
    #print('Selected Branches MPKI:', brs_mpki)
    #print('Selected Branches Accuracy:', brs_correct / brs_total)


def greedy_select_top_brs(list_inputs, mpki_dicts, sorted_brs, num_brs):
  selected_brs = [] 
  next_br_idx = [0] * len(list_inputs)
  for _ in range(num_brs):
    next_br_total_mpki = [0.0] * len(list_inputs)
    next_br_pc = [0] * len(list_inputs)
    for j, inp in enumerate(list_inputs):
      while sorted_brs[inp][next_br_idx[j]] in selected_brs:
        next_br_idx[j] += 1
      br = sorted_brs[inp][next_br_idx[j]]

      total_mpki = 0
      for inppp in list_inputs:
        total_mpki += mpki_dicts[inppp][br]
      next_br_total_mpki[j] = total_mpki
      next_br_pc[j] = br

    max_j = next_br_total_mpki.index(max(next_br_total_mpki))
    selected_brs.append(next_br_pc[max_j])
  return selected_brs


def get_input_set_stats(benchmark, list_inputs, brs=None):
  stats = {}
  mpki_dicts = {}
  sorted_brs = {}
  for inp in list_inputs:
    stats[inp] = BenchmarkInputStats(benchmark, inp)
    mpki_dicts[inp] = stats[inp].get_mpki_dict()
    sorted_brs[inp] = sorted(mpki_dicts[inp], key=mpki_dicts[inp].get, reverse=True)

  if brs is None:
    brs = greedy_select_top_brs(list_inputs, mpki_dicts, sorted_brs, NUM_BRS_TO_PRINT)

  for inp in stats:
    stats[inp].print_detailed_summary(brs)
  return brs

import pandas as pd

def main():
    for trace in os.listdir(TARGET_BENCHMARKS):
        if trace == 'SHORT_MOBILE-1_tagescl64.csv':
          brs=[]
          trace_path = TARGET_BENCHMARKS + "/" + trace
          data = pd.read_csv(trace_path, usecols=['Branch PC', 'Correct Predictions','Total'])
          data = data.head(TOP+1)
          # sorted_data = data.sort_values(by='Accuracy')
          correct=0
          total=0
          for index,row in data.iterrows():
            if index>0:
                correct+=int(row['Correct Predictions'])
                total+=int(row['Total'])
          accuracy = 1.0*correct/total
          print(trace_path,correct,total,accuracy)
        # validation_inputs = ML_INPUT_PARTIONS[benchmark]['validation_set']
        # test_inputs = ML_INPUT_PARTIONS[benchmark]['test_set']

        #brs = get_input_set_stats(benchmark, validation_inputs)

        # Uncomment this line to observe test set stats using the selected
        # branches.
        #get_input_set_stats(benchmark, test_inputs, brs=brs)

        # Uncomment this line to observe the stats of top mispredicting
        # branches of the test set.
        #get_input_set_stats(benchmark, test_inputs)

        # if PRODUCE_HARD_BR_FILES:
        #     os.makedirs(HARD_BR_DIR, exist_ok=True)
        #     filepath = '{}/{}_top{}'.format(
        #         HARD_BR_DIR, trace[:-14], TOP)
        #     with open(filepath, 'w') as f:
        #         for br in brs:
        #             f.write(hex(br) + '\n')


if __name__ == '__main__':
    main()