#include "trace_interface.h"
#include "bt9_reader.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

std::vector<HistElt> read_trace(char* input_trace, int max_brs) {
  std::vector<HistElt> history;

  const int BUFFER_SIZE = 4096;
  char      cmd[BUFFER_SIZE];

  auto out = snprintf(cmd, BUFFER_SIZE, "bzip2 -dc %s", input_trace);
  assert(out < BUFFER_SIZE);

  FILE*   fptr = popen(cmd, "r");
  HistElt history_elt_buffer;
  for(int i = 0; i < max_brs && fread(&history_elt_buffer,
                                      sizeof(history_elt_buffer), 1, fptr) == 1;
      ++i) {
    history.push_back(history_elt_buffer);
  }

  if(max_brs == std::numeric_limits<int>::max() && !feof(fptr)) {
    std::cerr << "Error while reading the input trace file\n";
    std::exit(1);
  }
  pclose(fptr);

  return history;
}

std::vector<BT9> read_trace_bt9(char* trace_path, int max_brs) {
    std::vector<BT9> history;
    std::string str(trace_path);
    bt9::BT9Reader bt9_reader(trace_path);
    OpType opType;
    uint64_t PC;
    bool branchTaken;
    uint64_t branchTarget;
    uint64_t opcode;

    int counter = 0;

    for (auto it = bt9_reader.begin(); it != bt9_reader.end() && counter < max_brs; ++it) {

    try {
      bt9::BrClass br_class = it->getSrcNode()->brClass();
      opType = OPTYPE_ERROR; 

      if ((br_class.type == bt9::BrClass::Type::UNKNOWN) && (it->getSrcNode()->brNodeIndex())) { //only fault if it isn't the first node in the graph (fake branch)
        opType = OPTYPE_ERROR; //sanity check
      }

      else if (br_class.type == bt9::BrClass::Type::RET) {
        if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
          opType = OPTYPE_RET_COND;
        else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
          opType = OPTYPE_RET_UNCOND;
        else {
          opType = OPTYPE_ERROR;
        }
      }
      else if (br_class.directness == bt9::BrClass::Directness::INDIRECT) {
        if (br_class.type == bt9::BrClass::Type::CALL) {
          if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
            opType = OPTYPE_CALL_INDIRECT_COND;
          else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
            opType = OPTYPE_CALL_INDIRECT_UNCOND;
          else {
            opType = OPTYPE_ERROR;
          }
        }
        else if (br_class.type == bt9::BrClass::Type::JMP) {
          if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
            opType = OPTYPE_JMP_INDIRECT_COND;
          else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
            opType = OPTYPE_JMP_INDIRECT_UNCOND;
          else {
            opType = OPTYPE_ERROR;
          }
        }
        else {
          opType = OPTYPE_ERROR;
        }
      }
      else if (br_class.directness == bt9::BrClass::Directness::DIRECT) {
        if (br_class.type == bt9::BrClass::Type::CALL) {
          if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL) {
            opType = OPTYPE_CALL_DIRECT_COND;
          }
          else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL) {
            opType = OPTYPE_CALL_DIRECT_UNCOND;
          }
          else {
            opType = OPTYPE_ERROR;
          }
        }
        else if (br_class.type == bt9::BrClass::Type::JMP) {
          if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL) {
            opType = OPTYPE_JMP_DIRECT_COND;
          }
          else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL) {
            opType = OPTYPE_JMP_DIRECT_UNCOND;
          }
          else {
            opType = OPTYPE_ERROR;
          }
        }
        else {
          opType = OPTYPE_ERROR;
        }
      }
      else {
        opType = OPTYPE_ERROR;
      }

      PC = it->getSrcNode()->brVirtualAddr();
      branchTaken = it->getEdge()->isTakenPath();
      branchTarget = it->getEdge()->brVirtualTarget();
      opcode = it->getSrcNode()->brOpcode();

      history.push_back(BT9(PC, branchTarget, branchTaken, opType, opcode));
      // if (counter % 1000000 == 0) {
      //   cout << "check at " + std::to_string(counter) + "th branch: " << history.at(counter).to_string() << endl;
      // }
      counter++;
    } catch(...) {

    }
  }
  // cout << "bt9 read, trace length: " << history.size() << endl;
  return history;
}