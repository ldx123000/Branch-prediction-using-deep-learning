#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cbp2016_utils.h"
#include "trace_interface.h"
#include "utils.h"

#define TAGESCL_64 1
#define TAGESCL_56 2
#define MTAGE_SC 3
#define GTAGE_SC 4
#define GTAGE_SC_NOLOCAL 5
#define GTAGE 6

#ifndef PREDICTOR_CONFIG
#define PREDICTOR_CONFIG TAGESCL_64
#endif

#if PREDICTOR_CONFIG == TAGESCL_64
#define PREDICTOR_SIZE 64
#include "tagescl.h"

#elif PREDICTOR_CONFIG == TAGESCL_56
#define PREDICTOR_SIZE 56
#include "tagescl.h"

#elif PREDICTOR_CONFIG == MTAGE_SC
#include "mtagesc.h"

#elif PREDICTOR_CONFIG == GTAGE_SC
#define ONLY_GTAGE
#include "mtagesc.h"

#elif PREDICTOR_CONFIG == GTAGE_SC_NOLOCAL
#define ONLY_GTAGE
#define DISABLE_LOCAL
#include "mtagesc.h"

#elif PREDICTOR_CONFIG == GTAGE
#define ONLY_GTAGE
#define DISABLE_SC
#define DISABLE_LOCAL
#include "mtagesc.h"

#else
static_assert(false, "Config name not supported");
#endif

OpType convert_brtype_to_optype(BR_TYPE br_type) {
  switch(br_type) {
    case BR_TYPE::NOT_BR:
      return OPTYPE_OP;
    case BR_TYPE::COND_DIRECT:
      return OPTYPE_JMP_DIRECT_COND;
    case BR_TYPE::COND_INDIRECT:
      return OPTYPE_JMP_INDIRECT_COND;
    case BR_TYPE::UNCOND_DIRECT:
      return OPTYPE_JMP_DIRECT_UNCOND;
    case BR_TYPE::UNCOND_INDIRECT:
      return OPTYPE_JMP_INDIRECT_UNCOND;
    case BR_TYPE::CALL:
      return OPTYPE_CALL_DIRECT_UNCOND;
    case BR_TYPE::RET:
      return OPTYPE_RET_UNCOND;
    default:
      std::cerr << "Unexpected BR_TYPE: " << static_cast<int8_t>(br_type)
                << '\n';
      std::exit(1);
  }
}

struct Args {
  char* input_trace_path;
  char* output_file_path;
  int   max_brs;
  char* hard_br_file_path;
};
Args parse_args(int argc, char** argv) {
  if(argc < 3 || argc > 5) {
    std::cerr << "Usage: " << argv[0]
              << " input_trace output_path [max_brs [hard_br_file_path]]\n";
    std::exit(1);
  }

  if(argc == 3)
    return {argv[1], argv[2], std::numeric_limits<int>::max(), nullptr};
  if(argc == 4)
    return {argv[1], argv[2], std::stoi(argv[3]), nullptr};
  else
    return {argv[1], argv[2], std::stoi(argv[3]), argv[4]};
} 

int main(int argc, char** argv) {
  const auto args     = parse_args(argc, argv);
  std::string str(args.input_trace_path);
  if(str.find(".bt9.trace.gz")){
    std::vector<BT9> br_trace = read_trace_bt9(args.input_trace_path, args.max_brs);
    std::ofstream ofs(args.output_file_path);
    ofs << "pc,target,direction,type,opcode\n";
    for(int i = 1; i < br_trace.size(); i++) {
      ofs << br_trace[i].to_string() << "\n";
    }
  }else if(str.find(".bz2")){
    const auto br_trace = read_trace(args.input_trace_path, args.max_brs);
    std::ofstream ofs(args.output_file_path);
    ofs << "pc,target,direction,type,opcode\n";
    for(const auto& br : br_trace) {
      ofs << std::to_string(br.pc)+","+std::to_string(br.target)+","+std::to_string(br.direction ? 1 : 0) + "," + std::to_string(static_cast<int>(br.type)) + "," + std::to_string(br.opcode)  << "\n";
    }
  }
  return 0;
}