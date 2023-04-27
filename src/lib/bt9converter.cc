#CANNOT COMPILE
#include "cbp2016_utils.h"
#include "trace_interface.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <string>

struct Args {
  char* input_trace_path;
  char* output_file_path;
};
Args parse_args(int argc, char** argv) {
  if(argc != 2) {
    std::cerr << "Usage: " << argv[0]
              << " input_trace output_path\n";
    std::exit(1);
  }
  return {argv[1], argv[2]};
}

int main(int argc, char** argv) {
  const auto args     = parse_args(argc, argv);
  // std::string str(args.input_trace_path);
  std::vector<BT9> br_trace = read_trace_bt9(args.input_trace_path, INT32_MAX);
  std::ofstream ofs(args.output_file_path);
  ofs << "pc,target,direction,type\n";
  for(int i = 1; i < br_trace.size(); i++) {
    ofs << "," << br_trace[i].to_string() << "\n";
  }
}