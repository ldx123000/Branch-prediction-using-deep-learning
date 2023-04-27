/* Author: Stephen Pruett and Siavash Zangeneh
 * Date: 9/26/17
 * Description: The interface for defining the branch traces.
 */

#ifndef TRACE_INTERFACE_H
#define TRACE_INTERFACE_H
#include "cbp2016_utils.h"

#include <cstdint>
#include <vector>
#include <string>

enum class BR_TYPE : int8_t {
  NOT_BR          = 0,
  COND_DIRECT     = 1,
  COND_INDIRECT   = 2,
  UNCOND_DIRECT   = 3,
  UNCOND_INDIRECT = 4,
  CALL            = 5,
  RET             = 6,
};

struct HistElt {
  uint64_t pc;
  uint64_t target;
  uint8_t  direction;
  BR_TYPE  type;
  uint64_t opcode;
  //uint64_t opcode;
} __attribute__((packed));

struct BT9 {
  BT9(uint64_t pc, uint64_t target, bool direction, OpType type, uint64_t opcode) {
    this->pc = pc;
    this->target = target;
    this->direction = direction;
    this->type = type;
    this->opcode = opcode;
  }

  uint64_t pc;
  uint64_t target;
  bool  direction;
  OpType  type;
  uint64_t opcode;

  std::string to_string() {
    return std::to_string(pc) + "," + std::to_string(target) + "," + std::to_string(direction ? 1 : 0) + "," + std::to_string(static_cast<int>(type)) + "," +std::to_string(opcode);
  }
} __attribute__((packed));

std::vector<HistElt> read_trace(char* input_trace, int max_brs);
std::vector<BT9> read_trace_bt9(char* trace_path, int max_brs);

#endif  // TRACE_INTERFACE_H