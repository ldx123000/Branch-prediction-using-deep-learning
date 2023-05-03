/*
 * Copyright (C) 2004-2021 Intel Corporation.
 * SPDX-License-Identifier: MIT
 */

/*! @file
 * This file exemplifies XED usage on IA-32 and Intel(R) 64 architectures.
 */

#include "pin.H"
extern "C"
{
#include "xed-interface.h"
#include "trace_interface.h"
}
#include <iostream>
#include <iomanip>
#include <fstream>
using std::cerr;
using std::endl;
using std::hex;
using std::cout;

/* ===================================================================== */
/* Global Variables */
/* ===================================================================== */
FILE *trace_file;

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

INT32 Usage()
{
    cerr << "This tool create trace files for BranchNet" << endl;
    return -1;
}

/* ===================================================================== */

#if 0
VOID use_xed(ADDRINT pc)
{
#if defined(TARGET_IA32E)
    static const xed_state_t dstate = {XED_MACHINE_MODE_LONG_64, XED_ADDRESS_WIDTH_64b};
#else
    static const xed_state_t dstate = {XED_MACHINE_MODE_LEGACY_32, XED_ADDRESS_WIDTH_32b};
#endif
    xed_decoded_inst_t xedd;
    xed_decoded_inst_zero_set_mode(&xedd, &dstate);

    //Pass in the proper length: 15 is the max. But if you do not want to
    //cross pages, you can pass less than 15 bytes, of course, the
    //instruction might not decode if not enough bytes are provided.
    const unsigned int max_inst_len = 15;

    xed_error_enum_t xed_code = xed_decode(&xedd, reinterpret_cast< UINT8* >(pc), max_inst_len);
    BOOL xed_ok               = (xed_code == XED_ERROR_NONE);
    if (xed_ok)
    {
        char buf[2048];

        // set the runtime adddress for disassembly
        xed_uint64_t runtime_address = static_cast< xed_uint64_t >(pc);

        xed_format_context(XED_SYNTAX_INTEL, &xedd, buf, 2048, runtime_address, 0, 0);
    }
}
#endif

/* ===================================================================== */

//VOID Instruction(INS ins, VOID* v) { INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)use_xed, IARG_INST_PTR, IARG_END); }

BR_TYPE get_br_type(const INS& inst) {

     xed_decoded_inst_t* xedd      = INS_XedDec(inst);
     xed_iclass_enum_t inst_iclass = xed_decoded_inst_get_iclass(xedd);

     if ((inst_iclass == XED_ICLASS_XBEGIN) || (inst_iclass == XED_ICLASS_XEND))
         return BR_TYPE::NOT_BR;

     switch(INS_Category(inst)) {
       case XED_CATEGORY_COND_BR:
         return INS_IsDirectBranch(inst) ? BR_TYPE::COND_DIRECT :
                                           BR_TYPE::COND_INDIRECT;
       case XED_CATEGORY_UNCOND_BR:
         return INS_IsDirectBranch(inst) ? BR_TYPE::UNCOND_DIRECT :
                                        BR_TYPE::UNCOND_INDIRECT;
       case XED_CATEGORY_CALL:
         return BR_TYPE::CALL;
       case XED_CATEGORY_RET:
         return BR_TYPE::RET;
       default:
         return BR_TYPE::NOT_BR;
     }
}

void dump_br(const ADDRINT fetch_addr, const BOOL resolve_dir,
             const ADDRINT branch_target, UINT32 br_type, UINT64 inst) {

     HistElt current_hist_elt;
     current_hist_elt.pc        = fetch_addr;
     current_hist_elt.target    = branch_target;
     current_hist_elt.direction = resolve_dir ? 1 : 0;
     current_hist_elt.type      = static_cast<BR_TYPE>(br_type);
     current_hist_elt.opcode    = inst;

     int elements_written = fwrite(&current_hist_elt, sizeof(current_hist_elt), 1, trace_file);

     assert(elements_written > 0);

#if 0
     fprintf(trace_file, "0x%lx -> 0x%lx ? %d - %d\n", fetch_addr, branch_target, resolve_dir, br_type);
#endif
}


VOID instrumentation_function(INS inst, VOID* v) {

     BR_TYPE br_type = get_br_type(inst);
     //cout<< INS_Size(inst)<<endl;
     //cout<< INS_Disassemble(inst)<<endl;
     if(br_type != BR_TYPE::NOT_BR) {
      cout<< INS_Disassemble(inst)<<endl;
       INS_InsertCall(inst, IPOINT_BEFORE, (AFUNPTR)dump_br, IARG_INST_PTR,
                      IARG_BRANCH_TAKEN, IARG_BRANCH_TARGET_ADDR, IARG_UINT32,
                      static_cast<uint32_t>(br_type), IARG_UINT64, INS_Opcode(inst), IARG_END);
     }
}

VOID fini_function(int, VOID* v) {
  fclose(trace_file);
}

/* ===================================================================== */

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char* argv[])
{
    if (PIN_Init(argc, argv)) return Usage();
    trace_file = fopen("tracer.out", "w");
    INS_AddInstrumentFunction(instrumentation_function, 0);
    PIN_AddFiniFunction(fini_function, 0);

    PIN_StartProgram(); // Never returns
    return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */

// make clean; make
// ../../../pin -t obj-intel64/tracer.so -- /bin/ls
// bzip2 tracer.out
// ~/BranchNet/build/tage/tagescl64 tracer.out.bz2 try.csv