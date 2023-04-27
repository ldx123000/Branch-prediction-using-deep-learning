///////////////////////////////////////////////////////////////////////
//  Copyright 2015 Samsung Austin Semiconductor, LLC.                //
///////////////////////////////////////////////////////////////////////


#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <inttypes.h>

using namespace std;

typedef uint32_t UINT32;
typedef int32_t INT32;
typedef uint64_t UINT64;
typedef uint64_t COUNTER;

#define NOT_TAKEN 0
#define TAKEN 1

#define FAILURE 0
#define SUCCESS 1

//JD2_2_2016
//typedef enum {
//  OPTYPE_OP               =2,
//  OPTYPE_BRANCH_COND      =3,
//  OPTYPE_RET              =4,
//  OPTYPE_BRANCH           =6,
//  OPTYPE_INDIRECT         =7,
//  OPTYPE_MAX              =8
//}OpType;

//JD2_17_2016 break down types into COND/UNCOND
typedef enum {
  OPTYPE_OP               =2,

  OPTYPE_RET_UNCOND = 3,
  OPTYPE_JMP_DIRECT_UNCOND = 4,
  OPTYPE_JMP_INDIRECT_UNCOND = 5,
  OPTYPE_CALL_DIRECT_UNCOND = 6,
  OPTYPE_CALL_INDIRECT_UNCOND = 7,

  OPTYPE_RET_COND = 8,
  OPTYPE_JMP_DIRECT_COND = 9,
  OPTYPE_JMP_INDIRECT_COND = 10,
  OPTYPE_CALL_DIRECT_COND = 11,
  OPTYPE_CALL_INDIRECT_COND = 12,

  OPTYPE_ERROR = 13,

  OPTYPE_MAX = 14
}OpType;



static inline UINT32 SatIncrement(UINT32 x, UINT32 max)
{
  if(x<max) return x+1;
  return x;
}

static inline UINT32 SatDecrement(UINT32 x)
{
  if(x>0) return x-1;
  return x;
}



#endif

