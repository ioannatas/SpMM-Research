#if !defined(QUICKSORT_GEN_TYPE_1)
	#error "QUICKSORT_GEN_TYPE_1 not defined: value type"
#elif !defined(QUICKSORT_GEN_TYPE_2)
	#error "QUICKSORT_GEN_TYPE_2 not defined: index type"
#elif !defined(QUICKSORT_GEN_TYPE_3)
	#error "QUICKSORT_GEN_TYPE_3 not defined: auxiliary data value type"
#elif !defined(QUICKSORT_GEN_SUFFIX)
	#error "QUICKSORT_GEN_SUFFIX not defined"
#endif

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"


#define QUICKSORT_GEN_EXPAND(name)  CONCAT(name, QUICKSORT_GEN_SUFFIX)

#undef  _TYPE_V
#define _TYPE_V  QUICKSORT_GEN_EXPAND(_TYPE_V)
typedef QUICKSORT_GEN_TYPE_1  _TYPE_V;

#undef  _TYPE_I
#define _TYPE_I  QUICKSORT_GEN_EXPAND(_TYPE_I)
typedef QUICKSORT_GEN_TYPE_2  _TYPE_I;

#undef  _TYPE_AD
#define _TYPE_AD  QUICKSORT_GEN_EXPAND(_TYPE_AD)
typedef QUICKSORT_GEN_TYPE_3  _TYPE_AD;


//==========================================================================================================================================
//= Functions
//==========================================================================================================================================


//------------------------------------------------------------------------------------------------------------------------------------------
//- Quicksort
//------------------------------------------------------------------------------------------------------------------------------------------

#undef  quicksort
#define quicksort  QUICKSORT_GEN_EXPAND(quicksort)
void quicksort(_TYPE_V * A, long N, _TYPE_AD * aux_data, _TYPE_I * partitions_buf);


#undef  quicksort_parallel
#define quicksort_parallel  QUICKSORT_GEN_EXPAND(quicksort_parallel)
void quicksort_parallel(_TYPE_V * A, long N, _TYPE_AD * aux_data, _TYPE_I * partitions_buf);

#undef  quicksort_parallel_inplace
#define quicksort_parallel_inplace  QUICKSORT_GEN_EXPAND(quicksort_parallel_inplace)
void quicksort_parallel_inplace(_TYPE_V * A, long N, _TYPE_AD * aux_data, _TYPE_I * partitions_buf);


//------------------------------------------------------------------------------------------------------------------------------------------
//- Insertionsort
//------------------------------------------------------------------------------------------------------------------------------------------

#undef  insertionsort
#define insertionsort  QUICKSORT_GEN_EXPAND(insertionsort)
void insertionsort(_TYPE_V * A, long N, _TYPE_AD * aux_data);

