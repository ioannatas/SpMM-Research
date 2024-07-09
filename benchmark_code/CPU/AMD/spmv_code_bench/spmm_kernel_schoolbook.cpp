#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


#include "macros/cpp_defines.h"

#include "spmv_bench_common.h"
#include "spmv_kernel.h"
#ifdef __cplusplus
extern "C"{
#endif
	#include "macros/macrolib.h"
	#include "time_it.h"
	#include "parallel_util.h"
	#include "array_metrics.h"
#ifdef __cplusplus
}
#endif

struct CSRArrays : Matrix_Format
{
	ValueType * a;   // the values (of size NNZ)
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)

	CSRArrays(long m, long n, long nnz) : Matrix_Format(m, n, nnz)
	{
		a = NULL;
		ia = NULL;
		ja= NULL;
	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);
	}

	void spmm(ValueType * x, ValueType * y, INT_T k);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};


void compute_csr(CSRArrays * csr, ValueType * x , ValueType * y, INT_T k);


void
CSRArrays::spmm(ValueType * x, ValueType * y, INT_T k)
{
	compute_csr(this, x, y, k);
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz , int k=0)
{
	struct CSRArrays * csr = new CSRArrays(m, n, nnz);
	csr->format_name = (char *) "Naive_CSR_CPU";
	csr->ia = row_ptr;
	csr->ja = col_ind;
	csr->a = values;
	return csr;
}

void subkernel_csr(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, long i_s, long i_e){

		for (long n = 0; n < k; n++) { 
		#pragma omp for
		for (long i = 0; i < csr->m; i++) {
        	 

                ValueType val, tmp, compensation;
                compensation = 0;
                ValueType sum = 0;
                val = 0;
				
				for (long j = csr->ia[i]; j < csr->ia[i + 1]; j++) {
					val = csr->a[j] * x[n * csr->n + csr->ja[j]];// - compensation;
                    // tmp = sum + val;
                    // compensation = (tmp - sum) - val;
                    sum += val;
                }
                y[i * k + n] = sum;
            }
        }
	
}

void
compute_csr(CSRArrays * csr, ValueType * x , ValueType * y, INT_T k)
{
	
	// int num_threads = omp_get_max_threads();
	// printf("threads: %d\n", num_threads);
    #pragma omp parallel
	{
        
    }
	
}


//==========================================================================================================================================
//= Print Statistics
//==========================================================================================================================================


void
CSRArrays::statistics_start()
{
}


int
statistics_print_labels(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}


int
CSRArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}

