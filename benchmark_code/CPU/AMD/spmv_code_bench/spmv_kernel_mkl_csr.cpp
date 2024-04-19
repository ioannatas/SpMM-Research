#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include <mkl.h>

#include "macros/cpp_defines.h"

#include "spmv_bench_common.h"
#include "spmv_kernel.h"


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
	csr->format_name = (char *) "MKL_CSR";
	csr->ia = row_ptr;
	csr->ja = col_ind;
	csr->a = values;
	return csr;
}


void
compute_csr(CSRArrays * csr, ValueType * x , ValueType * y, INT_T k)
{
	char transa = 'N';
	ValueType alpha = 1.0, beta = 0.0;
	char matdescra[6];
	matdescra[0] = 'G';
    matdescra[1] = 'L';
    matdescra[2] = 'N';
    matdescra[3] = 'C';
	#if DOUBLE == 0
		// mkl_cspblas_scsrgemv(&transa, &csr->m , csr->a , csr->ia , csr->ja , x , y);
		mkl_scsrmm(&transa, &csr->m, &k, &csr->n, &alpha, matdescra, csr->a, csr->ja, csr->ia,  &(csr->ia[1]), &(x[0]), &k,  &beta, &(y[0]), &k);
	#elif DOUBLE == 1
		// mkl_cspblas_dcsrgemv(&transa, &csr->m , csr->a , csr->ia , csr->ja , x , y);
		mkl_dcsrmm(&transa, &csr->m, &k, &csr->n, &alpha, matdescra, csr->a, csr->ja, csr->ia,  &(csr->ia[1]), &(x[0]), &k,  &beta, &(y[0]), &k);
	#endif
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

