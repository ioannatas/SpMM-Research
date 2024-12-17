#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "taco.h"
#include <mkl.h>

#include "macros/cpp_defines.h"

#include "spmv_bench_common.h"
#include "spmv_kernel.h"
using namespace taco;

struct CSRTensors : Matrix_Format
{
	Format dcsr({Sparse,Sparse});
    Format   rm({Dense,Dense});
    Format   cm({Dense,Dense}, {1,0});

    Tensor<double> C({B.getDimension(0), 1000}, rm);
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)

	CSRTensors(long m, long n, long nnz) : Matrix_Format(m, n, nnz)
	{
		a = NULL;
		ia = NULL;
		ja= NULL;
	}

	~CSRTensors()
	{
		free(a);
		free(ia);
		free(ja);
	}

	void spmm(ValueType * x, ValueType * y, INT_T k);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};


void compute_csr(CSRTensors * csr, ValueType * x , ValueType * y, INT_T k);


void
CSRTensors::spmm(ValueType * x, ValueType * y, INT_T k)
{
	compute_csr(this, x, y, k);
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz , int k=0)
{
	struct CSRTensors * csr = new CSRTensors(m, n, nnz);
	csr->format_name = (char *) "TACO_CSR";
	csr->ia = row_ptr;
	csr->ja = col_ind;
	csr->a = values;
	return csr;

   
    for (int i = 0; i < C.getDimension(0); ++i) {
        for (int j = 0; j < C.getDimension(1); ++j) {
        C.insert({i,j}, unif(gen));
        }
    }
    C.pack();
}


void
compute_csr(CSRTensors * csr, ValueType * x , ValueType * y, INT_T k)
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
	// printf("oops3\n");
}


//==========================================================================================================================================
//= Print Statistics
//==========================================================================================================================================


void
CSRTensors::statistics_start()
{
}


int
statistics_print_labels(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}


int
CSRTensors::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}

