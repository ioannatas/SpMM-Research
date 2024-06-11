#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include <mkl.h>

#include "macros/cpp_defines.h"

#include "spmv_bench_common.h"
#include "spmv_kernel.h"


struct GEArrays : Matrix_Format
{
	ValueType * a;   // the values (of size NNZ)
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)

	GEArrays(long m, long n, long nnz) : Matrix_Format(m, n, nnz)
	{
		a = NULL;
		ia = NULL;
		ja= NULL;
	}

	virtual ~GEArrays()
	{
		free(a);
		free(ia);
		free(ja);
	}

	void spmm(ValueType * x, ValueType * y, INT_T k);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};


void compute_gemm(GEArrays * dense, ValueType * x , ValueType * y, INT_T k);


void
GEArrays::spmm(ValueType * x, ValueType * y, INT_T k)
{
	compute_gemm(this, x, y, k);
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
	// printf("hi2\n");
	struct GEArrays * dense = new GEArrays(m, n, nnz);
	// ValueType *dense_matrix = (ValueType*)malloc(sizeof(ValueType) * m * n);
	// ValueType * dense_matrix = (typeof(dense_matrix)) malloc(m * n * sizeof(*dense_matrix));
	ValueType * dense_matrix = NULL;
	dense_matrix = (typeof(dense_matrix)) aligned_alloc(64, (m * n) * sizeof(*dense_matrix));
	// printf("what%ld\n", dense_matrix[0]);
	  // Check if memory allocation was successful
    if (dense_matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
	for (long i = 0; i < m * n ; ++i) {
		// printf("...");
        dense_matrix[i] = 0.0;
		
		
    }
	// printf("hi3\n");
    // Fill in non-zero elements
    for (long i = 0; i < m; ++i) {
		// printf("hii\n");
        for (long j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
			// printf("%ld\n",j);
            dense_matrix[i * n + col_ind[j]] = values[j];
			// if((i * n + col_ind[j])>=m*n)
				// printf("hey %ld < %ld", i * n + col_ind[j], m*n);
			// if(j>=n)
			// 	printf("hey %ld < %ld", j, n);

        }
    }
	// printf("hi4\n");
	dense->format_name = (char *) "MKL_GEMM";
	dense->a = dense_matrix;
	return dense;
}


void
compute_gemm(GEArrays * dense, ValueType * x , ValueType * y, INT_T k)
{
	// char transa = 'N';
	ValueType alpha = 1.0, beta = 0.0;
	// char matdescra[6];
	// matdescra[0] = 'G';
    // matdescra[1] = 'L';
    // matdescra[2] = 'N';
    // matdescra[3] = 'C';
	#if DOUBLE == 0
		// mkl_cspblas_scsrgemv(&transa, &csr->m , csr->a , csr->ia , csr->ja , x , y);
		// mkl_scsrmm(&transa, &csr->m, &k, &csr->n, &alpha, matdescra, csr->a, csr->ja, csr->ia,  &(csr->ia[1]), &(x[0]), &k,  &beta, &(y[0]), &k);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dense->m, k, dense->n, alpha, dense->a, dense->n, x, k, beta, y, k);

	#elif DOUBLE == 1
		// mkl_cspblas_dcsrgemv(&transa, &csr->m , csr->a , csr->ia , csr->ja , x , y);
		// mkl_dcsrmm(&transa, &csr->m, &k, &csr->n, &alpha, matdescra, csr->a, csr->ja, csr->ia,  &(csr->ia[1]), &(x[0]), &k,  &beta, &(y[0]), &k);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dense->m, k, dense->n, alpha, dense->a, dense->n, x, k, beta, y, k);
	#endif
}


//==========================================================================================================================================
//= Print Statistics
//==========================================================================================================================================


void
GEArrays::statistics_start()
{
}


int
statistics_print_labels(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}


int
GEArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}

