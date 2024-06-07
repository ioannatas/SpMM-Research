#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


#include <mkl_spblas.h>
#include "macros/cpp_defines.h"

#include "spmv_bench_common.h"
#include "spmv_kernel.h"

#ifdef __cplusplus
extern "C"{
#endif
	#include "macros/macrolib.h"
	#include "time_it.h"
	#include "parallel_util.h"
#ifdef __cplusplus
}
#endif


struct CSRArrays : Matrix_Format
{
	ValueType * a;   // the values (of size NNZ)
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	sparse_matrix_t A;
	struct matrix_descr descr;

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


void compute_sparse_mm(CSRArrays * csr, ValueType * x , ValueType * y, INT_T k);


void
CSRArrays::spmm(ValueType * x, ValueType * y, INT_T k)
{
	compute_sparse_mm(this, x, y, k);
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
	struct CSRArrays * csr = new CSRArrays(m, n, nnz);
	double time, time_balance;

	// mkl_verbose(1);

	csr->format_name = (char *) "MKL_IE";
	csr->ia = row_ptr;
	csr->ja = col_ind;
	csr->a = values;
	time = time_it(1,
		// The input arrays provided are left unchanged except for the call to mkl_sparse_order, which performs ordering of column indexes of the matrix.
		// To avoid any changes to the input data, use mkl_sparse_copy.
		#if DOUBLE == 0
			mkl_sparse_s_create_csr(&csr->A, SPARSE_INDEX_BASE_ZERO, m, n, row_ptr, row_ptr+1, col_ind, values);
		#elif DOUBLE == 1
			mkl_sparse_d_create_csr(&csr->A, SPARSE_INDEX_BASE_ZERO, m, n, row_ptr, row_ptr+1, col_ind, values);
		#endif
		mkl_sparse_order(csr->A);  // Sort the columns.
	);
	printf("mkl mkl_sparse_s_create_csr + mkl_sparse_order time = %g\n", time);

	csr->descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	// csr->descr.mode = SPARSE_FILL_MODE_UPPER;
	csr->descr.diag = SPARSE_DIAG_NON_UNIT;
	const sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
	const INT_T expected_calls = 128;

	// Using SPARSE_MEMORY_AGGRESSIVE policy for some reason gives libgomp error 'out of memory' at 128 theads.
	//     SPARSE_MEMORY_NONE
	//         Routine can allocate memory only for auxiliary structures (such as for workload balancing); the amount of memory is proportional to vector size.
	//     SPARSE_MEMORY_AGGRESSIVE
	//         Default.
	//         Routine can allocate memory up to the size of matrix A for converting into the appropriate sparse format.
	// const sparse_memory_usage_t policy = SPARSE_MEMORY_AGGRESSIVE;
	const sparse_memory_usage_t policy = SPARSE_MEMORY_NONE;

	time_balance = time_it(1,
		mkl_sparse_set_mm_hint(csr->A, operation, csr->descr,SPARSE_LAYOUT_COLUMN_MAJOR, k, expected_calls);
		mkl_sparse_set_memory_hint(csr->A, policy);
		mkl_sparse_optimize(csr->A);
	);
	printf("mkl optimize time = %g\n", time_balance);

	#if 0
		for (i=0;i<csr_m;i++)
		{
			for (long j=row_ptr[i];j<row_ptr[i+1];j++)
			{
				if (j < row_ptr[i+1]-1 && col_ind[j] >= col_ind[j+1])
					error("%ld: unsorted columns: %d >= %d", i, col_ind[j], col_ind[j+1]);
			}
		}
	#endif

	// _Pragma("omp parallel")
	// {
		// printf("max_threads=%d , num_threads=%d , tnum=%d\n", omp_get_max_threads(), omp_get_num_threads(), omp_get_thread_num());
	// }
	// printf("out\n");

	csr->mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	return csr;
}


void
compute_sparse_mm(CSRArrays * csr, ValueType * x , ValueType * y, INT_T k)
{
		// printf("hi\n");
		// MKL_INT k_mkl = k;
		// MKL_INT n_mkl = csr->n;
		// MKL_INT m_mkl = csr->m;
		// const sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
		// const sparse_layout_t order = SPARSE_LAYOUT_ROW_MAJOR;
        #if DOUBLE == 0
		mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr->A, csr->descr, SPARSE_LAYOUT_COLUMN_MAJOR, x, k_mkl, n_mkl, 0.0, y, m_mkl);
		// mkl_sparse_s_mm(operation, 1.0, csr->A, csr->descr, order, x, n_mkl, k_mkl, 0.0, y, m_mkl);
        #elif DOUBLE == 1
		mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, csr->A, csr->descr, SPARSE_LAYOUT_COLUMN_MAJOR, x, k, csr->n, 0.0f, y, csr->m);
        #endif
		// printf("hi\n");
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

