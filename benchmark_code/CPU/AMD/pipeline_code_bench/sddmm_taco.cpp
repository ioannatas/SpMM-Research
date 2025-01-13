#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "/various/itasou/taco/include/taco.h"

#include "macros/cpp_defines.h"

#include "sddmm_bench_common.h"
#include "sddmm_kernel.h"
using namespace taco;

struct CSRTensors : Matrix_Format
{

    Tensor<ValueType> Mask;
	Tensor<ValueType> x;
	Tensor<ValueType> z;
	INT_T * row_ptr;
	INT_T * col_ind;

	CSRTensors(long m, long n, long nnz) : Matrix_Format(m, n, nnz),Mask({m, m}, Format({Sparse, Sparse})),x({m, n}, Format({Dense, Dense})),z({n, m}, Format({Dense, Dense}, {1, 0}))
	{
		// Format csr({Sparse,Sparse});
    	// Format   rm({Dense,Dense});
    	// Format   cm({Dense,Dense}, {1,0});
		// Mask({m, m}, csr);
		// x({m, n}, rm);
		// z({n,m}, cm);
	}

	~CSRTensors()
	{
	}

	void sddmm(ValueType * y);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};


void compute_csr(CSRTensors * csr, ValueType * y);


void
CSRTensors::sddmm(ValueType * y)
{
	compute_csr(this, y);
	// for (int i=0;i<this->nnz;i++)
	// 		printf("%lf ",y[i]);
	// 	printf("sddmm \n");
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long nnz , long n, ValueType *x, ValueType *z)
{
	printf("csr_to_format\n");
	struct CSRTensors * csr = new CSRTensors(m, n, nnz);
	csr->x=csr->x;
	csr->format_name = (char *) "TACO_CSR";
	for (long i = 0; i < csr->x.getDimension(0); ++i) {
        for (long j = 0; j < csr->x.getDimension(1); ++j) {
			csr->x.insert({i,j}, x[i*n+j]);
			csr->z.insert({j,i}, z[j*n+i]);
        }
    }
    csr->x.pack();
	csr->z.pack();
	for (long i = 0; i < csr->m; ++i) {
		for (INT_T j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
			// if (j>csr->m)
				// printf("%ld \n", j);
        	csr->Mask.insert({i,col_ind[j]}, values[j]);
			// printf("%ld ", col_ind[j]);
		}
    }
	
	
	csr->Mask.pack();
	// const auto& values_1 = csr->Mask.getStorage().getValues();
	// const ValueType* data_1 = static_cast<const ValueType*>(values_1.getData());
	// long zeros=0;
	
	// for (long i = 0; i < csr->m; ++i) {
	// 	for (long j = 0; j < csr->m; ++j) {
	// 		if (data_1[zeros] != 0.0){
	// 			if (data_1[zeros] !=values[zeros]){
	// 				// printf("error %ld %ld %f %f\n", i, j, data_1[zeros], values[zeros]);
	// 			}
	// 			zeros++;
	// 		}
	// 	}

	// }
    // printf("ZEROS %ld %ld %ld\n", zeros, csr->nnz, row_ptr[csr->m]);
	csr->row_ptr=row_ptr;
	csr->col_ind=col_ind;
	return csr;

}


void
compute_csr(CSRTensors * csr, ValueType * y)
{
	Tensor<ValueType> A(csr->Mask.getDimensions(), Format({Sparse, Sparse}));
	IndexVar i, j, k;
	// ParallelSchedule *sched;
	// int *chunk_size;
	// taco_get_parallel_schedule(sched, chunk_size);
	int threads = atoi(getenv("cores"));
	taco_set_num_threads(threads);
	int threads_used = taco_get_num_threads();
	// printf("threads %d %d\n", threads_used, threads);
  	A(i,j) = csr->Mask(i,j) * csr->x(i,k) * csr->z(k,j);
	A.compile();
	A.assemble();
  	A.compute();
	std::cout << A.getStorage().getValues().getData() << std::endl;
	// printf("compute\n");
	// long nnz=0;
	// for (long m = 0; m < csr->m; ++i) {
	// 	for (long n = csr->row_ptr[m]; n < csr->row_ptr[m+1]; ++n) {
    //     	y[nnz]=*A(m,csr->col_ind[n]);
	// 		nnz++;
	// 	}
    // }
	const auto& storage = A.getStorage();
	const auto& values = A.getStorage().getValues();
	const ValueType* data = static_cast<const ValueType*>(values.getData());
	// const auto& values_1 = csr->z.getStorage().getValues();
	// const ValueType* data_1 = static_cast<const ValueType*>(values_1.getData());
	// long nnz=0;
	// #pragma omp parallel for
    // for (long j = 0; j < csr->nnz ; ++j) {
	// 	if (data[j] != 0.0){
	// 		y[j] =data[j]; 
	// 		// nnz++;
	// 		// std::cout << y[j] << std::endl;
	// 		// printf("%f ", data[j]);
	// 	}
	// 	// y[j] = static_cast<ValueType>(values[j].operator ValueType());
    // }
	// printf("nnz %ld %ld\n", nnz, csr->nnz);

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

