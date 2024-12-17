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
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long nnz , long n, ValueType *x, ValueType *z)
{
	
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
	long line=0;
	for (long i = 0; i < csr->m; ++i) {
		for (long j = row_ptr[i]; i < row_ptr[i+1]; ++i) {
        	csr->Mask.insert({line,col_ind[j]}, values[j]);
		}
		line++;
    }
    csr->Mask.pack();
	csr->row_ptr=row_ptr;
	csr->col_ind=col_ind;
	return csr;

}


void
compute_csr(CSRTensors * csr, ValueType * y)
{
	Tensor<ValueType> A(csr->Mask.getDimensions(), Format({Sparse, Sparse}));
	IndexVar i, j, k;
  	A(i,j) = csr->Mask(i,j) * csr->x(i,k) * csr->z(k,j);
	A.compile();
	A.assemble();
  	A.compute();
	// long nnz=0;
	// for (long m = 0; m < csr->m; ++i) {
	// 	for (long n = csr->row_ptr[m]; n < csr->row_ptr[m+1]; ++n) {
    //     	y[nnz]=*A(m,csr->col_ind[n]);
	// 		nnz++;
	// 	}
    // }
	// const auto& storage = A.getStorage();
	// const auto& values = A.getStorage().getValues();
    // for (long j = 0; j < csr->nnz ; ++j) {
    //     y[j] =*(values[j]); // Extract values directly from the tensor storage
	// 	// y[j] = static_cast<ValueType>(values[j].operator ValueType());
    // }

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

