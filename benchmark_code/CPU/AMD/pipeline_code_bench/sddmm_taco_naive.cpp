#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
// #include "/various/itasou/taco/include/taco.h"
#include <time.h>
#include<iomanip>
#include <mkl.h>


#include <mkl_spblas.h>

#include "macros/cpp_defines.h"

#include "sddmm_bench_common.h"
#include "sddmm_kernel.h"

#define ITERS 1
// #define ValueType ValueType
//-----------COPIED FROM TACO----------------------------------------------
#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdint.h>
#include <math.h>
#include<iostream>
using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))


#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_dim_dense, taco_dim_sparse } taco_dim_t;
typedef struct {
    int     order;      // tensor order (number of dimensions)
    int*    dims;       // tensor dimensions
    taco_dim_t* dim_types;  // dimension storage types
    int     csize;      // component size
    int*    dim_order;  // dimension storage order
    int***  indices;    // tensor index data (per dimension)
    ValueType*    vals;       // tensor values
} taco_tensor_t;
#endif
#endif


void cacheFlush(ValueType *X, ValueType *Y) {
  
  for(int i=0; i<20*1000000; i++) {
    X[i]=Y[i]+rand() % 5;
    Y[i] += rand() % 7;

  }

}


int assemble(taco_tensor_t *O, taco_tensor_t *A, taco_tensor_t *D) {
    int O1_size = *(int*)(O->indices[0][0]);
    int O2_size = *(int*)(O->indices[1][0]);
    ValueType* __restrict O_vals = (ValueType*)(O->vals);
    
    O_vals = (ValueType*)malloc(sizeof(ValueType) * (O1_size * O2_size));
    
    O->vals = (ValueType*)O_vals;
    return 0;
}

int compute1(taco_tensor_t *O, taco_tensor_t *A, taco_tensor_t *D) {
    int O1_size = *(int*)(O->indices[0][0]);
    int O2_size = *(int*)(O->indices[1][0]);
    //cout<<"\n11\n";
    ValueType* __restrict O_vals = (ValueType*)(O->vals);
    int A1_size = *(int*)(A->indices[0][0]);
    //cout<<"\nA1_size "<<A1_size;
    int* __restrict A2_pos = (int*)(A->indices[1][0]);
    int* __restrict A2_idx = (int*)(A->indices[1][1]);
    ValueType* __restrict A_vals = (ValueType*)(A->vals);
    //cout<<"\n13\n";
    int D1_size = *(int*)(D->indices[0][0]);
    int D2_size = *(int*)(D->indices[1][0]);
    //cout<<"\n14\n";
	printf("%d %d %d %d %d ",A1_size, D1_size, D2_size, O1_size, O2_size);
    ValueType* __restrict D_vals = (ValueType*)(D->vals);
    //cout<<"\n15\n"<<"O1_size"<<O1_size<<"O2_size"<<O2_size<<endl;
    for (int pO = 0; pO < (O1_size * O2_size); pO++) {
        O_vals[pO] =  (ValueType)(rand()%1048576)/1048576;
    }
    //cout<<"\nbefore parallel for\n";
     
    return 0;
}


int compute2(taco_tensor_t *O, taco_tensor_t *A, ValueType *B, taco_tensor_t *D) {
    int O1_size = *(int*)(O->indices[0][0]);
    int O2_size = *(int*)(O->indices[1][0]);
    //cout<<"\n11\n";
    ValueType* __restrict O_vals = (ValueType*)(O->vals);
    int A1_size = *(int*)(A->indices[0][0]);
    //cout<<"\nA1_size "<<A1_size;
    int* __restrict A2_pos = (int*)(A->indices[1][0]);
    int* __restrict A2_idx = (int*)(A->indices[1][1]);
    ValueType* __restrict A_vals = (ValueType*)(A->vals);
    //cout<<"\n13\n";
    int D1_size = *(int*)(D->indices[0][0]);
    int D2_size = *(int*)(D->indices[1][0]);
    //cout<<"\n14\n";
	printf("%d %d %d %d %d ",A1_size, D1_size, D2_size, O1_size, O2_size);
    ValueType* __restrict D_vals = (ValueType*)(D->vals);
	// printf("%d %d %d %d %d ",A1_size, D1_size, D2_size, O1_size, O2_size);  
    //cout<<"\nbefore parallel for\n";
    #pragma omp parallel for
    for (int mA = 0; mA < A1_size; mA++) {
        
        for (int pA2 = A2_pos[mA]; pA2 < A2_pos[mA + 1]; pA2++) {
            int kA = A2_idx[pA2]; 
            for (int nD = 0; nD < D2_size; nD++) {
                int pD2 = (kA * D2_size) + nD;
				// int pD2 = (nD * D2_size) + kA;
                int pO2 = (mA * O2_size) + nD;

                B[pA2] += O_vals[pO2] * D_vals[pO2]; 
				// if (pD2> D2_size*D1_size)
				        // printf("%d: %f %f \n",nD, O_vals[pO2], D_vals[pO2]);     
                
            }
		
		
		B[pA2] *= A_vals[pA2];
        }
    }
    
    return 0;
}
//-----------------------------------------------------------------------------------------------------------
struct CSRTensors : Matrix_Format
{

    taco_tensor_t *Mask;
	taco_tensor_t *K;
	taco_tensor_t *Q;
	INT_T * row_ptr;
	INT_T * col_ind;

	CSRTensors(long m, long n, long nnz) : Matrix_Format(m, n, nnz)
	{
		Mask = new taco_tensor_t;
		Mask->indices = new int** [2];
		Mask->indices[0] = new int*[2];
		Mask->indices[1] = new int*[2];
		Mask->indices[0][0] = (int*)new int;
    
     	*(int*)(Mask->indices[0][0]) = m ;
		K = new taco_tensor_t;
		K->indices = new int** [2];
        K->indices[0] = new int*;
        K->indices[1] = new int*;
        K->indices[0][0] = (int*)new int;
        K->indices[1][0] = (int*)new int;
        *(int*)(K->indices[0][0]) = m ;
        *(int*)(K->indices[1][0]) = n ;
		Q = new taco_tensor_t;
		Q->indices = new int** [2];
        Q->indices[0] = new int*;
        Q->indices[1] = new int*;
        Q->indices[0][0] = (int*)new int;
        Q->indices[1][0] = (int*)new int;
        *(int*)(Q->indices[0][0]) = m ;
        *(int*)(Q->indices[1][0]) = n ;
	}

	~CSRTensors()
	{
	}

    void spmm(char type, INT_T m, INT_T k, INT_T n, INT_T *ia, INT_T *ja, ValueType *a, ValueType *x, ValueType *y) ;
	void sddmm(ValueType * y);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};


void compute_csr(CSRTensors * csr, ValueType * y);


void
CSRTensors::sddmm(ValueType * y)
{
	compute_csr(this, y);
    // printf("SDDMM\n");
}

void
CSRTensors::spmm(char type, INT_T m, INT_T k, INT_T n, INT_T *ia, INT_T *ja, ValueType *a, ValueType *x, ValueType *y)
{
    char transa = 'N';
	ValueType alpha = 1.0, beta = 0.0;
	char matdescra[6];
	matdescra[0] = 'G';
    matdescra[1] = 'L';
    matdescra[2] = 'N';
    matdescra[3] = 'C';

	#if DOUBLE == 0
		mkl_scsrmm(&transa, &m, &n, &k, &alpha, matdescra, a, ja, ia,  &(ia[1]), &(x[0]), &n,  &beta, &(y[0]), &n);
	#elif DOUBLE == 1
		mkl_dcsrmm(&transa, &m, &n, &k, &alpha, matdescra, a, ja, ia,  &(ia[1]), &(x[0]), &n,  &beta, &(y[0]), &n);
	#endif
    // printf("SPMM\n");

    if (type=='K'){
	    this->K->vals = (ValueType*)y;
        printf("k\n");
    }
    else if (type=='Q'){
	    this->Q->vals = (ValueType*)y;
        printf("q\n");
    }
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long nnz , long n)
{
	
	struct CSRTensors * csr = new CSRTensors(m, n, nnz);
	csr->format_name = (char *) "TACO_CSR";
	csr->Mask->indices[1][0] = (int*)row_ptr;
	csr->Mask->indices[1][1] = (int*)col_ind; 
	csr->Mask->vals = (ValueType*)values;

	csr->row_ptr=row_ptr;
	csr->col_ind=col_ind;
	return csr;

}


void
compute_csr(CSRTensors * csr, ValueType * y)
{
	// compute1(csr->x, csr->Mask, csr->z);
	// printf("hi\n");
	for (int i=0; i<csr->nnz; i++){
		y[i]=0.0;
	}
	compute2(csr->Q, csr->Mask, y, csr->K);
	
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

