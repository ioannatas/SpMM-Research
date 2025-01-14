#ifndef SDDMM_MASK_H
#define SDDMM_MASK_H

#include "macros/cpp_defines.h"

#include "sddmm_bench_common.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <time.h>

INT_T * band_and_random(long length, INT_T & nnz, long band_size, double sparsity) {

    long total_elements = (length * (length + 1)) / 2; 
    long zero_elements = (long)(sparsity * total_elements); 
    long nonzero_elements = total_elements - zero_elements; 
    INT_T *mask;
    mask= (typeof(mask)) aligned_alloc(64, length*length * sizeof(*mask));

    // Initialize all elements to zero
    #pragma omp parallel for
    for (long i = 0; i < length; i++) {
        for (long j = 0; j < length; j++) {
            mask[i*length+j] = 0;
        }
    }
    // printf("INITIALIZE\n");
    // Create the dense diagonal band
    long band_zeros=0;
    for (long i = 0; i < length; ++i) {
        for (long j = std::max((long)0, i - band_size + 1); j <= std::min(length - 1, i + band_size - 1); ++j) {
            mask[i*length+j] = 1.0;
            band_zeros++;
        }
    }
    // printf("here %ld %ld\n",length, band_zeros);

    // Randomly place non-zero values in the lower triangular part
    long placed_nonzeros = band_zeros;
    long counter=0;
    long period=10;
    while (placed_nonzeros < nonzero_elements) {
        // printf("%d", counter);
        if (counter % period == 0)            // Periodic reseeding.
				srand(time(NULL)+counter);
        long row = rand() % length;
        long col = rand() % (row + 1); // Ensure col <= row for lower triangular part
        if (mask[row*length+col] == 0) {
            mask[row*length+col] = 1; 
            placed_nonzeros++;
        }
        counter++;
    }
    nnz=placed_nonzeros;
     if (nnz!=nonzero_elements)
        printf("error creating mask: %d %d", nnz, nonzero_elements);
    return mask;
}


INT_T *band_and_decay(long length, INT_T &nnz, long band_size, double sparsity) {
    long total_elements = (length * (length + 1)) / 2; 
    long zero_elements = (long)(sparsity * total_elements); 
    long nonzero_elements = total_elements - zero_elements; 

    // Allocate memory for the mask
    INT_T *mask;
    mask = (typeof(mask))aligned_alloc(64, length * length * sizeof(*mask));

    // Initialize all elements to zero
    #pragma omp parallel for
    for (long i = 0; i < length; i++) {
        for (long j = 0; j < length; j++) {
            mask[i * length + j] = 0;
        }
    }

    // Create the dense diagonal band
    long band_zeros = 0;
    for (long i = 0; i < length; ++i) {
        for (long j = fmax(0, i - band_size + 1); j <= fmin(length - 1, i + band_size - 1); ++j) {
            mask[i * length + j] = 1;
            band_zeros++;
        }
    }

    double *weights = (double *)malloc(total_elements * sizeof(double));


    long idx = 0;
    for (long i = 0; i < length; ++i) {
        for (long j = 0; j <= i; ++j) { 
            if (j < i - band_size || j > i + band_size) { 
                double distance = i - j;
                weights[idx] = 1.0 / (distance + 1e-5);
            } else {
                weights[idx] = 0.0;
            }
            idx++;
        }
    }

    // Normalize the weights
    double weight_sum = 0.0;
    for (long i = 0; i < total_elements; ++i) {
        weight_sum += weights[i];
    }
    for (long i = 0; i < total_elements; ++i) {
        weights[i] /= weight_sum;
    }

    // Place random non-zero values based on weights
    long placed_nonzeros = band_zeros;
    long counter = 0;
    while (placed_nonzeros < nonzero_elements) {
        if (counter % 10 == 0) { // Periodic reseeding
            srand(time(NULL) + counter);
        }

        double r = (double)rand() / RAND_MAX;
        double cumulative = 0.0;
        long selected_index = -1;

        for (long i = 0; i < total_elements; ++i) {
            cumulative += weights[i];
            if (r <= cumulative) {
                selected_index = i;
                break;
            }
        }

        if (selected_index >= 0) {
            long row = (long)sqrt(2 * selected_index); 
            long col = selected_index - (row * (row + 1)) / 2;
            if (mask[row * length + col] == 0) {
                mask[row * length + col] = 1;
                placed_nonzeros++;
            }
        }
        counter++;
    }

    nnz = placed_nonzeros;

    // Free weights array
    free(weights);

    if (nnz != nonzero_elements) {
        printf("Error creating mask: %d %d\n", nnz, nonzero_elements);
    }

    return mask;
}

INT_T * generate_sparse_attention_mask(long sequence_length, INT_T & nnz, char * sparse_attention_type, long band_size, double sparsity) {
    srand(time(NULL));
    // printf("%s %d %d\n",sparse_attention_type, strcmp(sparse_attention_type,"band_and_decay"), strcmp(sparse_attention_type,"band_and_random"));
    if (strcmp(sparse_attention_type,"band_and_decay") ==0 ) {
        // printf("hi1\n");
        return band_and_decay(sequence_length, nnz, band_size, sparsity);
    } else if (strcmp(sparse_attention_type,"band_and_random") ==0) {
        // printf("hi2\n");
        return band_and_random(sequence_length,nnz, band_size, sparsity);
    }
}

struct Mask
{
	char * sparse_attention_type;
	INT_T m;                         // num rows
	INT_T nnz;                       // num non-zeros
	double sparsity;
    long band_size;
	double csr_mem_footprint;
    INT_T *csr_ja;
    INT_T *csr_ia;
    ValueType *csr_a;


	Mask(char * sparse_attention_type, INT_T m, double sparsity, long band_size) :sparse_attention_type(sparse_attention_type), m(m), sparsity(sparsity), band_size(band_size)
	{
		csr_mem_footprint = (1-sparsity)* m * m * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	}

    ~Mask()
	{
        free(csr_ia);
        free(csr_ja);
        free(csr_a);
	}

};

void dense_to_csr(int * mask, struct Mask * Mask){
    INT_T nnz=0;
    for (long i = 0; i < Mask->m; ++i) {
        Mask->csr_ia[i]=nnz;
        for (long j = 0; j <Mask->m; ++j) {
            // printf(" %lf", mask[i][j]);
            if (mask[i*Mask->m+j]!=0.0){
                // printf("mp");
                Mask->csr_ja[nnz]=j;
                Mask->csr_a[nnz]=mask[i*Mask->m+j];
                nnz++;
            }
        }
    }
    Mask->csr_ia[Mask->m]=nnz;
    // for (long i=0; i<Mask->m; i++ ){
    //     printf("%ld ", Mask->csr_ia[i]);
    // }
    // printf("%ld %ld ",Mask->m,Mask->nnz);
    free(mask);
    if (nnz!=Mask->nnz)
        printf("error creating mask: %d %d", nnz, Mask->nnz);
};

struct Mask * create_mask(char * sparse_attention_type, INT_T m, double sparsity, long band_size){
    INT_T * mask_v;
    INT_T nnz;
    mask_v = generate_sparse_attention_mask(m,nnz, sparse_attention_type, band_size, sparsity);
    struct Mask * mask =new Mask(sparse_attention_type, m, sparsity, band_size);
    mask->nnz=nnz;
    mask->csr_ja = (typeof(mask->csr_ja)) aligned_alloc(64, mask->nnz * sizeof(*mask->csr_ja));
    mask->csr_ia = (typeof(mask->csr_ia)) aligned_alloc(64, (mask->m+1) * sizeof(*mask->csr_ia));
    mask->csr_a = (typeof(mask->csr_a)) aligned_alloc(64, (mask->nnz) * sizeof(*mask->csr_a));
    dense_to_csr( mask_v, mask);
    // printf("%ld %ld ",mask->m,mask->nnz);
    return mask;
};



#endif /* SDDMM_MASK_H */