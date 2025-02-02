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
#include <math.h>

INT_T * band_and_random(char * sddmm_sparsification_type, long length, INT_T & nnz, long &band_size, double sparsity, double & l_sparsity) {

    long total_elements = length * length; 
    long band_values;
    long C;
    long b;
    if (strcmp(sddmm_sparsification_type, "l_sparsity") == 0) {
        C = 1 / 2 - ((sparsity - 0.5) / l_sparsity);
        b = 2 * length - 1;
        band_size = (long)((-b + sqrt(b * b + 8 * total_elements * C)) / 2);
    } else if (strcmp(sddmm_sparsification_type, "band_size") == 0) {
        // printf("band_size \n");
        if (sparsity == 0.95){
            band_size = 16; //can do and 24 barely
        }else if(sparsity== 0.98){
            band_size=8;
        }else if (sparsity==0.5){
            band_size=length-100;
        }
        // band_values= band_size * length - (band_size * (band_size - 1)) / 2;
        band_values = (band_size/2) * (2 * length + band_size - 1);
        l_sparsity = ((sparsity - 0.5) * total_elements) / (total_elements/2 - band_values);
        // printf("band_size %lf\n", sparsity-0.5);
    }
    // long zero_elements = (long)(sparsity * total_elements); 
    long nonzero_elements = sparsity*total_elements; 
    INT_T *mask;
    mask= (typeof(mask)) aligned_alloc(64, total_elements * sizeof(*mask));

    #pragma omp parallel for
    for (long i = 0; i < length; i++) {
        for (long j = 0; j < length; j++) {
            mask[i*length+j] = 0;
        }
    }

    long band_zeros=0;
    for (long i = 0; i < length; ++i) {
        for (long j = std::max((long)0, i - band_size + 1); j <= std::min(length - 1, i + band_size - 1); ++j) {
            mask[i*length+j] = 1.0;
            band_zeros++;
        }
    }
    // printf("here %ld %ld\n",length, band_zeros);

    long placed_nonzeros = band_zeros;
    long counter=0;
    long period=10;
    while (placed_nonzeros < nonzero_elements) {
        // printf("%d", counter);
        if (counter % period == 0)            // Periodic reseeding.
				srand(time(NULL)+counter);
        long row = rand() % length;
        long col = rand() % (row + 1); 
        if (mask[row*length+col] == 0) {
            mask[row*length+col] = 1; 
            placed_nonzeros++;
        }
        counter++;
    }
    nnz=placed_nonzeros;
     if (nnz!=nonzero_elements)
        printf("Error creating mask: placed_nonzeros%d nonzero_elements%d sparsity:%f l_sparsity:%f band_size:%d\n", nnz, nonzero_elements, sparsity, l_sparsity, band_size);
    return mask;
}


INT_T *band_and_decay(char *sddmm_sparsification_type, long length, INT_T &nnz, long &band_size, double sparsity, double &l_sparsity) {
    long total_elements = length * length;
    long band_values;
    long C;
    long b;

    if (strcmp(sddmm_sparsification_type, "l_sparsity") == 0) {
        C = 1 / 2 - ((sparsity - 0.5) / l_sparsity);
        b = 2 * length - 1;
        band_size = (long)((-b + sqrt(b * b + 8 * total_elements * C)) / 2);
    } else if (strcmp(sddmm_sparsification_type, "band_size") == 0) {
        printf("band_size \n");
        if (sparsity == 0.95){
            band_size = 16; //can do and 24 barely
        }else if(sparsity== 0.98){
            band_size=8;
        }else if (sparsity==0.5){
            band_size=length-100;
        }
        // band_values= band_size * length - (band_size * (band_size - 1)) / 2;
        band_values = (band_size/2) * (2 * length + band_size - 1);
        l_sparsity = ((sparsity - 0.5) * total_elements) / (total_elements/2 - band_values);
        printf("band_size %lf\n", sparsity-0.5);
    }

    long nonzero_elements = sparsity * total_elements;
    INT_T *mask = (INT_T *)aligned_alloc(64, total_elements * sizeof(INT_T));

    // Initialize all elements to zero in parallel
    #pragma omp parallel for
    for (long i = 0; i < total_elements; i++) {
        mask[i] = 0;
    }

    // Create the dense diagonal band
    long band_zeros = 0;
    #pragma omp parallel for reduction(+:band_zeros)
    for (long i = 0; i < length; ++i) {
        for (long j = std::max((long)0, i - band_size + 1); j <= i; ++j) {
            mask[i * length + j] = 1;
            band_zeros++;
        }
    }

    double *weights = (double *)malloc(total_elements * sizeof(double));

    // Calculate weights in parallel
    #pragma omp parallel for
    for (long i = 0; i < length; ++i) {
        for (long j = 0; j <= i; ++j) {
            long idx = i * length + j;
            if (j < i - band_size || j > i + band_size) {
                double distance = i - j;
                weights[idx] = 1.0 / (distance + 1e-5);
            } else {
                weights[idx] = 0.0;
            }
        }
    }

    // Normalize the weights in parallel
    double weight_sum = 0.0;
    #pragma omp parallel for reduction(+:weight_sum)
    for (long i = 0; i < total_elements; ++i) {
        weight_sum += weights[i];
    }

    #pragma omp parallel for
    for (long i = 0; i < total_elements; ++i) {
        weights[i] /= weight_sum;
    }

    // Precompute cumulative weights
    // double *cumulative_weights = (double *)malloc(total_elements * sizeof(double));
    // cumulative_weights[0] = weights[0];
    // for (long i = 1; i < total_elements; ++i) {
    //     cumulative_weights[i] = cumulative_weights[i - 1] + weights[i];
    // }

    // Place random non-zero values based on weights
    long placed_nonzeros = band_zeros;
    // srand(time(NULL)); // Seed the random number generator once
    // printf("placing\n");
    // int counter = 0;
    // while (placed_nonzeros < nonzero_elements) {
    //     // if (counter % 10 == 0) { // Periodic reseeding
    //     //     srand(time(NULL) + counter);
    //     // }
        
    //     double r = (double)rand() / RAND_MAX;
    //     long selected_index = std::lower_bound(cumulative_weights, cumulative_weights + total_elements, r) - cumulative_weights;
    //     printf("%d %d %d\n", selected_index, placed_nonzeros, nonzero_elements);
    //     if (selected_index < total_elements && mask[selected_index] == 0) {
    //         mask[selected_index] = 1;
    //         placed_nonzeros++;
    //     }
    //     counter++;
    // }
         // Precompute candidate indices (where mask[index] == 0)
    std::vector<long> candidate_indices;
    std::vector<double> candidate_weights;
    for (long i = 0; i < total_elements; ++i) {
        if (mask[i] == 0) {
            candidate_indices.push_back(i);
            candidate_weights.push_back(weights[i]); // Weights are already normalized
        }
    }

    // Compute cumulative weights
    std::vector<double> cumulative_weights(candidate_weights.size());
    cumulative_weights[0] = candidate_weights[0];
    for (size_t i = 1; i < candidate_weights.size(); ++i) {
        cumulative_weights[i] = cumulative_weights[i - 1] + candidate_weights[i];
    }

    // Use a faster random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Place random non-zero values based on weights
    long remaining_nonzeros = nonzero_elements - placed_nonzeros;
    while (placed_nonzeros < nonzero_elements) {
        double r = dis(gen);
        auto it = std::lower_bound(cumulative_weights.begin(), cumulative_weights.end(), r);
        long selected_index = candidate_indices[std::distance(cumulative_weights.begin(), it)];
//         if (selected_index < 0 || selected_index >= total_elements) {
//     printf("Error: selected_index out of bounds: %ld\n", selected_index);
//     continue; // Skip this iteration and try again
// }
        // printf("%ld %ld %ld\n", selected_index, placed_nonzeros, nonzero_elements);
        if (mask[selected_index] == 0) {
            mask[selected_index] = 1;
            placed_nonzeros++;
        }
    }
    nnz = placed_nonzeros;

    // Free weights and cumulative_weights arrays
    free(weights);
    // free(cumulative_weights);

    if (nnz != nonzero_elements) {
        printf("Error creating mask: placed_nonzeros %d nonzero_elements %d sparsity: %f l_sparsity: %f band_size: %d\n", nnz, nonzero_elements, sparsity, l_sparsity, band_size);
    }

    return mask;
}
INT_T * generate_sparse_attention_mask(long sequence_length, INT_T & nnz, char * sparse_attention_type, char * sddmm_sparsification_type, long &band_size, double sparsity, double &l_sparsity) {
    srand(time(NULL));
    // printf("%s %d %d\n",sparse_attention_type, strcmp(sparse_attention_type,"band_and_decay"), strcmp(sparse_attention_type,"band_and_random"));
    if (strcmp(sparse_attention_type,"band_and_decay") ==0 ) {
        // printf("hi1\n");
        return band_and_decay(sddmm_sparsification_type, sequence_length, nnz, band_size, sparsity, l_sparsity);
    } else if (strcmp(sparse_attention_type,"band_and_random") ==0) {
        // printf("hi2\n");
        return band_and_random(sddmm_sparsification_type, sequence_length,nnz, band_size, sparsity, l_sparsity);
    }
}

struct Mask
{
	char * sparse_attention_type;
    char * sddmm_sparsification_type;
	INT_T m;                         // num rows
	INT_T nnz;                       // num non-zeros
	double sparsity;
    double l_sparsity;
    long band_size;
	double csr_mem_footprint;
    INT_T *csr_ja;
    INT_T *csr_ia;
    ValueType *csr_a;


	Mask(char * sparse_attention_type, char * sddmm_sparsification_type, INT_T m, double sparsity, double l_sparsity, long band_size) :sparse_attention_type(sparse_attention_type),sddmm_sparsification_type(sddmm_sparsification_type), m(m), sparsity(sparsity), l_sparsity(l_sparsity), band_size(band_size)
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

struct Mask * create_mask(char * sparse_attention_type, char *sddmm_sparsification_type, INT_T m, double sparsity, double l_sparsity, long band_size){
    INT_T * mask_v;
    INT_T nnz;
    mask_v = generate_sparse_attention_mask(m,nnz, sparse_attention_type, sddmm_sparsification_type, band_size, sparsity, l_sparsity);
    struct Mask * mask =new Mask(sparse_attention_type,sddmm_sparsification_type, m, sparsity, l_sparsity, band_size);
    mask->nnz=nnz;
    mask->csr_ja = (typeof(mask->csr_ja)) aligned_alloc(64, mask->nnz * sizeof(*mask->csr_ja));
    mask->csr_ia = (typeof(mask->csr_ia)) aligned_alloc(64, (mask->m+1) * sizeof(*mask->csr_ia));
    mask->csr_a = (typeof(mask->csr_a)) aligned_alloc(64, (mask->nnz) * sizeof(*mask->csr_a));
    dense_to_csr( mask_v, mask);
    // printf("%ld %ld ",mask->m,mask->nnz);
    return mask;
};



#endif /* SDDMM_MASK_H */