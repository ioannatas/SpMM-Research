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
#include <algorithm> // for std::sort
#include <tuple>     // for std::tuple

// COO format structure
struct COOMask {
    INT_T *row_indices;
    INT_T *col_indices;
    ValueType *values;
    INT_T nnz;
};

// Comparator to sort COO format by row and then by column
bool coo_comparator(const std::tuple<INT_T, INT_T, ValueType> &a, const std::tuple<INT_T, INT_T, ValueType> &b) {
    if (std::get<0>(a) != std::get<0>(b))
        return std::get<0>(a) < std::get<0>(b); // Sort by row
    return std::get<1>(a) < std::get<1>(b);     // Sort by column within the same row
}

// Function to sort COO format
void sort_coo(COOMask *coo_mask) {
    // Create a vector of tuples for sorting
    std::vector<std::tuple<INT_T, INT_T, ValueType>> coo_data;
    for (INT_T i = 0; i < coo_mask->nnz; ++i) {
        coo_data.emplace_back(coo_mask->row_indices[i], coo_mask->col_indices[i], coo_mask->values[i]);
    }

    // Sort the vector
    std::sort(coo_data.begin(), coo_data.end(), coo_comparator);

    // Write the sorted data back to the COO arrays
    for (INT_T i = 0; i < coo_mask->nnz; ++i) {
        coo_mask->row_indices[i] = std::get<0>(coo_data[i]);
        coo_mask->col_indices[i] = std::get<1>(coo_data[i]);
        coo_mask->values[i] = std::get<2>(coo_data[i]);
    }
}

// COO to CSR conversion function
void coo_to_csr(const COOMask *coo_mask, INT_T *csr_ia, INT_T *csr_ja, ValueType *csr_a, INT_T num_rows) {
    // Initialize CSR row pointers
    std::fill(csr_ia, csr_ia + num_rows + 1, 0);

    // Count the number of non-zero elements in each row
    for (INT_T i = 0; i < coo_mask->nnz; ++i) {
        csr_ia[coo_mask->row_indices[i] + 1]++;
    }

    // Compute the prefix sum to get the row pointers
    for (INT_T i = 0; i < num_rows; ++i) {
        csr_ia[i + 1] += csr_ia[i];
    }

    // Fill the column indices and values
    for (INT_T i = 0; i < coo_mask->nnz; ++i) {
        INT_T row = coo_mask->row_indices[i];
        INT_T dest = csr_ia[row];
        csr_ja[dest] = coo_mask->col_indices[i];
        csr_a[dest] = coo_mask->values[i];
        csr_ia[row]++;
    }

    // Shift the row pointers back
    for (INT_T i = num_rows; i > 0; --i) {
        csr_ia[i] = csr_ia[i - 1];
    }
    csr_ia[0] = 0;
}

// Function to generate a banded and random mask in COO format
COOMask *band_and_random(char *sddmm_sparsification_type, long length, INT_T &nnz, long &band_size, double sparsity, double &l_sparsity) {
    long total_elements = length * length;
    long band_values;
    long C;
    long b;

    if (strcmp(sddmm_sparsification_type, "l_sparsity") == 0) {
        C = 1 / 2 - ((sparsity - 0.5) / l_sparsity);
        b = 2 * length - 1;
        band_size = (long)((-b + sqrt(b * b + 8 * total_elements * C)) / 2);
    } else if (strcmp(sddmm_sparsification_type, "band_size") == 0) {
        if (sparsity == 0.95) {
            band_size = 16;
        } else if (sparsity == 0.98) {
            band_size = 8;
        } else if (sparsity == 0.5) {
            band_size = length - 100;
        }
        band_values = (band_size / 2) * (2 * length + band_size - 1);
        l_sparsity = ((sparsity - 0.5) * total_elements) / (total_elements / 2 - band_values);
    }

    long nonzero_elements = sparsity * total_elements;
    COOMask *coo_mask = new COOMask;
    coo_mask->row_indices = (INT_T *)aligned_alloc(64, nonzero_elements * sizeof(INT_T));
    coo_mask->col_indices = (INT_T *)aligned_alloc(64, nonzero_elements * sizeof(INT_T));
    coo_mask->values = (ValueType *)aligned_alloc(64, nonzero_elements * sizeof(ValueType));
    coo_mask->nnz = 0;
    printf("coo allocation %ld\n", length);
    // Create the dense diagonal band
    long band_zeros = 0;
    for (long i = 0; i < length; ++i) {
        for (long j = std::max((long)0, i - band_size + 1); j <=i; ++j) {
            coo_mask->row_indices[coo_mask->nnz] = i;
            coo_mask->col_indices[coo_mask->nnz] = j;
            coo_mask->values[coo_mask->nnz] = 1.0;
            coo_mask->nnz++;
            band_zeros++;
        }
    }

    // Place random non-zero values
    long placed_nonzeros = band_zeros;
    long counter = 0;
    long period = 10;
    while (placed_nonzeros < nonzero_elements) {
        if (counter % period == 0)
            srand(time(NULL) + counter);
        
        long row = rand() % length;
        long col = rand() % (row + 1);
        bool exists = false;
        for (long k = 0; k < coo_mask->nnz; ++k) {
            if (coo_mask->row_indices[k] == row && coo_mask->col_indices[k] == col) {
                
                exists = true;
                printf("row %ld col %ld bool %d\n", row, col, exists);
                break;
            }
        }
        if (!exists) {
            coo_mask->row_indices[coo_mask->nnz] = row;
            coo_mask->col_indices[coo_mask->nnz] = col;
            coo_mask->values[coo_mask->nnz] = 1.0;
            coo_mask->nnz++;
            placed_nonzeros++;
        }
        counter++;
        exists = false;
    }
    printf("placed_nonzeros %d nonzero_elements %d sparsity: %f l_sparsity: %f band_size: %d\n", placed_nonzeros, nonzero_elements, sparsity, l_sparsity, band_size);
    nnz = placed_nonzeros;
    if (nnz != nonzero_elements)
        printf("Error creating mask: placed_nonzeros %d nonzero_elements %d sparsity: %f l_sparsity: %f band_size: %d\n", nnz, nonzero_elements, sparsity, l_sparsity, band_size);

    return coo_mask;
}

// Function to generate a sparse attention mask in COO format
COOMask *generate_sparse_attention_mask(long sequence_length, INT_T &nnz, char *sparse_attention_type, char *sddmm_sparsification_type, long &band_size, double sparsity, double &l_sparsity) {
    srand(time(NULL));
    if (strcmp(sparse_attention_type, "band_and_decay") == 0) {
        return band_and_random(sddmm_sparsification_type, sequence_length, nnz, band_size, sparsity, l_sparsity);
    } else if (strcmp(sparse_attention_type, "band_and_random") == 0) {
        return band_and_random(sddmm_sparsification_type, sequence_length, nnz, band_size, sparsity, l_sparsity);
    }
    return nullptr;
}

// Mask structure
struct Mask {
    char *sparse_attention_type;
    char *sddmm_sparsification_type;
    INT_T m;                         // num rows
    INT_T nnz;                       // num non-zeros
    double sparsity;
    double l_sparsity;
    long band_size;
    double csr_mem_footprint;
    INT_T *csr_ja;
    INT_T *csr_ia;
    ValueType *csr_a;

    Mask(char *sparse_attention_type, char *sddmm_sparsification_type, INT_T m, double sparsity, double l_sparsity, long band_size)
        : sparse_attention_type(sparse_attention_type), sddmm_sparsification_type(sddmm_sparsification_type), m(m), sparsity(sparsity), l_sparsity(l_sparsity), band_size(band_size) {
        csr_mem_footprint = (1 - sparsity) * m * m * (sizeof(ValueType) + sizeof(INT_T)) + (m + 1) * sizeof(INT_T);
    }

    ~Mask() {
        free(csr_ia);
        free(csr_ja);
        free(csr_a);
    }
};

// Function to create a mask in CSR format
struct Mask *create_mask(char *sparse_attention_type, char *sddmm_sparsification_type, INT_T m, double sparsity, double l_sparsity, long band_size) {
    INT_T nnz;
    COOMask *coo_mask = generate_sparse_attention_mask(m, nnz, sparse_attention_type, sddmm_sparsification_type, band_size, sparsity, l_sparsity);

    // Sort the COO format
    sort_coo(coo_mask);

    // Create the Mask struct
    struct Mask *mask = new Mask(sparse_attention_type, sddmm_sparsification_type, m, sparsity, l_sparsity, band_size);
    mask->nnz = nnz;

    // Allocate memory for CSR format
    mask->csr_ia = (INT_T *)aligned_alloc(64, (m + 1) * sizeof(INT_T));
    mask->csr_ja = (INT_T *)aligned_alloc(64, nnz * sizeof(INT_T));
    mask->csr_a = (ValueType *)aligned_alloc(64, nnz * sizeof(ValueType));

    // Convert COO to CSR
    coo_to_csr(coo_mask, mask->csr_ia, mask->csr_ja, mask->csr_a, m);

    // Free the temporary COOMask
    free(coo_mask->row_indices);
    free(coo_mask->col_indices);
    free(coo_mask->values);
    delete coo_mask;

    return mask;
}

#endif /* SDDMM_MASK_H */