#ifndef SDDMM_MASK_H
#define SDDMM_MASK_H

#include "macros/cpp_defines.h"

#include "sddmm_bench_common.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// Function to initialize a 2D matrix with zeros
std::vector<std::vector<double>> create_matrix(int rows, int cols) {
    return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
}

// Generate a band and random sparsity mask
std::vector<std::vector<double>> band_and_random(int length, INT_T & nnz, int band_size, double sparsity) {
    auto mask = create_matrix(length, length);

    // Create the dense diagonal band
    for (int i = 0; i < length; ++i) {
        for (int j = std::max(0, i - band_size + 1); j <= std::min(length - 1, i + band_size - 1); ++j) {
            mask[i][j] = 1.0;
        }
    }

    // Calculate number of random sparse entries
    int x = length - band_size;
    int random_triangle_size = (x * (x + 1)) / 2;
    int nonzero = static_cast<int>(std::round(random_triangle_size * (1.0 - sparsity)));
    nnz=nonzero;

    // Generate random indices for the sparse off-diagonal elements
    std::vector<int> indices(random_triangle_size);
    for (int i = 0; i < random_triangle_size; ++i) indices[i] = i;

    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

    // Map random triangle indices to matrix positions
    for (int idx = 0; idx < nonzero; ++idx) {
        int tri_idx = indices[idx];
        int row = tri_idx / x; // Row index in the triangle
        int col = tri_idx % x; // Column index in the triangle
        if (row + band_size < length && col < row) {
            mask[row + band_size][col] = 1.0;
        }
    }

    return mask;
}

// Generate a band and decaying sparsity mask
std::vector<std::vector<double>> band_and_decay(int length, INT_T & nnz, int band_size, double sparsity) {
    auto mask = create_matrix(length, length);

    // Calculate weights based on distance from diagonal
    std::vector<double> weights;
    std::vector<std::pair<int, int>> indices;

    for (int i = band_size; i < length; ++i) {
        for (int j = 0; j < i - band_size; ++j) {
            double weight = 1.0 / (i - j + 1e-5);
            weights.push_back(weight);
            indices.emplace_back(i, j);
        }
    }

    // Normalize weights
    double weight_sum = 0.0;
    for (double w : weights) weight_sum += w;
    for (double& w : weights) w /= weight_sum;

    // Sample indices based on weights
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    std::mt19937 gen(std::random_device{}());
    int nonzero = static_cast<int>(std::round(weights.size() * (1.0 - sparsity)));
    nnz=nonzero;

    for (int n = 0; n < nonzero; ++n) {
        int idx = dist(gen);
        auto [row, col] = indices[idx];
        mask[row][col] = 1.0;
    }

    // Add the dense diagonal band
    for (int i = 0; i < length; ++i) {
        for (int j = std::max(0, i - band_size + 1); j <= std::min(length - 1, i + band_size - 1); ++j) {
            mask[i][j] = 1.0;
        }
    }

    return mask;
}

std::vector<std::vector<double>> generate_sparse_attention_mask(int sequence_length, INT_T & nnz, char * sparse_attention_type, int band_size, double sparsity) {
    if (sparse_attention_type == "band_and_decay") {
        return band_and_decay(sequence_length, nnz, band_size, sparsity);
    } else if (sparse_attention_type == "band_and_random") {
        return band_and_random(sequence_length,nnz, band_size, sparsity);
    }
}

struct Mask
{
	char * sparse_attention_type;
	INT_T m;                         // num rows
	INT_T nnz;                       // num non-zeros
	double sparsity;
    int band_size;
	double csr_mem_footprint;
    INT_T *csr_ja;
    INT_T *csr_ia;
    ValueType *csr_a;


	Mask(char * sparse_attention_type, INT_T m, double sparsity, int band_size) :sparse_attention_type(sparse_attention_type), m(m), sparsity(sparsity), band_size(band_size)
	{
		csr_mem_footprint = (1-sparsity)* m * m * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	}

    ~Mask()
	{
	}

};

void dense_to_csr(std::vector<std::vector<double>> mask, struct Mask * Mask){
    INT_T nnz=0;
    for (int i = 0; i < Mask->m; ++i) {
        Mask->csr_ia[i]=nnz;
        for (int j = 0; j <Mask->m; ++j) {
            if (mask[i][j]!=0.0){
                Mask->csr_ja[nnz]=j;
                Mask->csr_a[nnz]=mask[i][j];
                nnz++;
            }
        }
    }
    Mask->csr_ia[Mask->m]=nnz;
    if (nnz!=Mask->nnz)
        printf("error creating mask: %d %d", nnz, Mask->nnz);
};

struct Mask * create_mask(char * sparse_attention_type, INT_T m, double sparsity, int band_size){
    std::vector<std::vector<double>> mask_v;
    INT_T nnz;
    mask_v = generate_sparse_attention_mask(m,nnz, sparse_attention_type, band_size, sparsity);
    struct Mask * mask =new Mask(sparse_attention_type, m, sparsity, band_size);
    mask->nnz=nnz;
    mask->csr_ja = (typeof(mask->csr_ja)) aligned_alloc(64, mask->nnz * sizeof(*mask->csr_ja));
    mask->csr_ia = (typeof(mask->csr_ia)) aligned_alloc(64, (mask->m+1) * sizeof(*mask->csr_ia));
    mask->csr_a = (typeof(mask->csr_a)) aligned_alloc(64, (mask->nnz) * sizeof(*mask->csr_a));
    dense_to_csr( mask_v, mask);
    return mask;
};



#endif /* SDDMM_MASK_H */