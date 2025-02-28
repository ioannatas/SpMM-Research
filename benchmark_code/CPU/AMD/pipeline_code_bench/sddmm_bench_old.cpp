#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sstream>
#include <bits/stdc++.h>
#include <mkl.h>
#include <omp.h>
#include <sys/time.h>



#include <unistd.h>

#include "sddmm_bench_common.h"
#include "sddmm_mask.h"

#ifdef __cplusplus
extern "C"{
#endif

	#include "macros/cpp_defines.h"
	#include "macros/macrolib.h"
	#include "time_it.h"
	#include "parallel_util.h"
	#include "pthread_functions.h"
	// #include "matrix_util.h"
	#include "array_metrics.h"

	#include "string_util.h"
	#include "parallel_io.h"
	#include "storage_formats/matrix_market/matrix_market.h"
	#include "storage_formats/dlcm_matrices/dlcm_matrix.h"
	#include "storage_formats/openfoam/openfoam_matrix.h"
	#include "read_mtx.h"

	#include "aux/csr_converter_double.h"

	#include "aux/csr_util.h"

	#include "monitoring/power/rapl.h"

	#include "artificial_matrix_generation.h"

#ifdef __cplusplus
}
#endif

#include "sddmm_kernel.h"


#ifndef BLOCK_SIZE
	#define BLOCK_SIZE  64
#endif


int prefetch_distance = 8;


int num_procs;
int process_custom_id;

long num_loops_out;


// Utils macro
#define Min(x,y) ((x)<(y)?(x):(y))
#define Max(x,y) ((x)>(y)?(x):(y))
#define Abs(x) ((x)>(0)?(x):-(x))


// #define ReferenceType  double
// #define ReferenceType  long double
#define ReferenceType  __float128

/* ldoor, mkl_ie, 8 threads:
 *     ValueType | ReferenceType       | Errors
 *     double    | double              | errors spmv: mae=2.0679e-10, max_ae=7.45058e-08, mse=1.11396e-18, mape=3.7028e-17, smape=1.8514e-17
 *     double    | double + kahan      | errors spmv: mae=1.20597e-10, max_ae=4.47035e-08, mse=3.30276e-19, mape=2.11222e-17, smape=1.05611e-17
 *     double    | long double         | errors spmv: mae=1.11432e-10, max_ae=4.47035e-08, mse=3.05508e-19, mape=1.14059e-17, smape=5.70295e-18
 *     double    | long double + kahan | errors spmv: mae=1.11426e-10, max_ae=4.47035e-08, mse=3.05491e-19, mape=1.14059e-17, smape=5.70295e-18
 *     double    | __float128          | errors spmv: mae=1.11425e-10, max_ae=4.47035e-08, mse=3.05482e-19, mape=1.14059e-17, smape=5.70295e-18
 *     double    | __float128 + kahan  | errors spmv: mae=1.11425e-10, max_ae=4.47035e-08, mse=3.05482e-19, mape=1.14059e-17, smape=5.70295e-18
 *
 *     double    | double              | errors spmv: mae=2.01305e-10, max_ae=7.45058e-08, mse=1.04495e-18, mape=6.95171e-17, smape=3.47585e-17
 *     double    | double + kahan:     | errors spmv: mae=1.47387e-10, max_ae=5.96046e-08, mse=5.21976e-19, mape=5.22525e-17, smape=2.61262e-17
 *     double    | __float128          | errors spmv: mae=1.39996e-10, max_ae=5.96046e-08, mse=4.99829e-19, mape=4.049e-17, smape=2.0245e-17
 *     double    | __float128 + kahan  | errors spmv: mae=1.39996e-10, max_ae=5.96046e-08, mse=4.99829e-19, mape=4.049e-17, smape=2.0245e-17
 *
 *     float     | double              | errors spmv: mae=0.0628685, max_ae=21.1667, mse=0.0826114, mape=1.63995e-08, smape=8.20012e-09
 *     float     | long double         | errors spmv: mae=0.0628685, max_ae=21.1667, mse=0.0826114, mape=1.63995e-08, smape=8.20012e-09
 *     float     | __float128          | errors spmv: mae=0.0628685, max_ae=21.1667, mse=0.0826114, mape=1.63995e-08, smape=8.20012e-09
 */

static inline
double
reference_to_double(void * A, long i)
{
	return (double) ((ReferenceType *) A)[i];
}

void pin_thread(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t thread = pthread_self();
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("Error setting CPU affinity");
    }
}

void spmm_helper(INT_T * csr_ia, INT_T * csr_ja, double * csr_a_ref, INT_T csr_m, INT_T csr_k, INT_T csr_nnz, INT_T n, double * x_ref, ReferenceType * y_gold)
{
	// #pragma omp parallel for
		for (long i = 0; i < csr_m; i++) {

            for (long k = 0; k < n; k++) {

                ReferenceType val, tmp, compensation;
                compensation = 0;
                ReferenceType sum = 0;
                
				for (long j = csr_ia[i]; j < csr_ia[i + 1]; j++) {
					val = csr_a_ref[j] * x_ref[k + csr_ja[j] * n]- compensation;
                    tmp = sum + val;
                    compensation = (tmp - sum) - val;
                    sum = tmp;
                }
                y_gold[i * n + k] = sum;
				// printf("%lf ", (double)y_gold[i * n + k]);
            }
        }
}

void softmax(double *input, int size)
{
    double max_val = input[0];
    double sum = 0.0;

    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }
    // #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

/** Simply return the max relative diff */
void
CheckAccuracy(struct Mask * Mask, INT_T * csr_ia_k, INT_T * csr_ja_k, double * csr_a_k, INT_T csr_m_k, INT_T csr_k_k, INT_T csr_nnz_k, INT_T n,
	INT_T * csr_ia_q, INT_T * csr_ja_q, double * csr_a_q, INT_T csr_m_q, INT_T csr_k_q, INT_T csr_nnz_q,
	INT_T * csr_ia_v, INT_T * csr_ja_v, double * csr_a_v, INT_T csr_m_v, INT_T csr_k_v, INT_T csr_nnz_v,
	double * x_ref, ValueType * K, ValueType * Q, ValueType * V, ValueType * y_final,
	ValueType * y)
{
	// printf("accuracy\n");
	__attribute__((unused)) ReferenceType epsilon_relaxed = 1e-4;
	#if DOUBLE == 0
		ReferenceType epsilon = 1e-7;
	#elif DOUBLE == 1
		ReferenceType epsilon = 1e-10;
	#endif
	long i;
	double * y_gold_ref = (typeof(y_gold_ref)) malloc(Mask->nnz * sizeof(*y_gold_ref));
	ReferenceType * y_gold = (typeof(y_gold)) malloc(Mask->nnz * sizeof(*y_gold));
	ReferenceType * y_test = (typeof(y_test)) malloc(Mask->nnz * sizeof(*y_test));
	ReferenceType * K_gold = (typeof(K_gold)) malloc(csr_m_k * n * sizeof(*K_gold));
	ReferenceType * K_test = (typeof(K_test)) malloc(csr_m_k * n * sizeof(*K_test));
	ReferenceType * Q_gold = (typeof(Q_gold)) malloc(csr_m_q * n * sizeof(*Q_gold));
	ReferenceType * Q_test = (typeof(Q_test)) malloc(csr_m_q * n * sizeof(*Q_test));
	double * V_gold_ref = (typeof(V_gold_ref)) malloc(csr_m_v * n * sizeof(*V_gold_ref));
	ReferenceType * V_gold = (typeof(V_gold)) malloc(csr_m_v * n * sizeof(*V_gold));
	ReferenceType * V_test = (typeof(V_test)) malloc(csr_m_v * n * sizeof(*V_test));
	double * y_final_gold_ref = (typeof(y_final_gold_ref)) malloc(Mask->m * n * sizeof(*y_final_gold_ref));
	ReferenceType * y_final_gold = (typeof(y_final_gold)) malloc(Mask->m * n * sizeof(*y_final_gold));
	ReferenceType * y_final_test = (typeof(y_final_test)) malloc(Mask->m * n * sizeof(*y_final_test));
	#pragma omp parallel
	{
		// ReferenceType sum;
		long i, j;
		#pragma omp for
		for(i=0;i<Mask->nnz;i++)
		{
			y_gold[i] = 0;
			y_test[i] = y[i];
			// printf(" %f ",y[i]);
		}
		#pragma omp for
		for(i=0;i<csr_m_k * n;i++)
		{
			K_gold[i] = 0;
			K_test[i] = K[i];
		}
		#pragma omp for
		for(i=0;i<csr_m_q * n;i++)
		{
			Q_gold[i] = 0;
			Q_test[i] = Q[i];
		}
		#pragma omp for
		for(i=0;i<csr_m_v * n;i++)
		{
			V_gold[i] = 0;
			V_test[i] = V[i];
		}
		#pragma omp for
		for(i=0;i<Mask->m * n;i++)
		{
			y_final_gold[i] = 0;
			y_final_test[i] = y_final[i];
		}
	}
	// printf("initiation %ld %ld\n", Mask->m, Mask->nnz);
	spmm_helper( csr_ia_k, csr_ja_k, csr_a_k, csr_m_k, csr_k_k, csr_nnz_k, n, x_ref, K_gold);
	spmm_helper( csr_ia_q, csr_ja_q, csr_a_q, csr_m_q, csr_k_q, csr_nnz_q, n, x_ref, Q_gold);
	spmm_helper( csr_ia_v, csr_ja_v, csr_a_v, csr_m_v, csr_k_v, csr_nnz_v, n, x_ref, V_gold);
	// #pragma omp parallel for
	for (long i=0; i<Mask->m * n; i++){
		V_gold_ref[i]=(double)V_gold[i];
		// printf("%lf %lf \n",V[i], V_gold_ref[i]);
	}
	long nnz=0;
	// #pragma omp parallel for
	for (long i = 0; i < Mask->m; i++) {
		// printf("%ld ", Mask->csr_ia[i]);
		for (long j = Mask->csr_ia[i]; j < Mask->csr_ia[i+1]; j++) {
			ReferenceType val, tmp, compensation;
			compensation = 0;
			ReferenceType sum = 0;
			for (long k = 0; k < n; k++) {
				val = Q_gold[i*n+k] * K_gold[i*n+k] - compensation;
				tmp = sum + val;
				compensation = (tmp - sum) - val;
				sum = tmp;  
			}
			
			y_gold[nnz] = sum;
			y_gold_ref[nnz] = (double)sum;
			nnz++;
		}
		// line++;
	}
	// softmax(y_gold_ref, Mask->nnz);
	spmm_helper(Mask->csr_ia, Mask->csr_ja, y_gold_ref, Mask->m, Mask->m, Mask->nnz, n, V_gold_ref, y_final_gold);
	
	// for (long i=0;i<Mask->nnz; i++ )
	// printf("%lf %lf \n",(double) y_final[i], (double)y_final_gold[i]);
// printf("%lf %lf \n",y_final[i], (double)y_final_gold[i]);

	ReferenceType maxDiff = 0, diff;
	// int cnt=0;
	for(i=0;i<Mask->nnz;i++)
	{
		diff = Abs(y_final_gold[i] - y_final_test[i]);
		// maxDiff = Max(maxDiff, diff);
		// printf(" %f", y_gold[i]); 
		if (y_final_gold[i] > epsilon)
		{ 
			diff = diff / abs(y_final_gold[i]);
			maxDiff = Max(maxDiff, diff);
			// std::cout << y_gold[i] << y_test[i];
			// printf("error: i=%ld %f %f \n", i, y_final_gold[i], y_final_test[i]); 
		}
		// if (i<100) 
		// 	printf("error: i=%ld/%d , a=%.10g f=%.10g\n", i, csr_m-1, (double) y_gold[i], (double) y_test[i]);
		// if (diff > epsilon_relaxed)
			// printf("error: i=%ld/%d , a=%.10g f=%.10g\n", i, csr_m-1, (double) y_gold[i], (double) y_test[i]);
		// std::cout << i << ": " << y_gold[i]-y_test[i] << "\n";
		// if (y_gold[i] != 0.0)
		// {
			// if (Abs((y_gold[i]-y_test[i])/y_gold[i]) > epsilon)
				// printf("Error: %g != %g , diff=%g , diff_frac=%g\n", y_gold[i], y_test[i], Abs(y_gold[i]-y_test[i]), Abs((y_gold[i]-y_test[i])/y_gold[i]));
			// maxDiff = Max(maxDiff, Abs((y_gold[i]-y_test[i])/y_gold[i]));
			// maxDiff = Max(maxDiff, Abs(y_gold[i]-y_test[i]));
		// }
	}
	if(maxDiff > epsilon)
		printf("Test failed! (%g)\n", reference_to_double(&maxDiff, 0));
	#pragma omp parallel
	{
		double mae, max_ae, mse, mape, smape;
		double lnQ_error, mlare, gmare;
		array_mae_concurrent(y_gold, y_test, Mask->nnz, &mae, reference_to_double);
		array_max_ae_concurrent(y_gold, y_test, Mask->nnz, &max_ae, reference_to_double);
		array_mse_concurrent(y_gold, y_test, Mask->nnz, &mse, reference_to_double);
		array_mape_concurrent(y_gold, y_test, Mask->nnz, &mape, reference_to_double);
		array_smape_concurrent(y_gold, y_test, Mask->nnz, &smape, reference_to_double);
		array_lnQ_error_concurrent(y_gold, y_test, Mask->nnz, &lnQ_error, reference_to_double);
		array_mlare_concurrent(y_gold, y_test, Mask->nnz, &mlare, reference_to_double);
		array_gmare_concurrent(y_gold, y_test, Mask->nnz, &gmare, reference_to_double);
		#pragma omp single
		printf("errors spmv: mae=%g, max_ae=%g, mse=%g, mape=%g, smape=%g, lnQ_error=%g, mlare=%g, gmare=%g\n", mae, max_ae, mse, mape, smape, lnQ_error, mlare, gmare);
	}
	free(y_gold);
	free(y_test);
	free(K_gold);
	free(K_test);
	free(Q_gold);
	free(Q_test);
	free(V_gold);
	free(V_test);
	free(y_final_gold);
	free(y_final_test);
}


int
is_directory(const char *path)
{
    struct stat stats;
    stat(path, &stats);
    // Check for file existence
    if (S_ISDIR(stats.st_mode))
        return 1;
    return 0;
}


int
get_pinning_position_from_affinity_string(const char * range_string, long len, int target_pos)
{
	long pos = 0;
	int aff = -1;
	int n1, n2;
	long i;
	for (i=0;i<len;)
	{
		n1 = atoi(&range_string[i]);
		if (pos == target_pos)
		{
			aff = n1;
			break;
		}
		while ((i < len) && (range_string[i] != ',') && (range_string[i] != '-'))
			i++;
		if (i+1 >= len)
			break;
		if (range_string[i] == ',')
		{
			pos++;
			i++;
		}
		else
		{
			i++;
			n2 = atoi(&range_string[i]);
			if (n2 < n1)
				error("Bad affinity string format.");
			if (pos + n2 - n1 >= target_pos)
			{
				aff = n1 + target_pos - pos;
				break;
			}
			pos += n2 - n1 + 1;
			while ((i < len) && (range_string[i] != ','))
				i++;
			i++;
			if (i >= len)
				break;
		}
	}
	if (aff < 0)
		error("Bad affinity string format.");
	return aff;
}

// typedef struct {
//     char opname;
//     int csr_m, csr_k, n;
//     INT_T *csr_ia, *csr_ja;
//     ValueType *csr_a, *x, *result;
// 	double *timer; 
//     struct Matrix_Format * MF_in;
// } thread_args_t;

// // Thread function for spmm
// void* void_mkl_wrap(void* arg) {
//     thread_args_t* args = (thread_args_t*)arg;
//     // Set the number of MKL threads locally
//     omp_set_num_threads(16);
//     // Measure time for spmm
//     *(args->timer) += time_it(1, args->MF_in->spmm(args->opname, args->csr_m, args->csr_k, args->n, 
//         args->csr_ia, args->csr_ja, args->csr_a, args->x, args->result););
//     return NULL;
// }

// Thread arguments structure
typedef struct {
    char opname;
    int csr_m, csr_k, n;
    INT_T *csr_ia, *csr_ja;
    ValueType *csr_a, *x, *result;
    double *timer;
    struct Matrix_Format *MF_in;
    int core_id; // Core ID for CPU affinity
} thread_args_t;

// Thread function for spmm with CPU affinity
void* void_mkl_wrap(void* arg) {
    thread_args_t* args = (thread_args_t*)arg;

    // Set CPU affinity for the thread
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(args->core_id, &cpuset);
    // pthread_t thread = pthread_self();
    // if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
    //     perror("Error setting CPU affinity");
    //     pthread_exit(NULL);
    // }

	// omp_set_nested(1);
	// mkl_set_dynamic(0);
    // omp_set_num_threads(32);
	// Configure MKL threads and affinity
    // int mkl_threads = 64; // Number of MKL threads per pthread
    // omp_set_nested(1);   // Disable nested parallelism
    // omp_set_num_threads(mkl_threads); // Set OpenMP threads for MKL
    // mkl_set_dynamic(0); scancel  // Disable dynamic threading in MKL

    // Bind MKL threads to specific cores
    // kmp_affinity_mask_t mask;
    // __kmpc_set_affinity_mask_proc(args->core_id, &mask); // Bind starting at args->core_id
    // for (int i = 0; i < mkl_threads; i++) {
    //     __kmpc_set_affinity_mask_proc(args->core_id + i, &mask);
    // }
    // __kmpc_set_affinity(&mask); // Apply affinity mask

    // Measure time for spmm
	
	// clock_t start, end;
	struct timeval start, end;
    long seconds, useconds;
    // double elapsed;

    gettimeofday(&start, NULL);
	// start = clock();
	args->MF_in->spmm(
        args->opname, args->csr_m, args->csr_k, args->n,
        args->csr_ia, args->csr_ja, args->csr_a, args->x, args->result, 20
    );
	// end = clock();
	gettimeofday(&end, NULL);
	// *(args->timer) =((double) (end - start)) / CLOCKS_PER_SEC;
	seconds = end.tv_sec - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
	*(args->timer) =  seconds + useconds / 1e6;
    // *(args->timer) += time_it(1, args->MF_in->spmm(
    //     args->opname, args->csr_m, args->csr_k, args->n,
    //     args->csr_ia, args->csr_ja, args->csr_a, args->x, args->result
    // ););
    return NULL;
}

void compute(char * matrix_name,
		INT_T * csr_ia_k, INT_T * csr_ja_k, ValueType * csr_a_k, double * csr_a_ref_k, INT_T csr_m_k, INT_T csr_k_k, INT_T n, INT_T csr_nnz_k,
		INT_T * csr_ia_q, INT_T * csr_ja_q, ValueType * csr_a_q, double * csr_a_ref_q, INT_T csr_m_q, INT_T csr_k_q, INT_T csr_nnz_q,
		INT_T * csr_ia_v, INT_T * csr_ja_v, ValueType * csr_a_v, double * csr_a_ref_v, INT_T csr_m_v, INT_T csr_k_v, INT_T csr_nnz_v,
		ValueType * K, ValueType * Q, ValueType * V,
		struct Matrix_Format * MF,
		struct Mask * Mask,
		csr_matrix * AM,
		ValueType * x, double * x_ref,ValueType * y,ValueType * y_final,
		long min_num_loops,
		long print_labels)
{
	int num_threads = omp_get_max_threads();
	int use_processes = atoi(getenv("USE_PROCESSES"));
	long num_loops;
	double gflops, gflops_final_spmm,gflops_sddmm,gflops_spmm_K,gflops_spmm_Q, gflops_spmm_V;
	double time, time_spmm_K, time_spmm_Q, time_spmm_V, time_sddmm, time_final_spmm, time_warm_up, time_after_warm_up, time_KQV;
	long buf_n = 10000;
	char buf[buf_n + 1];
	long i, j;
	double J_estimated, W_avg;
	int use_artificial_matrices = atoi(getenv("USE_ARTIFICIAL_MATRICES"));
	int use_dlcm_matrices = atoi(getenv("USE_DLCM_MATRICES"));

	if (!print_labels)
	{
		// Warm up cpu.
		__attribute__((unused)) volatile double warmup_total;
		long A_warmup_n = (1<<20) * num_threads;
		double * A_warmup;
		time_warm_up = time_it(1,
			A_warmup = (typeof(A_warmup)) malloc(A_warmup_n * sizeof(*A_warmup));
			_Pragma("omp parallel for")
			for (long i=0;i<A_warmup_n;i++)
				A_warmup[i] = 0;
			for (j=0;j<16;j++)
			{
				_Pragma("omp parallel for")
				for (long i=1;i<A_warmup_n;i++)
				{
					A_warmup[i] += A_warmup[i-1] * 7 + 3;
				}
			}
			warmup_total = A_warmup[A_warmup_n];
			free(A_warmup);
		);
		printf("time warm up %lf\n", time_warm_up);

		// Warm up caches.
		// time_warm_up = time_it(1,
			MF->spmm('K', csr_m_k, csr_k_k, n, csr_ia_k, csr_ja_k, csr_a_k, x, K, 64);
								#pragma omp parallel num_threads(1)
					{
						// int thread_id = omp_get_thread_num();
						// int core_id = thread_id; // Map threads to cores (simple 1-to-1 mapping)
						// pin_thread(core_id);
						#pragma omp single
						{
							#pragma omp task
							{
								// pin_thread(core_id);
								mkl_set_num_threads(16);
								// omp_set_num_threads(16);
								// mkl_set_num_threads_local(16);
									MF->spmm('Q', csr_m_q, csr_k_q, n, csr_ia_q, csr_ja_q, csr_a_q, x, Q, 30);
							}
							#pragma omp task
							{
								// pin_thread(core_id);
								// mkl_set_num_threads(16);
								// omp_set_num_threads(63);
									MF->spmm('K', csr_m_k, csr_k_k, n, csr_ia_k, csr_ja_k, csr_a_k, x, K, 30);
								
							}
							
							// #pragma omp task
							// {
							// 	// pin_thread(core_id);
							// 	mkl_set_num_threads(16);
							// 	// omp_set_num_threads(16);
							// 	// mkl_set_num_threads_local(16);
							// 		MF->spmm('V', csr_m_v, csr_k_v, n, csr_ia_v, csr_ja_v, csr_a_v, x, V,16);
							// }
						}
					}
					// gettimeofday(&end, NULL);
					// seconds = end.tv_sec - start.tv_sec;
					// useconds = end.tv_usec - start.tv_usec;
					// time_KQV =  seconds + useconds / 1e6;
					// 				gettimeofday(&start, NULL);
					// #pragma omp parallel num_threads(1)
					// {
					// 	// int thread_id = omp_get_thread_num();
					// 	// int core_id = thread_id; // Map threads to cores (simple 1-to-1 mapping)
					// 	// pin_thread(core_id);
					// 	#pragma omp single
					// 	{
					// 		#pragma omp task
					// 		{
					// 			// pin_thread(core_id);
					// 			mkl_set_num_threads(32);
					// 			// omp_set_num_threads(63);
					// 			time_sddmm += time_it(1,
					// 				MF->sddmm(y,32);
					// 			);
					// 		}
							
					// 		#pragma omp task
					// 		{
					// 			// pin_thread(core_id);
					// 			mkl_set_num_threads(32);
					// 			// omp_set_num_threads(16);
					// 			// mkl_set_num_threads_local(16);
					// 			time_spmm_V += time_it(1,
					// 				MF->spmm('V', csr_m_v, csr_k_v, n, csr_ia_v, csr_ja_v, csr_a_v, x, V,32);
					// 			);
					// 		}
					// 	}
					// }
		// );
		// time_warm_up = time_it(1,
		// 	MF->spmm('final', Mask->m, Mask->m, n, Mask->csr_ia, Mask->csr_ja, y, V, y_final);
		// );
		
		
	
		// /* Calculate number of loops so that the total running time is at least 1 second for stability reasons
		// (some cpus show frequency inconsistencies when running times are too small). */
		// long num_calc_loops_runs_1 = 5;
		// long num_calc_loops_runs_2;
		// time_after_warm_up = 0;
		// time_after_warm_up += time_it(1,
			// for (i=0;i<num_calc_loops_runs_1;i++)
				// MF->spmv(x, y);
		// );
		// num_calc_loops_runs_2 = 0.1 / (time_after_warm_up / num_calc_loops_runs_1);
		// if (num_calc_loops_runs_2 < 5)
			// num_calc_loops_runs_2 = 5;
		// time_after_warm_up += time_it(1,
			// for (i=0;i<num_calc_loops_runs_2;i++)
				// MF->spmv(x, y);
		// );
		// printf("time after warm up %lf\n", time_after_warm_up);
		// num_loops = 1.0 / (time_after_warm_up / (num_calc_loops_runs_1 + num_calc_loops_runs_2));
		// if (num_loops < min_num_loops)
			// num_loops = min_num_loops;
		// num_loops_out = num_loops;
		// printf("number of loops = %ld\n", num_loops);

		if (use_processes)
			raise(SIGSTOP);

		#ifdef PRINT_STATISTICS
			MF->statistics_start();
		#endif

		/*****************************************************************************************/
		struct RAPL_Register * regs;
		long regs_n;
		char * reg_ids;

		reg_ids = NULL;
		reg_ids = (char *) getenv("RAPL_REGISTERS");

		rapl_open(reg_ids, &regs, &regs_n);
		/*****************************************************************************************/

		time=0.0;
		// time_spmm_K=0; 
		time_spmm_K=0.0;
		time_spmm_Q=0.0;
		time_spmm_V=0.0;
		time_final_spmm=0.0;
		time_sddmm = 0.0;
		time_KQV = 0.0;
		num_loops = 0.0;
		while (/*time < 1.0 ||*/ num_loops < min_num_loops)
		{
			rapl_read_start(regs, regs_n);
			 if (omp_get_nested()) {
        printf("Nested parallelism is enabled %d.\n", omp_get_max_active_levels());
    } else {
        printf("Nested parallelism is NOT enabled.\n");
    }
			if (0){
				
				printf("SPLIIIIIIT\n");
				time_KQV += time_it(1,
				pthread_t threads[1];
				// pthread_attr_t attr;
				// pthread_attr_init(&attr);
    			thread_args_t thread_args[1];
				int cores[1] = {0}; // Assign threads to cores 0, 1, and 2

				// Prepare thread arguments for 'K', 'Q', and 'V' operations
				thread_args[0] = (thread_args_t){'K', csr_m_k, csr_k_k, n, csr_ia_k, csr_ja_k, csr_a_k, x, K, &time_spmm_K, MF, cores[0]};
				// thread_args[1] = (thread_args_t){'Q', csr_m_q, csr_k_q, n, csr_ia_q, csr_ja_q, csr_a_q, x, Q, &time_spmm_Q, MF, cores[1]};
				// thread_args[2] = (thread_args_t){'V', csr_m_v, csr_k_v, n, csr_ia_v, csr_ja_v, csr_a_v, x, V, &time_spmm_V, MF, cores[2]};
				// Create threads for spmm operations
				// pthread_attr_setstacksize(&attr, 1000000);
    			for (int i = 0; i < 1; i++)
        			pthread_create(&threads[i], NULL, void_mkl_wrap, (void*)&thread_args[i]);

				// Join threads
				for (int i = 0; i < 1; i++) {
					pthread_join(threads[i], NULL);
				}	
			);
			// omp_set_num_threads(32);
			// time_spmm_Q += time_it(1,
			// 						MF->spmm('Q', csr_m_q, csr_k_q, n, csr_ia_q, csr_ja_q, csr_a_q, x, Q);
			// 					);
			// time_spmm_V += time_it(1,
			// 						MF->spmm('V', csr_m_v, csr_k_v, n, csr_ia_v, csr_ja_v, csr_a_v, x, V);
								// );
				time_sddmm += time_it(1, MF->sddmm(y,64););

				time_final_spmm += time_it(1,
					MF->spmm('final', Mask->m, Mask->m, n, Mask->csr_ia, Mask->csr_ja, y, V, y_final,64);
				);
			time=time_KQV+time_spmm_V+time_sddmm+time_final_spmm;
			}
			else if(1){
				///64,512,512,512,128761,0.000918,0.000475,0.000395,0.001965,0.000724,0.004478,143.555330,306.835718,317.580253,68.433415,185.709769,150.078255,1.003910,512,512,131328
				///16,512,512,512,128761,0.001120,0.000814,0.000681,0.000902,0.000215,0.003732,117.682313,179.114532,184.133613,29.833160,125.093642,122.413354,0.202351,512,512,26266
				// time_KQV += time_it(1,
				// omp_set_nested(1);
				// omp_set_nested(1);
				// mkl_set_dynamic(0);
				// omp_set_num_threads(16);
				// mkl_set_num_threads(16);
					struct timeval start, end;
				long seconds, useconds;
				// double elapsed;
				// omp_set_num_threads(16);
				// mkl_set_num_threads_local(16);
					omp_set_nested(1);
				    mkl_set_dynamic(0);
					omp_set_max_active_levels(2);
				gettimeofday(&start, NULL);
					#pragma omp parallel num_threads(1)
					{
						// int thread_id = omp_get_thread_num();
						// int core_id = thread_id; // Map threads to cores (simple 1-to-1 mapping)
						// pin_thread(core_id);
						#pragma omp single
						{
							#pragma omp task
							{
								// pin_thread(core_id);
								// mkl_set_num_threads(16);
								// omp_set_num_threads(16);
								// mkl_set_num_threads_local(16);
								time_spmm_Q += time_it(1,
									MF->spmm('Q', csr_m_q, csr_k_q, n, csr_ia_q, csr_ja_q, csr_a_q, x, Q, 30);
								);
							}
							#pragma omp task
							{
								// pin_thread(core_id);
								// mkl_set_num_threads(16);
								// omp_set_num_threads(63);
								time_spmm_K += time_it(1,
									MF->spmm('K', csr_m_k, csr_k_k, n, csr_ia_k, csr_ja_k, csr_a_k, x, K, 30);
								);
							}
							
							// #pragma omp task
							// {
							// 	// pin_thread(core_id);
							// 	mkl_set_num_threads(16);
							// 	// omp_set_num_threads(16);
							// 	// mkl_set_num_threads_local(16);
							// 	time_spmm_V += time_it(1,
							// 		MF->spmm('V', csr_m_v, csr_k_v, n, csr_ia_v, csr_ja_v, csr_a_v, x, V,16);
							// 	);
							// }
						}
					}
					// gettimeofday(&end, NULL);
					// seconds = end.tv_sec - start.tv_sec;
					// useconds = end.tv_usec - start.tv_usec;
					// time_KQV =  seconds + useconds / 1e6;
					// 				gettimeofday(&start, NULL);
					// #pragma omp parallel num_threads(1)
					// {
					// 	// int thread_id = omp_get_thread_num();
					// 	// int core_id = thread_id; // Map threads to cores (simple 1-to-1 mapping)
					// 	// pin_thread(core_id);
					// 	#pragma omp single
					// 	{
					// 		#pragma omp task
					// 		{
					// 			// pin_thread(core_id);
					// 			mkl_set_num_threads(32);
					// 			// omp_set_num_threads(63);
					// 			time_sddmm += time_it(1,
					// 				MF->sddmm(y,32);
					// 			);
					// 		}
							
					// 		#pragma omp task
					// 		{
					// 			// pin_thread(core_id);
					// 			mkl_set_num_threads(32);
					// 			// omp_set_num_threads(16);
					// 			// mkl_set_num_threads_local(16);
					// 			time_spmm_V += time_it(1,
					// 				MF->spmm('V', csr_m_v, csr_k_v, n, csr_ia_v, csr_ja_v, csr_a_v, x, V,32);
					// 			);
					// 		}
					// 	}
					// }
					gettimeofday(&end, NULL);
					seconds = end.tv_sec - start.tv_sec;
					useconds = end.tv_usec - start.tv_usec;
					time_KQV =  seconds + useconds / 1e6;
					omp_set_nested(0);
				    mkl_set_dynamic(1);
					mkl_set_num_threads(64);
					omp_set_num_threads(64);
				// );
				// time_KQV +=((double) (end - start)) / CLOCKS_PER_SEC;
				// omp_set_num_threads(64);
				// mkl_set_num_threads(64);
				// time_spmm_Q += time_it(1,
				// 	MF->spmm('Q', csr_m_q, csr_k_q, n, csr_ia_q, csr_ja_q, csr_a_q, x, Q,64);
				// );
				time_spmm_V += time_it(1,
									MF->spmm('V', csr_m_v, csr_k_v, n, csr_ia_v, csr_ja_v, csr_a_v, x, V,64);
								);
				time_sddmm += time_it(1,
					MF->sddmm(y,64);
				);
				time_final_spmm += time_it(1,
					MF->spmm('final', Mask->m, Mask->m, n, Mask->csr_ia, Mask->csr_ja, y, V, y_final,64);
				);
				time=time_KQV+time_spmm_V+time_sddmm+time_final_spmm;
			}
			else if (0){
				time_spmm_K += time_it(1,
					MF->spmm('K', csr_m_k, csr_k_k, n, csr_ia_k, csr_ja_k, csr_a_k, x, K,64);
				);
				printf("%ld %ld %ld \n", csr_m_k, csr_k_k, csr_nnz_k);

				time_spmm_Q += time_it(1,
					MF->spmm('Q', csr_m_q, csr_k_q, n, csr_ia_q, csr_ja_q, csr_a_q, x, Q,64);
				);

				time_sddmm += time_it(1,
					MF->sddmm(y,64);
				);

				time_spmm_V += time_it(1,
					MF->spmm('V', csr_m_v, csr_k_v, n, csr_ia_v, csr_ja_v, csr_a_v, x, V,64);
				);
				time_final_spmm += time_it(1,
					MF->spmm('final', Mask->m, Mask->m, n, Mask->csr_ia, Mask->csr_ja, y, V, y_final,64);
				);
				time=time_spmm_K+time_spmm_Q+time_spmm_V+time_sddmm+time_final_spmm;
			}
		// 	for (int i=0;i<Mask->nnz;i++)
		// 	printf("%lf ",y[i]);
		// printf("sddmm \n");
			rapl_read_end(regs, regs_n);

			num_loops++;
		}
		// #if SPLIT==1
		// 	time=time_KQV+time_sddmm+time_final_spmm;
		// #elif SPLIT==0
		// 	time=time_spmm_K+time_spmm_Q+time_spmm_V+time_sddmm+time_final_spmm;
		// #endif
		num_loops_out = num_loops;
		printf("number of loops = %ld\n", num_loops);

		/*****************************************************************************************/
		J_estimated = 0;
		for (i=0;i<regs_n;i++){
			// printf("'%s' total joule = %g\n", regs[i].type, ((double) regs[i].uj_accum) / 1000000);
			J_estimated += ((double) regs[i].uj_accum) / 1e6;
		}
		rapl_close(regs, regs_n);
		free(regs);
		W_avg = J_estimated / time;
		// printf("J_estimated = %lf\tW_avg = %lf\n", J_estimated, W_avg);
		/*****************************************************************************************/

		//=============================================================================
		//= Output section.
		//=============================================================================

		std::stringstream stream;
		if (MF->format_name=="MKL_GEMM"){
			// gflops = csr_k * csr_m * n / time * num_loops * 2 * 1e-9;  
			gflops = csr_k_k * 2 * 1e-9 * csr_m_k * n / time * num_loops;   
			// printf("wright %d * %d * %d / %lf * %ld * 2 * 1e-9;\n", csr_k, csr_m , n, time ,num_loops);
		} else{
			// gflops = csr_nnz * n / time * num_loops * 2 * 1e-9; 
			gflops_spmm_K = csr_nnz_k * 2 * 1e-9 * n / time_spmm_K * num_loops ;
			gflops_spmm_Q = csr_nnz_q * 2 * 1e-9 * n / time_spmm_Q * num_loops ;
			gflops_spmm_V = csr_nnz_v * 2 * 1e-9 * n / time_spmm_V * num_loops ; 
			gflops_sddmm = Mask->nnz * 2 * 1e-9 * n / time_sddmm * num_loops ;    // Use csr_nnz to be sure we have the initial nnz (there is no coo for artificial AM).
			gflops_final_spmm = Mask->nnz * 2 * 1e-9 * n / time_final_spmm * num_loops ;
			gflops = (csr_nnz_k+csr_nnz_q+csr_nnz_v+Mask->nnz+Mask->nnz) * 2 * 1e-9 * n / time * num_loops ;
			// printf("wright %d * %d * %d / %lf * %ld * 2 * 1e-9;\n", csr_k, csr_m , n, time ,num_loops);
			// printf("HERE %lf %lf %lf %lf %lf\n", gflops_spmm_K, gflops_spmm_Q, gflops_spmm_V, gflops_sddmm,gflops_final_spmm);
			// printf("HERE %lf %lf %lf %lf %lf\n", time_spmm_K, time_spmm_Q, time_spmm_V, time_sddmm,time_final_spmm);
			// printf("HERE %ld %ld %ld %ld\n", csr_nnz_k, csr_nnz_q, csr_nnz_v, Mask->nnz);
		}
	}

	if (!use_artificial_matrices)
	{
		if (print_labels)
		{
			i = 0;
			i += snprintf(buf + i, buf_n - i, "%s", "matrix_name");
			if (use_processes)
			{
				i += snprintf(buf + i, buf_n - i, ",%s", "num_procs");
			}
			else
			{
				i += snprintf(buf + i, buf_n - i, ",%s", "num_threads");
			}
			i += snprintf(buf + i, buf_n - i, ",%s", "input_columns");
			i += snprintf(buf + i, buf_n - i, ",%s", "csr_m");
			i += snprintf(buf + i, buf_n - i, ",%s", "csr_k");
			i += snprintf(buf + i, buf_n - i, ",%s", "csr_nnz");
			i += snprintf(buf + i, buf_n - i, ",%s", "time_spmm_K");
			i += snprintf(buf + i, buf_n - i, ",%s", "time_spmm_Q");
			i += snprintf(buf + i, buf_n - i, ",%s", "time_spmm_V");
			i += snprintf(buf + i, buf_n - i, ",%s", "time_sddmm");
			i += snprintf(buf + i, buf_n - i, ",%s", "time_final_spmm");
			i += snprintf(buf + i, buf_n - i, ",%s", "time");
			i += snprintf(buf + i, buf_n - i, ",%s", "gflops_spmm_K");
			i += snprintf(buf + i, buf_n - i, ",%s", "gflops_spmm_Q");
			i += snprintf(buf + i, buf_n - i, ",%s", "gflops_spmm_V");
			i += snprintf(buf + i, buf_n - i, ",%s", "gflops_sddmm");
			i += snprintf(buf + i, buf_n - i, ",%s", "gflops_final_spmm");
			i += snprintf(buf + i, buf_n - i, ",%s", "gflops");
			i += snprintf(buf + i, buf_n - i, ",%s", "csr_mem_footprint");
			// i += snprintf(buf + i, buf_n - i, ",%s", "W_avg");
			// i += snprintf(buf + i, buf_n - i, ",%s", "J_estimated");
			// i += snprintf(buf + i, buf_n - i, ",%s", "format_name");
			i += snprintf(buf + i, buf_n - i, ",%s", "m");
			i += snprintf(buf + i, buf_n - i, ",%s", "n");
			i += snprintf(buf + i, buf_n - i, ",%s", "nnz");
			// i += snprintf(buf + i, buf_n - i, ",%s", "mem_footprint");
			// i += snprintf(buf + i, buf_n - i, ",%s", "mem_ratio");
			// i += snprintf(buf + i, buf_n - i, ",%s", "CSRCV_NUM_PACKET_VALS");
			#ifdef PRINT_STATISTICS
				i += statistics_print_labels(buf + i, buf_n - i);
			#endif
			buf[i] = '\0';
			fprintf(stderr, "%s\n", buf);
			return;
		}
		i = 0;
		i += snprintf(buf + i, buf_n - i, "%s", matrix_name);
		if (use_processes)
		{
			i += snprintf(buf + i, buf_n - i, ",%d", num_procs);
		}
		else
		{
			i += snprintf(buf + i, buf_n - i, ",%d", omp_get_max_threads());
		}
		i += snprintf(buf + i, buf_n - i, ",%u", n);
		i += snprintf(buf + i, buf_n - i, ",%u", csr_m_k);
		i += snprintf(buf + i, buf_n - i, ",%u", csr_k_k);
		i += snprintf(buf + i, buf_n - i, ",%u", csr_nnz_k);
		i += snprintf(buf + i, buf_n - i, ",%lf", time_spmm_K);
		i += snprintf(buf + i, buf_n - i, ",%lf", time_spmm_Q);
		i += snprintf(buf + i, buf_n - i, ",%lf", time_spmm_V);
		i += snprintf(buf + i, buf_n - i, ",%lf", time_sddmm);
		i += snprintf(buf + i, buf_n - i, ",%lf", time_final_spmm);
		i += snprintf(buf + i, buf_n - i, ",%lf", time);
		i += snprintf(buf + i, buf_n - i, ",%lf", gflops_spmm_K);
		i += snprintf(buf + i, buf_n - i, ",%lf", gflops_spmm_Q);
		i += snprintf(buf + i, buf_n - i, ",%lf", gflops_spmm_V);
		i += snprintf(buf + i, buf_n - i, ",%lf", gflops_sddmm);
		i += snprintf(buf + i, buf_n - i, ",%lf", gflops_final_spmm);
		i += snprintf(buf + i, buf_n - i, ",%lf", gflops);
		i += snprintf(buf + i, buf_n - i, ",%lf", MF->csr_mem_footprint / (1024*1024));
		// i += snprintf(buf + i, buf_n - i, ",%lf", W_avg);
		// i += snprintf(buf + i, buf_n - i, ",%lf", J_estimated);
		// i += snprintf(buf + i, buf_n - i, ",%s", MF->format_name);
		i += snprintf(buf + i, buf_n - i, ",%u", MF->m);
		i += snprintf(buf + i, buf_n - i, ",%u", MF->n);
		i += snprintf(buf + i, buf_n - i, ",%u", MF->nnz);
		// i += snprintf(buf + i, buf_n - i, ",%lf", MF->mem_footprint / (1024*1024));
		// i += snprintf(buf + i, buf_n - i, ",%lf", MF->mem_footprint / MF->csr_mem_footprint);
		// i += snprintf(buf + i, buf_n - i, ",%ld", atol(getenv("CSRCV_NUM_PACKET_VALS")));
		#ifdef PRINT_STATISTICS
			i += MF->statistics_print_data(buf + i, buf_n - i);
		#endif
		buf[i] = '\0';
		fprintf(stderr, "%s\n", buf);
		printf("before acccuracy\n");
		// CheckAccuracy(Mask, csr_ia_k, csr_ja_k, csr_a_ref_k, csr_m_k, csr_k_k, csr_nnz_k, n,
		//  csr_ia_q, csr_ja_q, csr_a_ref_q, csr_m_q, csr_k_q, csr_nnz_q, csr_ia_v, csr_ja_v, csr_a_ref_v, csr_m_v, csr_k_v, csr_nnz_v, x_ref, K, Q, V, y_final, y);
	}
	else
	{
		if (print_labels)
		{
			i = 0;
			i += snprintf(buf + i, buf_n - i, "%s",  "matrix_name");
			i += snprintf(buf + i, buf_n - i, ",%s", "distribution");
			i += snprintf(buf + i, buf_n - i, ",%s", "placement");
			i += snprintf(buf + i, buf_n - i, ",%s", "seed");
			i += snprintf(buf + i, buf_n - i, ",%s", "nr_rows");
			i += snprintf(buf + i, buf_n - i, ",%s", "nr_cols");
			i += snprintf(buf + i, buf_n - i, ",%s", "nr_nzeros");
			i += snprintf(buf + i, buf_n - i, ",%s", "density");
			i += snprintf(buf + i, buf_n - i, ",%s", "mem_footprint");
			i += snprintf(buf + i, buf_n - i, ",%s", "mem_range");
			i += snprintf(buf + i, buf_n - i, ",%s", "avg_nnz_per_row");
			i += snprintf(buf + i, buf_n - i, ",%s", "std_nnz_per_row");
			i += snprintf(buf + i, buf_n - i, ",%s", "avg_bw");
			i += snprintf(buf + i, buf_n - i, ",%s", "std_bw");
			i += snprintf(buf + i, buf_n - i, ",%s", "avg_bw_scaled");
			i += snprintf(buf + i, buf_n - i, ",%s", "std_bw_scaled");
			i += snprintf(buf + i, buf_n - i, ",%s", "avg_sc");
			i += snprintf(buf + i, buf_n - i, ",%s", "std_sc");
			i += snprintf(buf + i, buf_n - i, ",%s", "avg_sc_scaled");
			i += snprintf(buf + i, buf_n - i, ",%s", "std_sc_scaled");
			i += snprintf(buf + i, buf_n - i, ",%s", "skew");
			i += snprintf(buf + i, buf_n - i, ",%s", "avg_num_neighbours");
			i += snprintf(buf + i, buf_n - i, ",%s", "cross_row_similarity");
			i += snprintf(buf + i, buf_n - i, ",%s", "format_name");
			i += snprintf(buf + i, buf_n - i, ",%s", "time");
			i += snprintf(buf + i, buf_n - i, ",%s", "gflops");
			i += snprintf(buf + i, buf_n - i, ",%s", "W_avg");
			i += snprintf(buf + i, buf_n - i, ",%s", "J_estimated");
			#ifdef PRINT_STATISTICS
				i += statistics_print_labels(buf + i, buf_n - i);
			#endif
			buf[i] = '\0';
			fprintf(stderr, "%s\n", buf);
			return;
		}
		i = 0;
		i += snprintf(buf + i, buf_n - i, "synthetic");
		i += snprintf(buf + i, buf_n - i, ",%s" , AM->distribution);
		i += snprintf(buf + i, buf_n - i, ",%s" , AM->placement);
		i += snprintf(buf + i, buf_n - i, ",%d" , AM->seed);
		i += snprintf(buf + i, buf_n - i, ",%u" , AM->nr_rows);
		i += snprintf(buf + i, buf_n - i, ",%u" , AM->nr_cols);
		i += snprintf(buf + i, buf_n - i, ",%u" , AM->nr_nzeros);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->density);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->mem_footprint);
		i += snprintf(buf + i, buf_n - i, ",%s" , AM->mem_range);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->avg_nnz_per_row);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->std_nnz_per_row);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->avg_bw);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->std_bw);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->avg_bw_scaled);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->std_bw_scaled);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->avg_sc);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->std_sc);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->avg_sc_scaled);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->std_sc_scaled);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->skew);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->avg_num_neighbours);
		i += snprintf(buf + i, buf_n - i, ",%lf", AM->cross_row_similarity);
		i += snprintf(buf + i, buf_n - i, ",%s" , MF->format_name);
		i += snprintf(buf + i, buf_n - i, ",%lf", time);
		i += snprintf(buf + i, buf_n - i, ",%lf", gflops);
		i += snprintf(buf + i, buf_n - i, ",%lf", W_avg);
		i += snprintf(buf + i, buf_n - i, ",%lf", J_estimated);
		#ifdef PRINT_STATISTICS
			i += MF->statistics_print_data(buf + i, buf_n - i);
		#endif
		buf[i] = '\0';
		fprintf(stderr, "%s\n", buf);
	}
}


//==========================================================================================================================================
//= Main
//==========================================================================================================================================

int
main(int argc, char **argv)
{
	__attribute__((unused)) int num_threads;

	struct Matrix_Market * MTX;
	struct DLCM_Matrix * SMTX_k;
	struct DLCM_Matrix * SMTX_q;
	struct DLCM_Matrix * SMTX_v;
	double * mtx_val_k = NULL;
	INT_T * mtx_rowind_k = NULL;
	INT_T * mtx_colind_k = NULL;
	INT_T mtx_m_k = 0;
	INT_T mtx_k_k = 0;
	INT_T mtx_nnz_k = 0;
	double * mtx_val_q = NULL;
	INT_T * mtx_rowind_q = NULL;
	INT_T * mtx_colind_q = NULL;
	INT_T mtx_m_q = 0;
	INT_T mtx_k_q = 0;
	INT_T mtx_nnz_q = 0;
	double * mtx_val_v = NULL;
	INT_T * mtx_rowind_v = NULL;
	INT_T * mtx_colind_v = NULL;
	INT_T mtx_m_v = 0;
	INT_T mtx_k_v = 0;
	INT_T mtx_nnz_v = 0;
	double * x_ref;
	double * z_ref;

	double * csr_a_ref_k = NULL;
	double * csr_a_ref_q = NULL;
	double * csr_a_ref_v = NULL;

	ValueType * csr_a_k = NULL; // values (of size NNZ)
	INT_T * csr_ia_k = NULL;    // rowptr (of size m+1)
	INT_T * csr_ja_k = NULL;    // colidx of each NNZ (of size nnz)
	INT_T csr_m_k = 0;
	INT_T csr_k_k = 0;
	INT_T csr_nnz_k = 0;
	ValueType * csr_a_q = NULL; // values (of size NNZ)
	INT_T * csr_ia_q = NULL;    // rowptr (of size m+1)
	INT_T * csr_ja_q = NULL;    // colidx of each NNZ (of size nnz)
	INT_T csr_m_q = 0;
	INT_T csr_k_q = 0;
	INT_T csr_nnz_q = 0;
	ValueType * csr_a_v = NULL; // values (of size NNZ)
	INT_T * csr_ia_v = NULL;    // rowptr (of size m+1)
	INT_T * csr_ja_v = NULL;    // colidx of each NNZ (of size nnz)
	INT_T csr_m_v = 0;
	INT_T csr_k_v = 0;
	INT_T csr_nnz_v = 0;
	INT_T n;
	INT_T num_cols = atoi(getenv("NUM_COLS"));
	INT_T band_size = atoi(getenv("BAND_SIZE"));
	char * sparse_attention_type = getenv("SPARSE_ATTENTION_TYPE");
	double sparsity = atof(getenv("SPARSITY"));
	printf("sparsity: %lf\n", sparsity);

	struct Matrix_Format * MF;   // Real matrices.
	csr_matrix * AM = NULL;
	struct Mask * mask = NULL;
	ValueType * K;
	ValueType * Q;
	ValueType * V;
	ValueType * x;
	ValueType * y;
	ValueType * y_final;
	char matrix_name[1000];
	__attribute__((unused)) double time;
	__attribute__((unused)) long i, j;

	int use_artificial_matrices = atoi(getenv("USE_ARTIFICIAL_MATRICES"));
	int use_dlcm_matrices = atoi(getenv("USE_DLCM_MATRICES"));

	// Wake omp up from eternal slumber.
	#pragma omp parallel
	{
		num_threads = omp_get_max_threads();
	}
	printf("max threads %d\n", num_threads);

	// Just print the labels and exit.
	if (argc == 1)
	{
		compute(NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0,NULL, NULL,NULL, NULL,0,0 ,0,NULL,NULL,NULL,NULL,0,0,0,NULL,NULL,NULL, NULL, NULL, NULL,NULL, NULL, NULL, 0, 1);
		return 0;
	}

	int use_processes = atoi(getenv("USE_PROCESSES"));
	if (use_processes)
	{
		num_procs = atoi(getenv("NUM_PROCESSES"));
		pid_t pids[num_procs];
		pid_t pid;
		pthread_t tid;
		int core;
		long j;
		for (j=0;j<num_procs;j++)
		{
			pid = fork();
			if (pid == -1)
				error("fork");
			if (pid == 0)
			{
				char * gomp_aff_str = getenv("GOMP_CPU_AFFINITY");
				long len = strlen(gomp_aff_str);
				long buf_n = 1000;
				char buf[buf_n];
				process_custom_id = j;
				core = get_pinning_position_from_affinity_string(gomp_aff_str, len, j);
				tid = pthread_self();
				set_affinity(tid, core);
				snprintf(buf, buf_n, "%d", core);             // Also set environment variables for other libraries that might try to change affinity themselves.
				setenv("GOMP_CPU_AFFINITY", buf, 1);          // Setting 'XLSMPOPTS' has no effect after the program has started.
				// printf("%ld: affinity=%d\n", j, core);
				goto child_proc_label;
			}
			pids[j] = pid;
		}
		tid = pthread_self();
		set_affinity(tid, 0);
		for (j=0;j<num_procs;j++)
			waitpid(-1, NULL, WUNTRACED);
		for (j=0;j<num_procs;j++)
			kill(pids[j], SIGCONT);
		for (j=0;j<num_procs;j++)
			waitpid(-1, NULL, WUNTRACED);
		exit(0);
	}

child_proc_label:

	if (!use_artificial_matrices)
	{
		char * file_in_k;
		char * file_in_q;	
		char * file_in_v;
		i = 1;
		file_in_k = argv[i++];
		file_in_q = argv[i++];
		file_in_v = argv[i++];
		snprintf(matrix_name, sizeof(matrix_name), "%s", file_in_k);
		printf("MATRIX NAME %s\n", matrix_name);

		time = time_it(1,
			if (use_dlcm_matrices)
			{
				long expand_symmetry = 1;
				long pattern_dummy_vals = 1;
				SMTX_k = smtx_read(file_in_k, expand_symmetry, pattern_dummy_vals);
				SMTX_q = smtx_read(file_in_q, expand_symmetry, pattern_dummy_vals);
				SMTX_v = smtx_read(file_in_v, expand_symmetry, pattern_dummy_vals);
				mtx_rowind_k = SMTX_k->R;
				mtx_rowind_q = SMTX_q->R;
				mtx_rowind_v = SMTX_v->R;
				mtx_colind_k = SMTX_k->C;
				mtx_colind_q = SMTX_q->C;
				mtx_colind_v = SMTX_v->C;
				mtx_m_k = SMTX_k->m;
				mtx_m_q = SMTX_q->m;
				mtx_m_v = SMTX_v->m;
				mtx_k_k = SMTX_k->k;
				mtx_k_q = SMTX_q->k;
				mtx_k_v = SMTX_v->k;
				mtx_nnz_k = SMTX_k->nnz;
				mtx_nnz_q = SMTX_q->nnz;
				mtx_nnz_v = SMTX_v->nnz;
				// mtx_rowind = (typeof(mtx_rowind)) aligned_alloc(64, mtx_nnz * sizeof(*mtx_rowind));
				// mtx_colind = (typeof(mtx_colind)) aligned_alloc(64, mtx_nnz * sizeof(*mtx_colind));
				mtx_val_k = (typeof(mtx_val_k)) malloc(mtx_nnz_k * sizeof(*mtx_val_k));
				mtx_val_q = (typeof(mtx_val_q)) malloc(mtx_nnz_q * sizeof(*mtx_val_q));
				mtx_val_v = (typeof(mtx_val_v)) malloc(mtx_nnz_v * sizeof(*mtx_val_v));
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_nnz_k;i++)
				{
					// mtx_colind[i] = SMTX->C[i];
					mtx_val_k[i] = ((ValueType *) SMTX_k->V)[i];
					// printf("%f %f ", ((ValueType *) SMTX->V)[i], mtx_val[i]);
				}
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_nnz_q;i++)
				{
					// mtx_colind[i] = SMTX->C[i];
					mtx_val_q[i] = ((ValueType *) SMTX_q->V)[i];
					// printf("%f %f ", ((ValueType *) SMTX->V)[i], mtx_val[i]);
				}
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_nnz_v;i++)
				{
					// mtx_colind[i] = SMTX->C[i];
					mtx_val_v[i] = ((ValueType *) SMTX_v->V)[i];
					// printf("%f %f ", ((ValueType *) SMTX->V)[i], mtx_val[i]);
				}
				// for (long i=0;i<mtx_m+1 ;i++){
				// 	mtx_rowind[i] = SMTX->R[i];
				// 	// printf("%d %d ", SMTX->R[i], mtx_rowind[i]);
				// }
				free(SMTX_k->V);
				free(SMTX_q->V);
				free(SMTX_v->V);
				// free(SMTX->R);
				// free(SMTX->C);

			}
			// else if (is_directory(file_in))
			// {
			// 	int nnz_non_diag, N;
			// 	int * rowind, * colind;
			// 	read_openfoam_matrix_dir(file_in, &rowind, &colind, &N, &nnz_non_diag);
			// 	mtx_m = N;
			// 	mtx_k = N;
			// 	mtx_nnz = N + nnz_non_diag;
			// 	mtx_rowind = (typeof(mtx_rowind)) aligned_alloc(64, mtx_nnz * sizeof(*mtx_rowind));
			// 	mtx_colind = (typeof(mtx_colind)) aligned_alloc(64, mtx_nnz * sizeof(*mtx_colind));
			// 	mtx_val = (typeof(mtx_val)) aligned_alloc(64, mtx_nnz * sizeof(*mtx_val));
			// 	_Pragma("omp parallel for")
			// 	for (long i=0;i<mtx_nnz;i++)
			// 	{
			// 		mtx_rowind[i] = rowind[i];
			// 		mtx_colind[i] = colind[i];
			// 		mtx_val[i] = 1.0;
			// 	}
			// 	free(rowind);
			// 	free(colind);
			// } 
			// else
			// {

			// 	// create_coo_matrix(file_in, &mtx_val, &mtx_rowind, &mtx_colind, &mtx_m, &mtx_n, &mtx_nnz);
			// 	long expand_symmetry = 1;
			// 	long pattern_dummy_vals = 1;
			// 	MTX = mtx_read(file_in, expand_symmetry, pattern_dummy_vals);
			// 	mtx_rowind = MTX->R;
			// 	mtx_colind = MTX->C;
			// 	mtx_m = MTX->m;
			// 	mtx_k = MTX->k;
			// 	mtx_nnz = MTX->nnz;
			// 	if (!strcmp(MTX->field, "integer"))
			// 	{
			// 		mtx_val = (typeof(mtx_val)) malloc(mtx_nnz * sizeof(*mtx_val));
			// 		_Pragma("omp parallel for")
			// 		for (long i=0;i<mtx_nnz;i++)
			// 		{
			// 			mtx_val[i] = ((int *) MTX->V)[i];
			// 		}
			// 		free(MTX->V);
			// 	}
			// 	else if (!strcmp(MTX->field, "complex"))
			// 	{
			// 		mtx_val = (typeof(mtx_val)) malloc(mtx_nnz * sizeof(*mtx_val));
			// 		_Pragma("omp parallel for")
			// 		for (long i=0;i<mtx_nnz;i++)
			// 		{
			// 			#if DOUBLE == 0
			// 				mtx_val[i] = cabsf(((complex ValueType *) MTX->V)[i]);
			// 			#else
			// 				mtx_val[i] = cabs(((complex ValueType *) MTX->V)[i]);
			// 			#endif
			// 		}
			// 		free(MTX->V);
			// 	}
			// 	else{
			// 		// mtx_val = (double *) MTX->V;
			// 		mtx_val = (typeof(mtx_val)) malloc(mtx_nnz * sizeof(*mtx_val));
			// 		_Pragma("omp parallel for")
			// 		for (long i=0;i<mtx_nnz;i++)
			// 		{
			// 			mtx_val[i] = ((ValueType *) MTX->V)[i];
			// 		}
			// 		free(MTX->V);
			// 	}
					

			// }
		);
		printf("time read: %lf\n", time);
		if (use_dlcm_matrices)
		{
			time = time_it(1,
				csr_a_ref_k = (typeof(csr_a_ref_k)) aligned_alloc(64, (mtx_nnz_k + VECTOR_ELEM_NUM) * sizeof(*csr_a_ref_k));
				csr_a_k = (typeof(csr_a_k)) aligned_alloc(64, (mtx_nnz_k + VECTOR_ELEM_NUM) * sizeof(*csr_a_k));
				csr_ja_k = (typeof(csr_ja_k)) aligned_alloc(64, (mtx_nnz_k + VECTOR_ELEM_NUM) * sizeof(*csr_ja_k));
				csr_ia_k = (typeof(csr_ia_k)) aligned_alloc(64, (mtx_m_k+1 + VECTOR_ELEM_NUM) * sizeof(*csr_ia_k));
				csr_m_k = mtx_m_k;
				csr_k_k = mtx_k_k;
				csr_nnz_k = mtx_nnz_k;
				csr_a_ref_q = (typeof(csr_a_ref_q)) aligned_alloc(64, (mtx_nnz_q + VECTOR_ELEM_NUM) * sizeof(*csr_a_ref_q));
				csr_a_q = (typeof(csr_a_q)) aligned_alloc(64, (mtx_nnz_q + VECTOR_ELEM_NUM) * sizeof(*csr_a_q));
				csr_ja_q = (typeof(csr_ja_q)) aligned_alloc(64, (mtx_nnz_q + VECTOR_ELEM_NUM) * sizeof(*csr_ja_q));
				csr_ia_q = (typeof(csr_ia_q)) aligned_alloc(64, (mtx_m_q+1 + VECTOR_ELEM_NUM) * sizeof(*csr_ia_q));
				csr_m_q= mtx_m_q;
				csr_k_q = mtx_k_q;
				csr_nnz_q = mtx_nnz_q;
				csr_a_ref_v = (typeof(csr_a_ref_v)) aligned_alloc(64, (mtx_nnz_v + VECTOR_ELEM_NUM) * sizeof(*csr_a_ref_v));
				csr_a_v = (typeof(csr_a_v)) aligned_alloc(64, (mtx_nnz_v + VECTOR_ELEM_NUM) * sizeof(*csr_a_v));
				csr_ja_v = (typeof(csr_ja_v)) aligned_alloc(64, (mtx_nnz_v + VECTOR_ELEM_NUM) * sizeof(*csr_ja_v));
				csr_ia_v = (typeof(csr_ia_v)) aligned_alloc(64, (mtx_m_v+1 + VECTOR_ELEM_NUM) * sizeof(*csr_ia_v));
				csr_m_v = mtx_m_v;
				csr_k_v = mtx_k_v;
				csr_nnz_v = mtx_nnz_v;
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_nnz_k ;i++)
				{
					csr_a_ref_k[i] = mtx_val_k[i];
					csr_a_k[i] = (ValueType) mtx_val_k[i];
					// printf("%ld %ld ", csr_ja[i], mtx_colind[i]);
					csr_ja_k[i] =  mtx_colind_k[i];
					// printf("%d %d ", csr_ja[i], mtx_colind[i]);
					// printf(" %f %f ", mtx_val[i], csr_a[i]);
					// csr_ja[i]=0;
				}
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_m_k+1 ;i++){
					csr_ia_k[i] =  mtx_rowind_k[i];
					// printf("%d %d ", csr_ia[i], mtx_rowind[i]);
				}
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_nnz_q ;i++)
				{
					csr_a_ref_q[i] = mtx_val_q[i];
					csr_a_q[i] = (ValueType) mtx_val_q[i];
					// printf("%ld %ld ", csr_ja[i], mtx_colind[i]);
					csr_ja_q[i] =  mtx_colind_q[i];
					// printf("%d %d ", csr_ja[i], mtx_colind[i]);
					// printf(" %f %f ", mtx_val[i], csr_a[i]);
					// csr_ja[i]=0;
				}
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_m_q+1 ;i++){
					csr_ia_q[i] =  mtx_rowind_q[i];
					// printf("%d %d ", csr_ia[i], mtx_rowind[i]);
				}
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_nnz_v ;i++)
				{
					csr_a_ref_v[i] = mtx_val_v[i];
					csr_a_v[i] = (ValueType) mtx_val_v[i];
					// printf("%ld %ld ", csr_ja[i], mtx_colind[i]);
					csr_ja_v[i] =  mtx_colind_v[i];
					// printf("%d %d ", csr_ja[i], mtx_colind[i]);
					// printf(" %f %f ", mtx_val[i], csr_a[i]);
					// csr_ja[i]=0;
				}
				_Pragma("omp parallel for")
				for (long i=0;i<mtx_m_v+1 ;i++){
					csr_ia_v[i] =  mtx_rowind_v[i];
					// printf("%d %d ", csr_ia[i], mtx_rowind[i]);
				}
				_Pragma("omp parallel for")
				for (long i=0;i<VECTOR_ELEM_NUM;i++)
				{
					csr_ia_k[mtx_m_k+1 + i] = 0;
					csr_a_k[mtx_nnz_k + i] = 0.0;
					csr_ja_k[mtx_nnz_k + i] = 0;
					csr_ia_q[mtx_m_q+1 + i] = 0;
					csr_a_q[mtx_nnz_q + i] = 0.0;
					csr_ja_q[mtx_nnz_q + i] = 0;
					csr_ia_v[mtx_m_v+1 + i] = 0;
					csr_a_v[mtx_nnz_v + i] = 0.0;
					csr_ja_v[mtx_nnz_v + i] = 0;
				}
			);
			printf("\ntime copy: %lf\n", time);
		}
		// else
		// {
		// 	time = time_it(1,
		// 		csr_a_ref = (typeof(csr_a_ref)) aligned_alloc(64, (mtx_nnz + VECTOR_ELEM_NUM) * sizeof(*csr_a_ref));
		// 		csr_a = (typeof(csr_a)) aligned_alloc(64, (mtx_nnz + VECTOR_ELEM_NUM) * sizeof(*csr_a));
		// 		csr_ja = (typeof(csr_ja)) aligned_alloc(64, (mtx_nnz + VECTOR_ELEM_NUM) * sizeof(*csr_ja));
		// 		csr_ia = (typeof(csr_ia)) aligned_alloc(64, (mtx_m+1 + VECTOR_ELEM_NUM) * sizeof(*csr_ia));
		// 		csr_m = mtx_m;
		// 		csr_k = mtx_k;
		// 		csr_nnz = mtx_nnz;
		// 		_Pragma("omp parallel for")
		// 		for (long i=0;i<mtx_nnz + VECTOR_ELEM_NUM;i++)
		// 		{
		// 			csr_a_ref[i] = 0.0;
		// 			csr_ja[i] = 0;
		// 		}
		// 		_Pragma("omp parallel for")
		// 		for (long i=0;i<mtx_m+1 + VECTOR_ELEM_NUM;i++)
		// 			csr_ia[i] = 0;
		// 		coo_to_csr(mtx_rowind, mtx_colind, mtx_val, mtx_m, mtx_k, mtx_nnz, csr_ia, csr_ja, csr_a_ref, 1, 0);
		// 		_Pragma("omp parallel for")
		// 		for (long i=0;i<mtx_nnz + VECTOR_ELEM_NUM;i++)
		// 			csr_a[i] = (ValueType) csr_a_ref[i];
		// 	);
		// 	printf("time coo to csr: %lf\n", time);
		// }
		// for (long i=0;i<mtx_m+1 ;i++)
		// 	printf("%d ", csr_ia[i]);
		// printf("hey1 %d \n", mtx_m);
		// for (long i=0;i<mtx_m+1 ;i++)
		// 	printf("%d ", mtx_rowind[i]);
		free(mtx_rowind_k);
		free(mtx_rowind_q);
		free(mtx_rowind_v);
		free(mtx_colind_k);
		free(mtx_colind_q);
		free(mtx_colind_v);
		free(mtx_val_k);
		free(mtx_val_q);
		free(mtx_val_v);
		// printf("FREE\n");
	}
	// else
	// {
	// 	time = time_it(1,

	// 		long nr_rows, nr_cols, seed;
	// 		double avg_nnz_per_row, std_nnz_per_row, bw, skew;
	// 		double avg_num_neighbours;
	// 		double cross_row_similarity;
	// 		char * distribution, * placement;
	// 		long i;

	// 		i = 1;
	// 		nr_rows = atoi(argv[i++]);
	// 		nr_cols = atoi(argv[i++]);
	// 		avg_nnz_per_row = atof(argv[i++]);
	// 		std_nnz_per_row = atof(argv[i++]);
	// 		distribution = argv[i++];
	// 		placement = argv[i++];
	// 		bw = atof(argv[i++]);
	// 		skew = atof(argv[i++]);
	// 		avg_num_neighbours = atof(argv[i++]);
	// 		cross_row_similarity = atof(argv[i++]);
	// 		seed = atoi(argv[i++]);
	// 		AM = artificial_matrix_generation(nr_rows, nr_cols, avg_nnz_per_row, std_nnz_per_row, distribution, seed, placement, bw, skew, avg_num_neighbours, cross_row_similarity);
	// 		if (i < argc)
	// 			snprintf(matrix_name, sizeof(matrix_name), "%s_artificial", argv[i++]);
	// 		else
	// 			snprintf(matrix_name, sizeof(matrix_name), "%d_%d_%d_%g_%g_%g_%g", AM->nr_rows, AM->nr_cols, AM->nr_nzeros, AM->avg_bw, AM->std_bw, AM->avg_sc, AM->std_sc);
	// 	);
	// 	printf("time generate artificial matrix: %lf\n", time);

	// 	csr_m = AM->nr_rows;
	// 	csr_k = AM->nr_cols;
	// 	csr_nnz = AM->nr_nzeros;

	// 	csr_ia = (typeof(csr_ia)) aligned_alloc(64, (csr_m+1 + VECTOR_ELEM_NUM) * sizeof(*csr_ia));
	// 	#pragma omp parallel for
	// 	for (long i=0;i<csr_m+1;i++)
	// 		csr_ia[i] = AM->row_ptr[i];
	// 	free(AM->row_ptr);
	// 	AM->row_ptr = NULL;

	// 	csr_a = (typeof(csr_a)) aligned_alloc(64, (csr_nnz + VECTOR_ELEM_NUM) * sizeof(*csr_a));
	// 	csr_ja = (typeof(csr_ja)) aligned_alloc(64, (csr_nnz + VECTOR_ELEM_NUM) * sizeof(*csr_ja));
	// 	#pragma omp parallel for
	// 	for (long i=0;i<csr_nnz;i++)
	// 	{
	// 		csr_a[i] = AM->values[i];
	// 		csr_ja[i] = AM->col_ind[i];
	// 	}
	// 	free(AM->values);
	// 	AM->values = NULL;
	// 	free(AM->col_ind);
	// 	AM->col_ind = NULL;
	// }
	for( n=num_cols ; n <num_cols + 1; n*=4){
		x_ref = (typeof(x_ref)) aligned_alloc(64, csr_k_k * n * sizeof(*x_ref));
		x = (typeof(x)) aligned_alloc(64, csr_k_k * n * sizeof(*x));
		// z_ref = (typeof(z_ref)) aligned_alloc(64, csr_m * n * sizeof(*z_ref));
		// z = (typeof(z)) aligned_alloc(64, csr_m * n * sizeof(*z));
		// printf("HERE\n");
		mask= create_mask(sparse_attention_type, csr_m_k, sparsity, band_size);
		// printf("MASK CREATION\n");
		#pragma omp parallel for
		for(int i=0;i<csr_k_k * n;++i)
		{
			x_ref[i] = 0.1;
			x[i] = x_ref[i];
			// z_ref[i] = 0.1;
			// z[i] = z_ref[i];
		}
		y = (typeof(y)) aligned_alloc(64, mask->nnz * sizeof(sizeof(*y)));
		y_final = (typeof(y_final)) aligned_alloc(64, mask->m*n * sizeof(sizeof(*y_final)));
		K = (typeof(K)) aligned_alloc(64, csr_m_k*n * sizeof(sizeof(*K)));
		Q = (typeof(Q)) aligned_alloc(64, csr_m_q*n * sizeof(sizeof(*Q)));
		V = (typeof(V)) aligned_alloc(64, csr_m_v*n * sizeof(sizeof(*V)));
		#pragma omp parallel for
		for(long i=0;i<mask->nnz;i++)
			y[i] = 0.0;
		#pragma omp parallel for
		for(long i=0;i<mask->m*n;i++)
			y_final[i] = 0.0;
		#pragma omp parallel for
		for(long i=0;i<csr_m_k*n;i++)
			K[i] = 0.0;
		#pragma omp parallel for
		for(long i=0;i<csr_m_q*n;i++)
			Q[i] = 0.0;
		#pragma omp parallel for
		for(long i=0;i<csr_m_v*n;i++)
			V[i] = 0.0;
		#if 0
			_Pragma("omp parallel")
			{
				int tnum = omp_get_thread_num();
				long i;
				long i_per_t = csr_k / num_threads;
				long i_s = tnum * i_per_t;

				// No operations.
				// _Pragma("omp parallel for")
				// for (i=0;i<csr_m+1;i++)
					// csr_ia[i] = 0;

				_Pragma("omp parallel for")
				for (i=0;i<csr_nnz;i++)
				{
					csr_ja_k[i] = 0;                      // idx0 - Remove X access pattern dependency.
					// csr_ja[i] = i % csr_n;              // idx_serial - Remove X access pattern dependency.
					// csr_ja[i] = i_s + (i % i_per_t);    // idx_t_local - Remove X access pattern dependency.
				}
			}
		#endif

		long buf_n = strlen(matrix_name) + 1 + 1000;
		char buf[buf_n];
		char * path, * filename, * filename_base;
		str_path_split_path(matrix_name, strlen(matrix_name) + 1, buf, buf_n, &path, &filename);
		path = strdup(path);
		filename = strdup(filename);
		str_path_split_ext(filename, strlen(filename) + 1, buf, buf_n, &filename_base, NULL);
		filename_base = strdup(filename_base);
		if (0)
		{
			long num_pixels = 1024;
			long num_pixels_x = (csr_k_k < num_pixels) ? csr_k_k : num_pixels;
			long num_pixels_y = (csr_m_k < num_pixels) ? csr_m_k : num_pixels;

			printf("ploting : %s\n", filename_base);
			csr_plot(filename_base, csr_ia_k, csr_ja_k, csr_a_k, csr_m_k, csr_k_k, csr_nnz_k, 0, num_pixels_x, num_pixels_y);
			return 0;
		}


		long split_matrix = 0;
		long nnz_per_row_cutoff = 50;

		ValueType * gpu_csr_a = NULL;
		INT_T * gpu_csr_ia = NULL;
		INT_T * gpu_csr_ja = NULL;
		INT_T gpu_csr_nnz = 0;
		if (split_matrix)
		{
			// long k;
			// long degree;
			// gpu_csr_ia = (typeof(gpu_csr_ia)) aligned_alloc(64, (csr_m+1 + VECTOR_ELEM_NUM) * sizeof(*gpu_csr_ia));
			// gpu_csr_a = (typeof(gpu_csr_a)) aligned_alloc(64, (csr_nnz + VECTOR_ELEM_NUM) * sizeof(*gpu_csr_a));
			// gpu_csr_ja = (typeof(gpu_csr_ja)) aligned_alloc(64, (csr_nnz + VECTOR_ELEM_NUM) * sizeof(*gpu_csr_ja));
			// k = 0;
			// gpu_csr_ia[0] = 0;
			// for (i=0;i<csr_m+1;i++)
			// {
			// 	degree = csr_ia[i+1] - csr_ia[i];
			// 	if (degree > nnz_per_row_cutoff)
			// 	{
			// 		for (j=csr_ia[i];j<csr_ia[i+1];j++,k++)
			// 		{
			// 			gpu_csr_ja[k] = csr_ja[j];
			// 			gpu_csr_a[k] = csr_a[j];
			// 		}
			// 		gpu_csr_ia[i+1] = k;
			// 	}
			// 	else
			// 	{
			// 		gpu_csr_ia[i+1] = gpu_csr_ia[i];
			// 	}
			// }
			// gpu_csr_nnz = k;
			// printf("GPU part nnz = %d (%.2lf%%)\n", gpu_csr_nnz, ((double) gpu_csr_nnz) / csr_nnz * 100);
		}
		// printf("hi\n");
		time = time_it(1,
			if (split_matrix)
			{
				// MF = csr_to_format(gpu_csr_ia, gpu_csr_ja, gpu_csr_a, csr_m, csr_k, gpu_csr_nnz, n);
			}
			else
			{
			// for (long i = 0; i < mask->m; ++i) {
			// 	for (long j = mask->csr_ia[i]; j < mask->csr_ia[i+1]; ++j) {
			// 		// if (j>csr->m)
			// 			// printf("%ld \n", j);
			// 		// csr->Mask.insert({i,col_ind[j]}, values[j]);
			// 		printf("%ld ", mask->csr_ja[j]);
			// 	}
			// }
				MF = csr_to_format(mask->csr_ia, mask->csr_ja, mask->csr_a, mask->m, mask->nnz, n);
			}
		);
		printf("time convert to format: %lf\n", time);

		long min_num_loops;
		min_num_loops = 1;

		prefetch_distance = 1;
		time = time_it(1,
			// for (i=0;i<5;i++)
			{
				// printf("prefetch_distance = %d\n", prefetch_distance);
				if (split_matrix)
				{
					// compute(matrix_name,
					// 	gpu_csr_ia, gpu_csr_ja, gpu_csr_a, csr_a_ref, csr_m, csr_k, n, gpu_csr_nnz,
					// 	MF, mask, AM, x, x_ref, z_ref, y, min_num_loops, 0);
				}
				else
				{
					compute(matrix_name,
						csr_ia_k, csr_ja_k, csr_a_k, csr_a_ref_k, csr_m_k, csr_k_k, n, csr_nnz_k,
						csr_ia_q, csr_ja_q, csr_a_q, csr_a_ref_q, csr_m_q, csr_k_q, csr_nnz_q,
						csr_ia_v, csr_ja_v, csr_a_v, csr_a_ref_v, csr_m_v, csr_k_v, csr_nnz_v,
						K, Q, V,
						MF, mask, AM, x, x_ref, y, y_final, min_num_loops, 0);
				}
				prefetch_distance++;
			}
		);
		if (atoi(getenv("COOLDOWN")) == 1)
		{
			printf("time total = %g, sleeping\n", time);
			usleep((long) (time * 1000000));
		}
	}
	free_csr_matrix(AM);
	free(x);
	free(y);
	// free(z);
	free(y_final);
	free(K);
	free(Q);
	free(V);
	free(x_ref);
	delete MF;
	delete mask;
	return 0;
}

