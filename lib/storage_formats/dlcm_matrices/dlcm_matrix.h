#ifndef DLCM_MATRIX_H
#define DLCM_MATRIX_H

#include "macros/cpp_defines.h"
#include <complex.h>
#ifdef __cplusplus
	#define complex  _Complex
#endif


#ifndef MATRIX_MARKET_FLOAT_T
	#define MATRIX_MARKET_FLOAT_T  double
#endif


/*
 * Matrix in DLCM-CSR format.
 *
 * field: weight type -> real, integer, complex, pattern (none)
 *
 * m: num rows
 * k: num columns
 * nnz: num of non-zeros
 *
 * R: row indexes
 * C: column indexes
 * V: values
 */
struct DLCM_Matrix {
	char * filename;

	char * format;
	char * field;
	int symmetric;
	int skew_symmetric;

	long m;
	long k;
	long nnz;
	long nnz_sym;

	int * R;
	int * C;

	void * V;
};


void smtx_init(struct DLCM_Matrix * obj);
struct DLCM_Matrix * smtx_new();
void smtx_clean(struct DLCM_Matrix * obj);
void smtx_destroy(struct DLCM_Matrix ** obj_ptr);


double (* smtx_functor_get_value(struct DLCM_Matrix * MTX)) (void *, long);

struct DLCM_Matrix * smtx_read(char * filename, long expand_symmetry, long pattern_dummy_vals);
// void smtx_write(struct DLCM_Matrix * MTX, char * filename);
void smtx_plot(struct DLCM_Matrix * MTX, char * filename);
void smtx_plot_density(struct DLCM_Matrix * MTX, char * filename);


#endif /* DLCM_MATRIX_H */

