/**
 * This is OpenBLAS Test Code
 * OpenBLAS User Manual: https://github.com/xianyi/OpenBLAS/wiki/User-Manual
 * Original: https://gist.github.com/xianyi/6930656
 **/

#include <cblas.h>
#include <stdio.h>

int main(){
	int i=0;
	double A[6] = {
		 1.0,  2.0,  1.0,
		-3.0,  4.0, -1.0
	};
	double B[6] = {
		 1.0,  2.0,  1.0,
		-3.0,  4.0, -1.0
	};
	double C[9] = {
		 0.5,  0.5,  0.5,
		 0.5,  0.5,  0.5,
		 0.5,  0.5,  0.5
	};
	cblas_dgemm(
			CblasColMajor,
			CblasNoTrans,	// opeA
			CblasTrans,		// opeB
			3, 3, 2,	// colA, rowB, rowA(or colB)
			1,		// alpha
			A, 3,	// A, ldA
			B, 3,	// B, ldB
			2,		// beta
			C, 3	// C, ldC
		);

	for(i=0; i<9; i++) printf("%lf ", C[i]);
	printf("\n");

	return 0;
}
