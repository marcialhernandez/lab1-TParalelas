#include<stdio.h>
#include<emmintrin.h>

int main(void){
	double A[4] __attribute__((aligned(16))) = { 2.0, 1.0, 1.0, 3.0};
	double B[4] __attribute__((aligned(16))) = { 3.4, 2.8, 1.2, 0.5};
	double C[4] __attribute__((aligned(16))) = { 0.0, 0.0, 0.0, 0.0};

int lda=2, i=0;
__m128d a1, a2, b, c1, c2;

c1 = _mm_load_pd(C+0*lda);
c2 = _mm_load_pd(C+1*lda);

for (i=0; i< 2; i++) {
	a1 = _mm_load1_pd(A + i + 0*lda);
	a2 = _mm_load1_pd(A + i + 1*lda);
	b = _mm_load_pd(B + i*lda);
	c1 = _mm_add_pd(c1, _mm_mul_pd(a1, b));
	c2 = _mm_add_pd(c2, _mm_mul_pd(a2, b));
}

_mm_store_pd(C+0*lda, c1);
_mm_store_pd(C+1*lda, c2);
printf("%g\t%g\t%g\t%g\n", C[0], C[1], C[2], C[3]);
}