#include <stdio.h>
#include <pmmintrin.h>
#include <emmintrin.h>
int main( )
{
__m128 intrinA , intrinB, salida;

float a[4] __attribute__((aligned(16))) = { 12,21, 4, 13 };

float b[4] __attribute__((aligned(16))) = { 9, 8, 6,7 };

printf("Loading %5.3f %5.3f %5.3f %5.3f into XMM register.\n",a[0], a[1], a[2], a[3] );

intrinA = _mm_load_ps(a);

printf("Loading %5.3f %5.3f %5.3f %5.3f into XMM register.\n", b[0], b[1], b[2], b[3] );

intrinB = _mm_load_ps(b);
//Se invierte la segunda entrada
__m128 B=_mm_shuffle_ps(intrinB, intrinB, _MM_SHUFFLE(0,1,2,3));
__m128 L1=_mm_min_ps ( intrinA , B);
__m128 H1=_mm_max_ps ( intrinA , B);
__m128 L1p=_mm_shuffle_ps(L1, H1, _MM_SHUFFLE(3,2,1,0));
__m128 H1p=_mm_shuffle_ps(L1, H1, _MM_SHUFFLE(0,1,2,3));

__m128 L2=_mm_min_ps ( L1p , H1p);
__m128 H2=_mm_max_ps ( L1p , H1p);
__m128 L2p=_mm_shuffle_ps(L2, H2, _MM_SHUFFLE(2,3,0,1));
__m128 H2p=_mm_shuffle_ps(L2, H2, _MM_SHUFFLE(1,0,2,3));

__m128 L3=_mm_min_ps ( L2p , H2p);
__m128 H3=_mm_max_ps ( L2p , H2p);
__m128 salida1=_mm_shuffle_ps(L3, H3, _MM_SHUFFLE(3,2,0,1));
__m128 salida2=_mm_shuffle_ps(L3, H3, _MM_SHUFFLE(2,0,1,3));
//salida = _mm_shuffle_ps(L1, H1, _MM_SHUFFLE(0,1,0,0));
_mm_store_ps(a, salida1);
_mm_store_ps(b, salida2);
printf("Result: %5.3f %5.3f %5.3f %5.3f\n", a[0], a[1], a[2], a[3] );
printf("Result: %5.3f %5.3f %5.3f %5.3f\n", b[0], b[1], b[2], b[3] );

}