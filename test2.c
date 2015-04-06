#include <stdio.h>
#include <pmmintrin.h>
#include <emmintrin.h>


int main( )
{
__m128 intrinA , intrinB, salida;

float a[4] __attribute__((aligned(16))) = {  4, 12, 13,21};

float b[4] __attribute__((aligned(16))) = { 9, 8, 7,6 };


intrinA = _mm_load_ps(a);
intrinB = _mm_load_ps(b);

printf("Loading %5.3f %5.3f %5.3f %5.3f into XMM register.\n",a[0], a[1], a[2], a[3] );
printf("Loading %5.3f %5.3f %5.3f %5.3f into XMM register.\n", b[0], b[1], b[2], b[3] );


__m128 L1=_mm_min_ps ( intrinA , intrinB);
__m128 H1=_mm_max_ps ( intrinA , intrinB);

/////////////////////////////////////////////////////////////////
__m128 L1p=_mm_shuffle_ps(L1, H1, _MM_SHUFFLE(2,0,2,0)); 
L1p=_mm_shuffle_ps(L1p, L1p, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L2 H2
////////////////////////////////////////////////////////////////
__m128 L1c=_mm_shuffle_ps(L1p, L1p, _MM_SHUFFLE(1,0,3,2));
__m128 Aux1L=_mm_min_ps ( L1p , L1c);
__m128 Aux1H=_mm_max_ps ( L1p , L1c);
__m128 L2p=_mm_shuffle_ps(Aux1L, Aux1H, _MM_SHUFFLE(1,0,1,0));
L2p=_mm_shuffle_ps(L2p, L2p, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L1 H1
////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
__m128 H1p=_mm_shuffle_ps(L1, H1, _MM_SHUFFLE(3,1,3,1)); 
H1p=_mm_shuffle_ps(H1p, H1p, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L2 H2
////////////////////////////////////////////////////////////////
__m128 H1c=_mm_shuffle_ps(H1p, H1p, _MM_SHUFFLE(1,0,3,2));
__m128 Aux2L=_mm_min_ps ( H1p , H1c);
__m128 Aux2H=_mm_max_ps ( H1p , H1c);
__m128 H2p=_mm_shuffle_ps(Aux2L, Aux2H, _MM_SHUFFLE(1,0,1,0));
H2p=_mm_shuffle_ps(H2p, H2p, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L1 H1
//////////////////////////////////////////////////////////////////

__m128 L3=_mm_min_ps ( L2p , H2p);
__m128 H3=_mm_max_ps ( L2p , H2p);

//////////////////////////////////////////////////////////////////
__m128 S1=_mm_shuffle_ps(L3, H3, _MM_SHUFFLE(1,0,1,0));
S1=_mm_shuffle_ps(S1, S1, _MM_SHUFFLE(3,1,2,0));
__m128 S2=_mm_shuffle_ps(L3, H3, _MM_SHUFFLE(3,2,3,2));
S2=_mm_shuffle_ps(S2, S2, _MM_SHUFFLE(3,1,2,0));

_mm_store_ps(a, S1);
_mm_store_ps(b, S2);
printf("Result: %5.f %5.f %5.f %5.f\n", a[0], a[1], a[2], a[3] );
printf("Result: %5.f %5.f %5.f %5.f\n", b[0], b[1], b[2], b[3] );

}