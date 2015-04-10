#include <stdio.h>
#include <pmmintrin.h>
#include <emmintrin.h> 
#include <xmmintrin.h> 


// Sea A0, A1, A2, A3 y B0, B1, B2, B3
// Retorna A0 B0 A1 B1
__m128 crossShuffle1(__m128 A,__m128 B){
	__m128 H1=_mm_shuffle_ps(A, B, _MM_SHUFFLE(1,0,1,0)); 
	H1=_mm_shuffle_ps(H1, H1, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L2 H2
	return H1;
};

void bitonicMergeNetwork(__m128 * entrada1,__m128 * entrada2){
	__m128 L1=_mm_min_ps ( *entrada1 , *entrada2);
	__m128 H1=_mm_max_ps ( *entrada1 , *entrada2);

	/////////////////////////////////////////////////////////////////
	__m128 L1p=_mm_shuffle_ps(L1, H1, _MM_SHUFFLE(2,0,2,0)); 
	L1p=_mm_shuffle_ps(L1p, L1p, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L2 H2
	////////////////////////////////////////////////////////////////
	__m128 L1c=_mm_shuffle_ps(L1p, L1p, _MM_SHUFFLE(1,0,3,2));
	__m128 Aux1L=_mm_min_ps ( L1p , L1c);
	__m128 Aux1H=_mm_max_ps ( L1p , L1c);
	//Se optimiza utilizando funcion//////////////////////////////////
	//L2p=_mm_shuffle_ps(Aux1L, Aux1H, _MM_SHUFFLE(1,0,1,0));
	//L2p=_mm_shuffle_ps(L2p, L2p, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L1 H1
	////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////
	__m128 H1p=_mm_shuffle_ps(L1, H1, _MM_SHUFFLE(3,1,3,1)); 
	H1p=_mm_shuffle_ps(H1p, H1p, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L2 H2
	////////////////////////////////////////////////////////////////
	__m128 H1c=_mm_shuffle_ps(H1p, H1p, _MM_SHUFFLE(1,0,3,2));
	__m128 Aux2L=_mm_min_ps ( H1p , H1c);
	__m128 Aux2H=_mm_max_ps ( H1p , H1c);

	//Se optimiza utilizando funcion//////////////////////////////////
	//__m128 H2p=_mm_shuffle_ps(Aux2L, Aux2H, _MM_SHUFFLE(1,0,1,0));
	//H2p=_mm_shuffle_ps(H2p, H2p, _MM_SHUFFLE(3,1,2,0)); //L0 H0 L1 H1
	//////////////////////////////////////////////////////////////////

	//L3 y H3
	L1=_mm_min_ps ( crossShuffle1(Aux1L,Aux1H), crossShuffle1(Aux2L,Aux2H));
	H1=_mm_max_ps ( crossShuffle1(Aux1L,Aux1H), crossShuffle1(Aux2L,Aux2H));

	//////////////////////////////////////////////////////////////////
	*entrada1=_mm_shuffle_ps(L1, H1, _MM_SHUFFLE(1,0,1,0));
	*entrada1=_mm_shuffle_ps(*entrada1, *entrada1, _MM_SHUFFLE(3,1,2,0));
	*entrada2=_mm_shuffle_ps(L1, H1, _MM_SHUFFLE(3,2,3,2));
	*entrada2=_mm_shuffle_ps(*entrada2, *entrada2, _MM_SHUFFLE(3,1,2,0)); 
}

void inRegisterSort(__m128 * entrada1,__m128 * entrada2, __m128 * entrada3,__m128 * entrada4){
	//paso 1
	__m128 minP1=_mm_min_ps ( *entrada2 , *entrada4); //A
	__m128 maxP1=_mm_max_ps ( *entrada2 , *entrada4); //B
	//paso2
	__m128 minP2=_mm_min_ps ( *entrada1 , *entrada3); //C
	__m128 maxP2=_mm_max_ps ( *entrada1 , *entrada3); //D
	//paso 3
	*entrada1=_mm_min_ps (  minP2 , minP1); 
	__m128 maxCxAP3=_mm_max_ps (  minP2 , minP1); 
	__m128 minDxBP3=_mm_min_ps ( maxP2 , maxP1); //C
	*entrada4=_mm_max_ps ( maxP2 , maxP1); //D
	//paso4
	*entrada2=_mm_min_ps ( maxCxAP3 , minDxBP3);
	*entrada3=_mm_max_ps ( maxCxAP3 , minDxBP3);

	_MM_TRANSPOSE4_PS(*entrada1, *entrada2, *entrada3, *entrada4);

};

void secondReverseBMN(__m128 * entrada1,__m128 * entrada2){
	*entrada2=_mm_shuffle_ps(*entrada2, *entrada2, _MM_SHUFFLE(0,1,2,3));
	bitonicMergeNetwork(entrada1,entrada2);
};

//Sean  las entradas 1 y 2 las menores de sus conjuntos
void mergeSIMD(__m128 * entrada1,__m128 * entrada2, __m128 * entrada3,__m128 * entrada4){
	//Ahora la menor de todas es entrada1
	//se compara entrada1 y entrada 2 obteniendo el menor en entrada 1	
	secondReverseBMN(entrada1, entrada2);
	//Luego la entrada2 se debe comparar con la la menor de las mayores, que es..
	//Si el primer elemento de la entrada 3 es menor que el primer de entrada 4
	if (_mm_ucomile_ss (*entrada3,*entrada4)==1){
		//Ahora el segundo menor es entrada3, quedando en la cola entrada2
		secondReverseBMN(entrada2, entrada3);
		//Se agrega la entrada3 en la BMN con entrada 4 y se obtiene 
		secondReverseBMN(entrada3, entrada4);
	}

	//Si el primer elemento de la entrada 4 es menor que el primer de entrada 2
	else{
		//Ahora el segundo menor es entrada2, quedando en la cola entrada4
		secondReverseBMN(entrada2, entrada4);
		//Se agrega la entrada4 en la BMN con entrada 2 y se obtiene 
		secondReverseBMN(entrada3, entrada4);
	}
};

int main( )
{
__m128 entrada1 , entrada2, entrada3, entrada4;

//Test de la funcion bitonicMergeNetwork
//float a[4] __attribute__((aligned(16))) = {  4, 12, 13,21};
//float b[4] __attribute__((aligned(16))) = { 9, 8, 7,6 };


float a[4] __attribute__((aligned(16))) = {  12, 21, 4,13};

float b[4] __attribute__((aligned(16))) = { 9, 8, 6,7 };

float c[4] __attribute__((aligned(16))) = {  1, 14, 3,0};

float d[4] __attribute__((aligned(16))) = { 5, 11, 15,10 };

entrada1 = _mm_load_ps(a);
entrada2 = _mm_load_ps(b);
entrada3 = _mm_load_ps(c);
entrada4 = _mm_load_ps(d);

//Procedimiento algoritmo
//Con entradas de 4 en 4
inRegisterSort(&entrada1,&entrada2,&entrada3,&entrada4);
//Luego se obtienen 2 conjuntos (de 8) ordenados usando la BMN 2 veces
secondReverseBMN(&entrada1, &entrada2);
secondReverseBMN(&entrada3, &entrada4);
//Luego se utiliza MergeSimd con los dos conjuntos de 8
mergeSIMD(&entrada1,&entrada3,&entrada2,&entrada4);

//El merge SIMD ordenada pero deja el segundo y tercer registro intercambiados
_mm_store_ps(a, entrada1);
_mm_store_ps(c, entrada2);
_mm_store_ps(b, entrada3);
_mm_store_ps(d, entrada4);

printf("Result: %5.f %5.f %5.f %5.f\n", a[0], a[1], a[2], a[3] );
printf("Result: %5.f %5.f %5.f %5.f\n", b[0], b[1], b[2], b[3] );
printf("Result: %5.f %5.f %5.f %5.f\n", c[0], c[1], c[2], c[3] );
printf("Result: %5.f %5.f %5.f %5.f\n", d[0], d[1], d[2], d[3] );
}