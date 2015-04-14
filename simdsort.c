/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 * Copyright (C) Marcial Hernandez Sanchez, 2015
 * University of Santiago, Chile (Usach) 
 */

/////////////////////////////////////////
//C
#include <stdio.h>
#include <unistd.h>
//File descriptors from sys - C
#include <sys/types.h>
#include <sys/stat.h>
//System tools to open and write - C
#include <fcntl.h>
//SSE
#include <pmmintrin.h>
#include <emmintrin.h> 
#include <xmmintrin.h> 
//C++
#include <iostream>

using namespace std;

/////////////////////////////////////////

//Funcion que escribe en un archivo de salida de nombre 'nombreSalida'
//La cantidad 'nBytesToWrite' de datos que contiene el buffer de flotantes 'bufferAEscribir'
void sysWrite(string nombreSalida, float * bufferAEscribir, int nBytesToWrite){
	int fd = open(nombreSalida.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU | S_IRWXG | S_IRWXO);
	if ( fd <0)
		{
			cout << "Error: no se puede abrir el archivo de Salida" << endl;
			cout << strerror(errno) << endl;
			exit(1);
		}

	else{
		write(  fd, bufferAEscribir, nBytesToWrite);
		close(fd);
	}
}

//Igual que sysRead, pero deja alineado de a 16 usando posix_memalign
//Y su segundo argumento es la cantidad de numeros que debe leer
float * sysReadAligned(string nombreEntrada, int * readSize){
	int fd = open(nombreEntrada.c_str(), O_RDONLY);
	if ( (fd = open(nombreEntrada.c_str(), O_RDONLY) ) == -1)
		{
			cout << "Error: no se puede abrir el archivo" << endl;
			cout << strerror(errno) << endl;
			exit(1);
		}

	else{
		//En un principio se obtenia el tamaño total del archivo usando fstat
		//struct stat buf;
		//fstat(fd, &buf);
		//*size = buf.st_size;

		//Ahora la validacion se hace en el getopt
		/*if (*size%16 !=0){
			cout << "Error: cantidad de datos invalidos (no es multiplo de 16)" << endl;
			exit(1);
		}*/

		//else{
			//float line[size];
			//float *line=(float *) malloc(*size);
			float *line;
			//size de lectura * 4 pues cada elemento de la lista se compone de 4 bytes
			posix_memalign((void**)&line, 16, *readSize*4);
			read(fd, line, *readSize*4);
			close(fd);
			//cada registro contiene 4 numero flotantes
			//*size=*size/4;
			return line;
		//}
	}
}

//Retorna la lista con todos los numeros cargados tipo float
//Ademas actualiza el valor de entrada size por la cantidad de registros de 128 de la entrada
float * sysRead(string nombreEntrada, int * size){
	int fd = open(nombreEntrada.c_str(), O_RDONLY);
	if ( (fd = open(nombreEntrada.c_str(), O_RDONLY) ) == -1)
		{
			cout << "Error: no se puede abrir el archivo" << endl;
			exit(1);
		}

	else{
		struct stat buf;
		fstat(fd, &buf);
		*size = buf.st_size;
		if (*size%16 !=0){
			cout << "Error: cantidad de datos invalidos (no es multiplo de 16)" << endl;
			exit(1);
		}
		else{
			//float line[size];
			float *line=(float *) malloc(*size);
			read(fd, line, *size);
			close(fd);
			//cada registro contiene 4 numero flotantes
			*size=*size/4;
			return line;
		}
	}
}

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

//Invoca a la BMN, pero revirtiendo los valores del segundo registro
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

void sortKernel(__m128 * entrada1,__m128 * entrada2, __m128 * entrada3,__m128 * entrada4){
	inRegisterSort(entrada1,entrada2,entrada3,entrada4);
//Luego se obtienen 2 conjuntos (de 8) ordenados usando la BMN 2 veces
	secondReverseBMN(entrada1, entrada2);
	secondReverseBMN(entrada3, entrada4);
	//Luego se utiliza MergeSimd con los dos conjuntos de 8
	mergeSIMD(entrada1,entrada3,entrada2,entrada4);
}

void loadSortKernel(float * a, float *b, float * c, float * d){
	__m128 entrada1 = _mm_load_ps(a);
	__m128 entrada2 = _mm_load_ps(b);
	__m128 entrada3 = _mm_load_ps(c);
	__m128 entrada4 = _mm_load_ps(d);

	sortKernel(&entrada1,&entrada2,&entrada3,&entrada4);

	_mm_store_ps(a, entrada1);
	_mm_store_ps(c, entrada2);
	_mm_store_ps(b, entrada3);
	_mm_store_ps(d, entrada4);

	//Debug
	//cout << a[0] << a[1] << a[2] << a[3] << endl;
	//cout << b[0] << b[1] << b[2] << b[3] << endl;
	//cout << c[0] << c[1] << c[2] << c[3] << endl;
	//cout << d[0] << d[1] << d[2] << d[3] << endl;
}

/**
 * merge() 
 * Merge two sorted arrays, A with a  integers and 
 * B with b integers, into a sorted array C.
 * Fuente = http://www.cs.cityu.edu.hk/~lwang/ccs4335/mergesort.c
 **/

 void  merge(float *A, int a, float *B, int b, float *C) {
  int i,j,k;
  i = 0; 
  j = 0;
  k = 0;
  while (i < a && j < b) {
    if (A[i] <= B[j]) {
	  /* copy A[i] to C[k] and move the pointer i and k forward */
	  C[k] = A[i];
	  i++;
	  k++;
    }
    else {
      /* copy B[j] to C[k] and move the pointer j and k forward */
      C[k] = B[j];
      j++;
      k++;
    }
  }
  /* move the remaining elements in A into C */
  while (i < a) {
    C[k]= A[i];
    i++;
    k++;
  }
  /* move the remaining elements in B into C */
  while (j < b)  {
    C[k]= B[j];
    j++;
    k++;
  }
}  

/**
 * merge_sort()
 * Sort array A with n integers, using merge-sort algorithm.
 **/
void merge_sort(float *A, int n) {
  int i;
  float *A1, *A2;
  int n1, n2;

  //Aprovecho el ordenamiento SIMD
  //Funciona de 32 en 32

  if (n<32)//(n < 2)
    return;   /* the array is sorted when n=32.*/
  
  /* divide A into two array A1 and A2 */
  n1 = n / 2;   /* the number of elements in A1 */
  n2 = n - n1;  /* the number of elements in A2 */

  //Se multiplica por 4 para que queden alineados a 16
  A1 = (float*)malloc(sizeof(float)*4 * n1);
  A2 = (float*)malloc(sizeof(float)*4 * n2);
  
  /* move the first n/2 elements to A1 */
  for (i =0; i < n1; i++) {
    A1[i] = A[i];
  }
  /* move the rest to A2 */
  for (i = 0; i < n2; i++) {
    A2[i] = A[i+n1];
  }
  /* recursive call */
  merge_sort(A1, n1);
  merge_sort(A2, n2);

  /* conquer */
  merge(A1, n1, A2, n2, A);
  free(A1);
  free(A2);
}

//Valida que la entrada sea numerica
//retorna 0 y guarda la entrada en entradaPrograma si la entrada es numero
//retorna 1 si hay falla
int isNumber(const string entradaConsola, int * entradaPrograma ){
	int largoEntrada=entradaConsola.size();
	for (int contador=0;contador<largoEntrada;contador++){
		if (!isdigit(entradaConsola.at(contador))){
			return 1;
		}
	}

	*entradaPrograma=stoi(entradaConsola);
	return 0;
}

/* Una cadena que lista las opciones cortas válidas para getOpt
   Se inicia con : pues si falta algun argumento, enviara un caso tipo ":"" */

const char* const opciones = "d:i:o:N:";

//-i : archivo binario con la lista de entrada desordenados
//-o : archivo binario de salida con la lista ordenada
//-N : largo de la lista
//-d : si debug es 0, no se imprime mensaje alguno por stdout, si es 1, se imprime la secuencia final, 1 por linea

/* Declaracion de las banderas */

int banderaErrorParametros=0, banderaErrorBanderas=0;

//
int bandera_i=0, bandera_N=0,bandera_o=0, bandera_d=0;
bool multiplo16=true;

//Las banderas _i, _N, _o y _d: para asegurar que solo haya un argumento
//Por ejemplo; podria escribir por consola -i entrada1 -i entrada2
//si pasa esto, se retornara un mensaje de error y se terminara la ejecucion

int main (int argc, char **argv)
{

	string nombreEntrada, nombreSalida="outputSorted.txt"; //nombre de salida por defecto si no se especifica
	int largoLista, debug=0, argumentoConsola;

	//////////// Analisis de entrada de la consola getOpt ///////////

	while (((argumentoConsola = getopt (argc, argv, opciones)) != -1) &&  banderaErrorParametros==0 && banderaErrorBanderas==0){
		//No tiene caso seguir con el while, si se ha detectado una falla en el camino
		switch (argumentoConsola){  

			case 'i': if (bandera_i==0) { //archivo entrada

					  bandera_i++; 

					  nombreEntrada=optarg;

					}
					else{
					banderaErrorBanderas++;						
					}
					  break;	  
			case 'N': if (bandera_N==0) {
							    bandera_N++;
							    banderaErrorParametros = banderaErrorParametros + isNumber(optarg, &largoLista );
							    if (largoLista%16!=0){
							    	banderaErrorParametros++;
							    	multiplo16=false;
							    }
						    }              
						    else {
							    banderaErrorBanderas++;
						    }
						    break;

			case 'o': if (bandera_o==0) { //archivo salida

					  bandera_o++; 

					  nombreSalida=optarg;

					}
					else{
					banderaErrorBanderas++;						
					}
					  break;

			case 'd': if (bandera_d==0) { //archivo

					  bandera_d++; 

					  banderaErrorParametros = banderaErrorParametros + isNumber(optarg, &debug);
					  if ( debug >1){
					  	banderaErrorParametros++;
					  }
					}
					else{
					banderaErrorBanderas++;						
					}
					  break;

					  case ':': banderaErrorParametros++; break;
					  case '?': 
						    if ((optopt=='i' || optopt=='o' || optopt=='N' || optopt=='d' || optopt=='t')){
							    banderaErrorParametros++;
						    }
						    else{
							    banderaErrorBanderas++;
						    }
						    break;
					  default: banderaErrorBanderas++; break;
				  }
		}

	/////////////////////////////////////////////////////////////////


	//////////// Analisis de los argumentos obtenidos de getOpt ///////////

	if (banderaErrorBanderas>0){
		cout << "Error: una o más opciones estan duplicadas o no estan disponibles" << endl;
		exit(1);
	}

	if (bandera_i==0){
		cout << "Error: no se ha especificado archivo de entrada" << endl;
		exit(1);
	}

	if (bandera_N==0){
		cout << "Error: no se ha indicado el largo de la lista de entrada" << endl;
		exit(1);
	}

	if (banderaErrorParametros>0){

		if (multiplo16==false){
			cout << "Error: el largo de la lista especificado como argumento de la opcion -N" << endl;
			cout << "No es divisible por 16" << endl;
			exit(1);
		}

		else{
		cout << "Error: uno o más argumentos de alguna de las opciones no son validos" << endl;
		exit(1);
		}
	}

	/////////////////////////////////////////////////////////////////

	////////////////MultiWay Merge Sort//////////////////////////////

	//Buffer donde se guardan los floats de entrada
	float *line =sysReadAligned(nombreEntrada,&largoLista);

	int offset;
	//Se divide en 16 pues se visitan de a 16 
	largoLista=largoLista/16;
	for (int i=0;i<largoLista;i++){
		offset=i*16;

		//A cada grupo de 16 se le aplica el sortKernel
		loadSortKernel(&line[offset], &line[offset+4], &line[offset+8], &line[offset+12]);
	}

	//largoLista * 16 ya que es la cantidad total de registros, y no la cantidad total de grupos de 16
	merge_sort(line, largoLista*16);

	/////////////////////////////////////////////////////////////////

	//Debug
	if (debug==1){
		for (int i=0; i<largoLista*16; i++){
			cout << line[i] << endl;
		}
	}

	//Se escribe en el archivo de salida
	//Considerar que el tamaño es cantidad de grupos de registros, por cantidad de grupos, por cantidad de bytes de cada registro
	sysWrite(nombreSalida, &*line, largoLista*16*4);
	
	return 0;

}