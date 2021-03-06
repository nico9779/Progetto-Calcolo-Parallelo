/******  COMPARE DIFFERENT VERSIONS OF SEQUENTIAL MATRIX MULTIPLICATION ALGORITHM AND STRASSEN ALGORITHM	******/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <xmmintrin.h>

// Size of the sub-matrices
#define CHUNK 32

// Add two square matrices
float* addMatrix(float* M1, float* M2, int n) {

	float* temp = (float*) malloc(n * n * sizeof(float));

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			int index = i*n+j;
			temp[index] = M1[index] + M2[index];
		}
	}

    return temp;
}

// Subtract two square matrices
float* subtractMatrix(float* M1, float* M2, int n) {

	float* temp = (float*) malloc(n * n * sizeof(float));

    for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			int index = i*n+j;
			temp[index] = M1[index] - M2[index];
		}  
	}
        
    return temp;
}

// I-K-J loop
float* multiplyMatrixIKJ(float* A, float* B, int n) {

    float* C = (float*) calloc(n * n, sizeof(float));
    float a = 0;

	for(int i = 0; i < n; i++) {
		for(int k = 0; k < n; k++) {
            a = A[i*n+k];
			for(int j = 0; j < n; j+=8) {
                C[i*n+j] += a * B[k*n+j];
                C[i*n+j+1] += a * B[k*n+j+1];
                C[i*n+j+2] += a * B[k*n+j+2];
                C[i*n+j+3] += a * B[k*n+j+3];
                C[i*n+j+4] += a * B[k*n+j+4];
                C[i*n+j+5] += a * B[k*n+j+5];
                C[i*n+j+6] += a * B[k*n+j+6];
                C[i*n+j+7] += a * B[k*n+j+7];
			}
		}
	}

    return C;
}

// K-I-J loop
float* multiplyMatrixKIJ(float* A, float* B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int k = 0; k < n; k++) {
		for(int i = 0; i < n; i++) {
			a = A[i*n+k];
			for(int j = 0; j < n; j+=8) {
				C[i*n+j] += a * B[k*n+j];
				C[i*n+j+1] += a * B[k*n+j+1];
				C[i*n+j+2] += a * B[k*n+j+2];
				C[i*n+j+3] += a * B[k*n+j+3];
				C[i*n+j+4] += a * B[k*n+j+4];
				C[i*n+j+5] += a * B[k*n+j+5];
				C[i*n+j+6] += a * B[k*n+j+6];
				C[i*n+j+7] += a * B[k*n+j+7];
			}
		}
	}

	return C;
}

// SIMD instructions
void multiplyMatrixSSE(float *A, float *B, float *C, int n) {
    __m128 ma , *mb , *mc = ( __m128 *) C;
    int t1 ,t2;
    int n_4 = (n>>3)<<1;
    for (int k=0; k<n; k++) {
        mb = ( __m128 *) B;
        for (int i=0; i<n; i++, mb+=n_4) {
            ma = _mm_load1_ps(A++); //k*a+i
            for (int j=0; j<n_4; j+=4) {
                mc[j] = _mm_add_ps (mc[j], _mm_mul_ps (ma, mb[j]));
                mc[j+1] = _mm_add_ps (mc[j+1], _mm_mul_ps (ma, mb[j+1]));
                mc[j+2] = _mm_add_ps (mc[j+2], _mm_mul_ps (ma, mb[j+2]));
                mc[j+3] = _mm_add_ps (mc[j+3], _mm_mul_ps (ma, mb[j+3]));
            }
        }
        mc += n_4 ;
    }
}

// Traspose matrix B
float* multiplyMatrixTransposition(float* A, float* B, int n) {

	float* C = (float*) malloc(n * n * sizeof(float));
	float* temp = (float*) malloc(n * n * sizeof(float));
	float local_sum, t1, t2, t3, t4;

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			temp[i*n+j] = B[j*n+i];
		}
	}

	for(int i = 0; i < n; i++) {
		float* p1 = &A[i*n];
		for(int j = 0; j < n; j++) {
			float* p2 = &temp[j*n];
			local_sum = 0;
			for(int k = 0; k < n; k+=4) {
				t1 = *(p1+k) * *(p2+k);
				t2 = *(p1+k+1) * *(p2+k+1);
				t3 = *(p1+k+2) * *(p2+k+2);
				t4 = *(p1+k+3) * *(p2+k+3);
				local_sum += (t1+t2)+(t3+t4);
			}
			C[i*n+j] = local_sum;
		}
	}

	free(temp);

	return C;
}

// DIFFERENT VERSIONS FOR SUBMATRICES SEQUENTIAL MULTIPLICATION

float* multiplyMatrixChunk1(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int kk=0; kk<n; kk+=CHUNK) {
		for(int ii=0; ii<n; ii+=CHUNK) {
			for(int jj=0; jj<n; jj+=CHUNK) {
				for(int k=kk; k<kk+CHUNK; ++k) {
					for(int i=ii; i<ii+CHUNK; ++i) {
						a = A[i*n+k];
						for(int j=jj; j<jj+CHUNK; j+=8) {
							C[i*n+j] += a*B[k*n+j];
							C[i*n+j+1] += a*B[k*n+j+1];
							C[i*n+j+2] += a*B[k*n+j+2];
							C[i*n+j+3] += a*B[k*n+j+3];
							C[i*n+j+4] += a*B[k*n+j+4];
							C[i*n+j+5] += a*B[k*n+j+5];
							C[i*n+j+6] += a*B[k*n+j+6];
							C[i*n+j+7] += a*B[k*n+j+7];
						}
					}
				}
			}
		}
	}

	return C;
}

float* multiplyMatrixChunk2(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int kk=0; kk<n; kk+=CHUNK) {
		for(int i=0; i<n; ++i) {
			for(int k=kk; k<kk+CHUNK; ++k) {
				a = A[i*n+k];
				for(int j=0; j<n; j+=8) {
					C[i*n+j] += a*B[k*n+j];
					C[i*n+j+1] += a*B[k*n+j+1];
					C[i*n+j+2] += a*B[k*n+j+2];
					C[i*n+j+3] += a*B[k*n+j+3];
					C[i*n+j+4] += a*B[k*n+j+4];
					C[i*n+j+5] += a*B[k*n+j+5];
					C[i*n+j+6] += a*B[k*n+j+6];
					C[i*n+j+7] += a*B[k*n+j+7];
				}
			}
		}
	}

	return C;
}

float* multiplyMatrixChunk3(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int kk=0; kk<n; kk+=CHUNK) {
		for(int k=kk; k<kk+CHUNK; ++k) {
			for(int i=0; i<n; ++i) {
				a = A[i*n+k];
				for(int j=0; j<n; j+=8) {
					C[i*n+j] += a*B[k*n+j];
					C[i*n+j+1] += a*B[k*n+j+1];
					C[i*n+j+2] += a*B[k*n+j+2];
					C[i*n+j+3] += a*B[k*n+j+3];
					C[i*n+j+4] += a*B[k*n+j+4];
					C[i*n+j+5] += a*B[k*n+j+5];
					C[i*n+j+6] += a*B[k*n+j+6];
					C[i*n+j+7] += a*B[k*n+j+7];
				}
			}
		}         
	}

	return C;
}

float* multiplyMatrixChunk4(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int ii=0; ii<n; ii+=CHUNK) {
		for(int i=ii; i<ii+CHUNK; ++i) {
			for(int k=0; k<n; ++k) {
				a = A[i*n+k];
				for(int j=0; j<n; j+=8) {
					C[i*n+j] += a*B[k*n+j];
					C[i*n+j+1] += a*B[k*n+j+1];
					C[i*n+j+2] += a*B[k*n+j+2];
					C[i*n+j+3] += a*B[k*n+j+3];
					C[i*n+j+4] += a*B[k*n+j+4];
					C[i*n+j+5] += a*B[k*n+j+5];
					C[i*n+j+6] += a*B[k*n+j+6];
					C[i*n+j+7] += a*B[k*n+j+7];
				}
			}
		}         
	}

	return C;
}

float* multiplyMatrixChunk5(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int jj=0; jj<n; jj+=CHUNK) {
		for(int i=0; i<n; ++i) {
			for(int k=0; k<n; ++k) {
				a = A[i*n+k];
				for(int j=jj; j<jj+CHUNK; j+=8) {
					C[i*n+j] += a*B[k*n+j];
					C[i*n+j+1] += a*B[k*n+j+1];
					C[i*n+j+2] += a*B[k*n+j+2];
					C[i*n+j+3] += a*B[k*n+j+3];
					C[i*n+j+4] += a*B[k*n+j+4];
					C[i*n+j+5] += a*B[k*n+j+5];
					C[i*n+j+6] += a*B[k*n+j+6];
					C[i*n+j+7] += a*B[k*n+j+7];
				}
			}
		}         
	}

	return C;
}

float* multiplyMatrixChunk6(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int kk=0; kk<n; kk+=CHUNK) {
		for(int jj=0; jj<n; jj+=CHUNK) {
			for(int i=0; i<n; ++i) {
				for(int k=kk; k<kk+CHUNK; ++k) {
					a = A[i*n+k];
					for(int j=jj; j<jj+CHUNK; j+=8) {
						C[i*n+j] += a*B[k*n+j];
						C[i*n+j+1] += a*B[k*n+j+1];
						C[i*n+j+2] += a*B[k*n+j+2];
						C[i*n+j+3] += a*B[k*n+j+3];
						C[i*n+j+4] += a*B[k*n+j+4];
						C[i*n+j+5] += a*B[k*n+j+5];
						C[i*n+j+6] += a*B[k*n+j+6];
						C[i*n+j+7] += a*B[k*n+j+7];
					}
				}
			}         
		}
	}

	return C;
}

float* multiplyMatrixChunk7(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int ii=0; ii<n; ii+=CHUNK) {
		for(int kk=0; kk<n; kk+=CHUNK) {
			for(int i=ii; i<ii+CHUNK; ++i) {
				for(int k=kk; k<kk+CHUNK; ++k) {
					a = A[i*n+k];
					for(int j=0; j<n; j+=8) {
						C[i*n+j] += a*B[k*n+j];
						C[i*n+j+1] += a*B[k*n+j+1];
						C[i*n+j+2] += a*B[k*n+j+2];
						C[i*n+j+3] += a*B[k*n+j+3];
						C[i*n+j+4] += a*B[k*n+j+4];
						C[i*n+j+5] += a*B[k*n+j+5];
						C[i*n+j+6] += a*B[k*n+j+6];
						C[i*n+j+7] += a*B[k*n+j+7];
					}
				}
			}         
		}
	}

	return C;
}

float* multiplyMatrixChunk8(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int kk=0; kk<n; kk+=CHUNK) {
		for(int ii=0; ii<n; ii+=CHUNK) {
			for(int k=kk; k<kk+CHUNK; ++k) {
				for(int i=ii; i<ii+CHUNK; ++i) {
					a = A[i*n+k];
					for(int j=0; j<n; j+=8) {
						C[i*n+j] += a*B[k*n+j];
						C[i*n+j+1] += a*B[k*n+j+1];
						C[i*n+j+2] += a*B[k*n+j+2];
						C[i*n+j+3] += a*B[k*n+j+3];
						C[i*n+j+4] += a*B[k*n+j+4];
						C[i*n+j+5] += a*B[k*n+j+5];
						C[i*n+j+6] += a*B[k*n+j+6];
						C[i*n+j+7] += a*B[k*n+j+7];
					}
				}
			}         
		}
	}

	return C;
}

float* multiplyMatrixChunk9(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int ii=0; ii<n; ii+=CHUNK) {
		for(int kk=0; kk<n; kk+=CHUNK) {
			for(int jj=0; jj<n; jj+=CHUNK) {
				for(int i=ii; i<ii+CHUNK; ++i) {
					for(int k=kk; k<kk+CHUNK; ++k) {
						a = A[i*n+k];
						for(int j=jj; j<jj+CHUNK; j+=8) {
							C[i*n+j] += a*B[k*n+j];
							C[i*n+j+1] += a*B[k*n+j+1];
							C[i*n+j+2] += a*B[k*n+j+2];
							C[i*n+j+3] += a*B[k*n+j+3];
							C[i*n+j+4] += a*B[k*n+j+4];
							C[i*n+j+5] += a*B[k*n+j+5];
							C[i*n+j+6] += a*B[k*n+j+6];
							C[i*n+j+7] += a*B[k*n+j+7];
						}
					}
				}
			}
		}
	}

	return C;
}

float* multiplyMatrixChunk10(float *A, float *B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
	float a = 0;

	for(int kk=0; kk<n; kk+=CHUNK) {
		for(int ii=0; ii<n; ii+=CHUNK) {
			for(int i=ii; i<ii+CHUNK; ++i) {
				for(int k=kk; k<kk+CHUNK; ++k) {
					a = A[i*n+k];
					for(int j=0; j<n; j+=8) {
						C[i*n+j] += a*B[k*n+j];
						C[i*n+j+1] += a*B[k*n+j+1];
						C[i*n+j+2] += a*B[k*n+j+2];
						C[i*n+j+3] += a*B[k*n+j+3];
						C[i*n+j+4] += a*B[k*n+j+4];
						C[i*n+j+5] += a*B[k*n+j+5];
						C[i*n+j+6] += a*B[k*n+j+6];
						C[i*n+j+7] += a*B[k*n+j+7];
					}
				}
			}
		}
	}

	return C;
}

void multiplyMatrixChunk11(float **A, float **B, float **C, int n) {
	float *At1, *Bt1, *Ct1;
	float *At2, *Bt2, *Ct2;
	for (int k=0; k<n; k+=CHUNK) {
		for (int i=0; i<n; i+=CHUNK) {
			At1 = A[i]+k;
			for (int j=0; j<n; j+=CHUNK) {
				Bt1 = B[k]+j;
				Ct1 = C[i]+j;
				for (int k1=0; k1<CHUNK; k1++, Bt1+=n, Ct1+=n) {
					At2 = At1+k1;
					int i2,i3;
					for (i2=0; i2<CHUNK; i2++, At2+=n) {
						float Ac = *At2;
						for (i3 =0; i3 <CHUNK; i3++) {
							Ct1[i3]+=Ac* Bt1[i3];
						}
					}
				}
			}
		}
	}
}

// Print square matrix in output
void printMatrix(float* M, int n) {

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			printf("%f  ", M[i*n+j]);
		}
		printf("\n");
	}

	printf("\n");
}

// Strassen algorithm using SIMD instructions
float* strassenMatrixSSE(float* A, float* B, int n) {

    // Initialize matrix C to return in output (C = A*B)
	float* C;

	// Base case
	if(n <= 64) {
        C = (float*) calloc(n * n, sizeof(float));
    	multiplyMatrixSSE(A, B, C, n);
        return C;
	}

    C = (float*) malloc(n * n * sizeof(float));

	// Dimension of the matrices is the half of the input size
	int k = n/2;

	// Decompose A and B into 8 submatrices
	float* A11 = (float*) malloc(k * k * sizeof(float));
	float* A12 = (float*) malloc(k * k * sizeof(float));
	float* A21 = (float*) malloc(k * k * sizeof(float));
	float* A22 = (float*) malloc(k * k * sizeof(float));
	float* B11 = (float*) malloc(k * k * sizeof(float));
	float* B12 = (float*) malloc(k * k * sizeof(float));
	float* B21 = (float*) malloc(k * k * sizeof(float));
	float* B22 = (float*) malloc(k * k * sizeof(float));

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			int index = i*k+j;
			int index_1 = i*n+j;
			int index_2 = i*n+k+j;
			int index_3 = (k+i)*n+j;
			int index_4 = (k+i)*n+k+j;
			A11[index] = A[index_1];
			A12[index] = A[index_2];
			A21[index] = A[index_3];
			A22[index] = A[index_4];
			B11[index] = B[index_1];
			B12[index] = B[index_2];
			B21[index] = B[index_3];
			B22[index] = B[index_4];
    	}
	}

	// Create support matrices in order to calculate Strassen matrices
	float* M1 = subtractMatrix(B12, B22, k);
	float* M2 = addMatrix(A11, A12, k);
	float* M3 = addMatrix(A21, A22, k);
	float* M4 = subtractMatrix(B21, B11, k);
	float* M5 = addMatrix(A11, A22, k);
	float* M6 = addMatrix(B11, B22, k);
	float* M7 = subtractMatrix(A12, A22, k);
	float* M8 = addMatrix(B21, B22, k);
	float* M9 = subtractMatrix(A11, A21, k);
	float* M10 = addMatrix(B11, B12, k);

	// Create the Strassen matrices
	float* P1 = strassenMatrixSSE(A11, M1, k);
	float* P2 = strassenMatrixSSE(M2, B22, k);
	float* P3 = strassenMatrixSSE(M3, B11, k);
	float* P4 = strassenMatrixSSE(A22, M4, k);
	float* P5 = strassenMatrixSSE(M5, M6, k);
	float* P6 = strassenMatrixSSE(M7, M8, k);
	float* P7 = strassenMatrixSSE(M9, M10, k);

	free(A11);
    free(A12);
	free(A21);
	free(A22);
	free(B11);
	free(B12);
	free(B21);
	free(B22);

	free(M1);
	free(M2);
	free(M3);
	free(M4);
	free(M5);
	free(M6);
	free(M7);
	free(M8);
	free(M9);
	free(M10);

	float* M11 = addMatrix(P5, P4, k);
	float* M12 = addMatrix(M11, P6, k);
	float* M13 = addMatrix(P5, P1, k);
	float* M14 = subtractMatrix(M13, P3, k);

	// Compose matrix C from the submatrices
	float* C11 = subtractMatrix(M12, P2, k);
	float* C12 = addMatrix(P1, P2, k);
	float* C21 = addMatrix(P3, P4, k);
	float* C22 = subtractMatrix(M14, P7, k);

    free(P1);
	free(P2);
	free(P3);
	free(P4);
	free(P5);
	free(P6);
	free(P7);

	free(M11);
	free(M12);
	free(M13);
	free(M14);

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			int index = i*k+j;
			C[i*n+j] = C11[index];
			C[i*n+j+k] = C12[index];
			C[(k+i)*n+j] = C21[index];
			C[(k+i)*n+k+j] = C22[index];
    	}
	}

    free(C11);
	free(C12);
	free(C21);
	free(C22);
    
	return C;
}

// Strassen algorithm using IKJ loop
float* strassenMatrixIKJ(float* A, float* B, int n) {

	// Base case
	if(n <= 64) {
    	return multiplyMatrixIKJ(A, B, n);
	}

    float* C = (float*) malloc(n * n * sizeof(float));

	// Dimension of the matrices is the half of the input size
	int k = n/2;

	// Decompose A and B into 8 submatrices
	float* A11 = (float*) malloc(k * k * sizeof(float));
	float* A12 = (float*) malloc(k * k * sizeof(float));
	float* A21 = (float*) malloc(k * k * sizeof(float));
	float* A22 = (float*) malloc(k * k * sizeof(float));
	float* B11 = (float*) malloc(k * k * sizeof(float));
	float* B12 = (float*) malloc(k * k * sizeof(float));
	float* B21 = (float*) malloc(k * k * sizeof(float));
	float* B22 = (float*) malloc(k * k * sizeof(float));

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			int index = i*k+j;
			int index_1 = i*n+j;
			int index_2 = i*n+k+j;
			int index_3 = (k+i)*n+j;
			int index_4 = (k+i)*n+k+j;
			A11[index] = A[index_1];
			A12[index] = A[index_2];
			A21[index] = A[index_3];
			A22[index] = A[index_4];
			B11[index] = B[index_1];
			B12[index] = B[index_2];
			B21[index] = B[index_3];
			B22[index] = B[index_4];
    	}
	}

	// Create support matrices in order to calculate Strassen matrices
	float* M1 = subtractMatrix(B12, B22, k);
	float* M2 = addMatrix(A11, A12, k);
	float* M3 = addMatrix(A21, A22, k);
	float* M4 = subtractMatrix(B21, B11, k);
	float* M5 = addMatrix(A11, A22, k);
	float* M6 = addMatrix(B11, B22, k);
	float* M7 = subtractMatrix(A12, A22, k);
	float* M8 = addMatrix(B21, B22, k);
	float* M9 = subtractMatrix(A11, A21, k);
	float* M10 = addMatrix(B11, B12, k);

	// Create the Strassen matrices
	float* P1 = strassenMatrixIKJ(A11, M1, k);
	float* P2 = strassenMatrixIKJ(M2, B22, k);
	float* P3 = strassenMatrixIKJ(M3, B11, k);
	float* P4 = strassenMatrixIKJ(A22, M4, k);
	float* P5 = strassenMatrixIKJ(M5, M6, k);
	float* P6 = strassenMatrixIKJ(M7, M8, k);
	float* P7 = strassenMatrixIKJ(M9, M10, k);

	free(A11);
    free(A12);
	free(A21);
	free(A22);
	free(B11);
	free(B12);
	free(B21);
	free(B22);

	free(M1);
	free(M2);
	free(M3);
	free(M4);
	free(M5);
	free(M6);
	free(M7);
	free(M8);
	free(M9);
	free(M10);

	float* M11 = addMatrix(P5, P4, k);
	float* M12 = addMatrix(M11, P6, k);
	float* M13 = addMatrix(P5, P1, k);
	float* M14 = subtractMatrix(M13, P3, k);

	// Compose matrix C from the submatrices
	float* C11 = subtractMatrix(M12, P2, k);
	float* C12 = addMatrix(P1, P2, k);
	float* C21 = addMatrix(P3, P4, k);
	float* C22 = subtractMatrix(M14, P7, k);

    free(P1);
	free(P2);
	free(P3);
	free(P4);
	free(P5);
	free(P6);
	free(P7);

	free(M11);
	free(M12);
	free(M13);
	free(M14);

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			int index = i*k+j;
			C[i*n+j] = C11[index];
			C[i*n+j+k] = C12[index];
			C[(k+i)*n+j] = C21[index];
			C[(k+i)*n+k+j] = C22[index];
    	}
	}

    free(C11);
	free(C12);
	free(C21);
	free(C22);
    
	return C;
}

// Strassen algorithm using KIJ loop
float* strassenMatrixKIJ(float* A, float* B, int n) {

	// Base case
	if(n <= 64) {
    	return multiplyMatrixKIJ(A, B, n);
	}

    float* C = (float*) malloc(n * n * sizeof(float));

	// Dimension of the matrices is the half of the input size
	int k = n/2;

	// Decompose A and B into 8 submatrices
	float* A11 = (float*) malloc(k * k * sizeof(float));
	float* A12 = (float*) malloc(k * k * sizeof(float));
	float* A21 = (float*) malloc(k * k * sizeof(float));
	float* A22 = (float*) malloc(k * k * sizeof(float));
	float* B11 = (float*) malloc(k * k * sizeof(float));
	float* B12 = (float*) malloc(k * k * sizeof(float));
	float* B21 = (float*) malloc(k * k * sizeof(float));
	float* B22 = (float*) malloc(k * k * sizeof(float));

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			int index = i*k+j;
			int index_1 = i*n+j;
			int index_2 = i*n+k+j;
			int index_3 = (k+i)*n+j;
			int index_4 = (k+i)*n+k+j;
			A11[index] = A[index_1];
			A12[index] = A[index_2];
			A21[index] = A[index_3];
			A22[index] = A[index_4];
			B11[index] = B[index_1];
			B12[index] = B[index_2];
			B21[index] = B[index_3];
			B22[index] = B[index_4];
    	}
	}

	// Create support matrices in order to calculate Strassen matrices
	float* M1 = subtractMatrix(B12, B22, k);
	float* M2 = addMatrix(A11, A12, k);
	float* M3 = addMatrix(A21, A22, k);
	float* M4 = subtractMatrix(B21, B11, k);
	float* M5 = addMatrix(A11, A22, k);
	float* M6 = addMatrix(B11, B22, k);
	float* M7 = subtractMatrix(A12, A22, k);
	float* M8 = addMatrix(B21, B22, k);
	float* M9 = subtractMatrix(A11, A21, k);
	float* M10 = addMatrix(B11, B12, k);

	// Create the Strassen matrices
	float* P1 = strassenMatrixKIJ(A11, M1, k);
	float* P2 = strassenMatrixKIJ(M2, B22, k);
	float* P3 = strassenMatrixKIJ(M3, B11, k);
	float* P4 = strassenMatrixKIJ(A22, M4, k);
	float* P5 = strassenMatrixKIJ(M5, M6, k);
	float* P6 = strassenMatrixKIJ(M7, M8, k);
	float* P7 = strassenMatrixKIJ(M9, M10, k);

	free(A11);
    free(A12);
	free(A21);
	free(A22);
	free(B11);
	free(B12);
	free(B21);
	free(B22);

	free(M1);
	free(M2);
	free(M3);
	free(M4);
	free(M5);
	free(M6);
	free(M7);
	free(M8);
	free(M9);
	free(M10);

	float* M11 = addMatrix(P5, P4, k);
	float* M12 = addMatrix(M11, P6, k);
	float* M13 = addMatrix(P5, P1, k);
	float* M14 = subtractMatrix(M13, P3, k);

	// Compose matrix C from the submatrices
	float* C11 = subtractMatrix(M12, P2, k);
	float* C12 = addMatrix(P1, P2, k);
	float* C21 = addMatrix(P3, P4, k);
	float* C22 = subtractMatrix(M14, P7, k);

    free(P1);
	free(P2);
	free(P3);
	free(P4);
	free(P5);
	free(P6);
	free(P7);

	free(M11);
	free(M12);
	free(M13);
	free(M14);

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			int index = i*k+j;
			C[i*n+j] = C11[index];
			C[i*n+j+k] = C12[index];
			C[(k+i)*n+j] = C21[index];
			C[(k+i)*n+k+j] = C22[index];
    	}
	}

    free(C11);
	free(C12);
	free(C21);
	free(C22);
    
	return C;
}

// Strassen algorithm using chunked version
float* strassenMatrixChunk(float* A, float* B, int n, int base) {

	// Base case
	if(n <= base) {
    		return multiplyMatrixChunk2(A, B, n);
	}

    float* C = (float*) malloc(n * n * sizeof(float));

	// Dimension of the matrices is the half of the input size
	int k = n/2;

	// Decompose A and B into 8 submatrices
	float* A11 = (float*) malloc(k * k * sizeof(float));
	float* A12 = (float*) malloc(k * k * sizeof(float));
	float* A21 = (float*) malloc(k * k * sizeof(float));
	float* A22 = (float*) malloc(k * k * sizeof(float));
	float* B11 = (float*) malloc(k * k * sizeof(float));
	float* B12 = (float*) malloc(k * k * sizeof(float));
	float* B21 = (float*) malloc(k * k * sizeof(float));
	float* B22 = (float*) malloc(k * k * sizeof(float));

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			int index = i*k+j;
			int index_1 = i*n+j;
			int index_2 = i*n+k+j;
			int index_3 = (k+i)*n+j;
			int index_4 = (k+i)*n+k+j;
			A11[index] = A[index_1];
			A12[index] = A[index_2];
			A21[index] = A[index_3];
			A22[index] = A[index_4];
			B11[index] = B[index_1];
			B12[index] = B[index_2];
			B21[index] = B[index_3];
			B22[index] = B[index_4];
    	}
	}

	// Create support matrices in order to calculate Strassen matrices
	float* M1 = subtractMatrix(B12, B22, k);
	float* M2 = addMatrix(A11, A12, k);
	float* M3 = addMatrix(A21, A22, k);
	float* M4 = subtractMatrix(B21, B11, k);
	float* M5 = addMatrix(A11, A22, k);
	float* M6 = addMatrix(B11, B22, k);
	float* M7 = subtractMatrix(A12, A22, k);
	float* M8 = addMatrix(B21, B22, k);
	float* M9 = subtractMatrix(A11, A21, k);
	float* M10 = addMatrix(B11, B12, k);

	// Create the Strassen matrices
	float* P1 = strassenMatrixChunk(A11, M1, k, base);
	float* P2 = strassenMatrixChunk(M2, B22, k, base);
	float* P3 = strassenMatrixChunk(M3, B11, k, base);
	float* P4 = strassenMatrixChunk(A22, M4, k, base);
	float* P5 = strassenMatrixChunk(M5, M6, k, base);
	float* P6 = strassenMatrixChunk(M7, M8, k, base);
	float* P7 = strassenMatrixChunk(M9, M10, k, base);

	free(A11);
    free(A12);
	free(A21);
	free(A22);
	free(B11);
	free(B12);
	free(B21);
	free(B22);

	free(M1);
	free(M2);
	free(M3);
	free(M4);
	free(M5);
	free(M6);
	free(M7);
	free(M8);
	free(M9);
	free(M10);

	float* M11 = addMatrix(P5, P4, k);
	float* M12 = addMatrix(M11, P6, k);
	float* M13 = addMatrix(P5, P1, k);
	float* M14 = subtractMatrix(M13, P3, k);

	// Compose matrix C from the submatrices
	float* C11 = subtractMatrix(M12, P2, k);
	float* C12 = addMatrix(P1, P2, k);
	float* C21 = addMatrix(P3, P4, k);
	float* C22 = subtractMatrix(M14, P7, k);

    free(P1);
	free(P2);
	free(P3);
	free(P4);
	free(P5);
	free(P6);
	free(P7);

	free(M11);
	free(M12);
	free(M13);
	free(M14);

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			int index = i*k+j;
			C[i*n+j] = C11[index];
			C[i*n+j+k] = C12[index];
			C[(k+i)*n+j] = C21[index];
			C[(k+i)*n+k+j] = C22[index];
    	}
	}

    free(C11);
	free(C12);
	free(C21);
	free(C22);
    
	return C;
}

int main(int argc, char **argv) {

    const int n = 4096;				// n is the size of the matrix
    int rank, size;					// rank identify the process and size is the number of processes available
    double start, end;				// variables used to evaluate execution time

	// Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Initialize matrices
    if(rank == 0) {
        float *A, *B, *C, *D;

        A = (float*) malloc(n * n * sizeof(float));
        B = (float*) malloc(n * n * sizeof(float));
        D = (float*) calloc(n * n, sizeof(float));

        float** A1 = (float**) malloc(n * sizeof(float*));
        float** B1 = (float**) malloc(n * sizeof(float*));
        float** C1 = (float**) malloc(n * sizeof(float*));

        for(int i=0; i<n; i++){
            A1[i] = (float*) malloc(n * sizeof(float));
            B1[i] = (float*) malloc(n * sizeof(float));
            C1[i] = (float*) malloc(n * sizeof(float));
        }


        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                A[i*n+j] = i+j;
                B[i*n+j] = i-j;
                A1[i][j] = i+j;
                B1[i][j] = i-j;
                C1[i][j] = 0;
            }
        }
	
		// Run all algorithms and measure execution time

		start = MPI_Wtime();
        C = multiplyMatrixChunk1(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 1: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        C = multiplyMatrixChunk2(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 2: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        C = multiplyMatrixChunk3(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 3: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        C = multiplyMatrixChunk4(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 4: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        C = multiplyMatrixChunk5(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 5: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        C = multiplyMatrixChunk6(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 6: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();
        C = multiplyMatrixChunk7(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 7: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        C = multiplyMatrixChunk8(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 8: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        C = multiplyMatrixChunk9(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 9: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        C = multiplyMatrixChunk10(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 10: %f ms\n", (end-start)*1000);

		start = MPI_Wtime();
        multiplyMatrixChunk11(A1, B1, C1, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Chunk version 11: %f ms\n", (end-start)*1000);
	
        start = MPI_Wtime();
        C = multiplyMatrixIKJ(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM IKJ: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();
        C = strassenMatrixIKJ(A, B, n);
        end = MPI_Wtime();

        printf("Time took strassen IKJ: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();
        C = multiplyMatrixKIJ(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM KIJ: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();
        C = strassenMatrixKIJ(A, B, n);
        end = MPI_Wtime();

        printf("Time took strassen KIJ: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();
        C = multiplyMatrixTransposition(A, B, n);
        end = MPI_Wtime();

        printf("Time took sequential MM Transposition: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();
        multiplyMatrixSSE(A, B, D, n);
        end = MPI_Wtime();

        printf("Time took sequential MM with SSE: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();
        C = strassenMatrixSSE(A, B, n);
        end = MPI_Wtime();

        printf("Time took strassen with SSE: %f ms\n", (end-start)*1000);

        free(A);
        free(B);
        free(C);
        free(D);
    }

    MPI_Finalize();

    return 0;
}
