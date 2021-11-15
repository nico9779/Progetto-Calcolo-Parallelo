#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <xmmintrin.h>

void multiplyMatrixIKJ(float* A, float* B, float* C, int n) {

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
}

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

void printMatrix(float* M, int n) {

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			printf("%f  ", M[i*n+j]);
		}
		printf("\n");
	}

	printf("\n");
}

int main(int argc, char **argv) {

    const int n = 4096;
    int rank, size;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {
        float *A, *B, *C, *D;

        A = (float*) malloc(n * n * sizeof(float));
        B = (float*) malloc(n * n * sizeof(float));
        C = (float*) calloc(n * n, sizeof(float));
        D = (float*) calloc(n * n, sizeof(float));

        srand(time(NULL));

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                A[i*n+j] = rand()%10+1;
                B[i*n+j] = rand()%10+1;
            }
        }

        start = MPI_Wtime();
        multiplyMatrixIKJ(A, B, C, n);
        end = MPI_Wtime();

        printf("Time took sequential MM IKJ: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();
        multiplyMatrixSSE(A, B, D, n);
        end = MPI_Wtime();

        printf("Time took sequential MM with SSE: %f ms\n", (end-start)*1000);

        free(A);
        free(B);
        free(C);
        free(D);
    }

    MPI_Finalize();

    return 0;
}
