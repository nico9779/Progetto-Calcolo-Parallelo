#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <xmmintrin.h>

void multiplyMatrix(float *A, float *B, float *C, int n) {
    __m128 ma , *mb , *mc = ( __m128 *) C;
    int t1 ,t2;
    int n_4 = (n>>3)<<1;
    for (int k=0; k<n; k++) {
        mb = ( __m128 *) B;
        for (int i=0; i<n; i++, mb+=n_4) {
            ma = _mm_load1_ps(A++); //k*a+i
            for (int j=0; j<n_4; j+=4) {
                mc[j] = _mm_add_ps (mc[j], _mm_mul_ps (ma, mb[j]));
                mc[j+1] = _mm_add_ps (mc[j+1] , _mm_mul_ps (ma, mb[j+1]));
                mc[j+2] = _mm_add_ps (mc[j+2] , _mm_mul_ps (ma, mb[j+2]));
                mc[j+3] = _mm_add_ps (mc[j+3] , _mm_mul_ps (ma, mb[j+3]));
            }
        }
        mc += n_4 ;
    }
}

int main(int argc, char **argv) {

    const int n = 4096;
    int rank, size;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {
        float *A, *B, *C;

        A = (float*) malloc(n * n * sizeof(float));
        B = (float*) malloc(n * n * sizeof(float));
        C = (float*) malloc(n * n * sizeof(float));

        srand(time(NULL));

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                A[i*n+j] = rand()%10+1;
                B[i*n+j] = rand()%10+1;
            }
        }

        start = MPI_Wtime();
        multiplyMatrix(A, B, C, n);
        end = MPI_Wtime();

        free(A);
        free(B);
        free(C);

        printf("Time took sequential MM: %f ms\n", (end-start)*1000);
    }
}
