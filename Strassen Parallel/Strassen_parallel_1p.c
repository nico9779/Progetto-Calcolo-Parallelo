#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define CHUNK 32

double average(double times[], int size) {

    double average = 0;

    for(int i=0; i<size; i++) {
        average += times[i];
    }

    average = average/size;

    return average;
}

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

// Multiply two matrices sequentially using standard definition
float* multiplyMatrixSequential(float* A, float* B, int n) {

	float* C = (float*) calloc(n * n, sizeof(float));
    float a = 0;

    for(int kk=0; kk<n; kk+=CHUNK) {
        for(int i=0; i<n; ++i) {
            for(int k=kk; k<kk+CHUNK; ++k) {
                a = A[i*n+k];
                for(int j=0; j<n; j+=8) {
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
	
	return C;
}

// Multiply two matrices in parallel
void multiplyMatrixParallel(float* A, float* B, float* C, int root, int n, int size) {

    int num_elements = n*n/size;

    float* local_A = (float*) malloc(num_elements * sizeof(float));
    float* local_C = (float*) calloc(num_elements, sizeof(float));

    // Scatter matrix A between all processors
    MPI_Scatter(A, num_elements, MPI_FLOAT, local_A, num_elements, MPI_FLOAT, root, MPI_COMM_WORLD);

    // Broadcast matrix B between all processors
    MPI_Bcast(B, n*n, MPI_FLOAT, root, MPI_COMM_WORLD);
    
    // Multiply the two matrices to obtain a piece of matrix C
    float a = 0;

    for(int kk=0; kk<n; kk+=CHUNK) {
        for(int i=0; i<n/size; ++i) {
            for(int k=kk; k<kk+CHUNK; ++k) {
                a = local_A[i*n+k];
                for(int j=0; j<n; j+=8) {
                    local_C[i*n+j] += a * B[k*n+j];
                    local_C[i*n+j+1] += a * B[k*n+j+1];
                    local_C[i*n+j+2] += a * B[k*n+j+2];
                    local_C[i*n+j+3] += a * B[k*n+j+3];
                    local_C[i*n+j+4] += a * B[k*n+j+4];
                    local_C[i*n+j+5] += a * B[k*n+j+5];
                    local_C[i*n+j+6] += a * B[k*n+j+6];
                    local_C[i*n+j+7] += a * B[k*n+j+7];
                }
            }
        }
    }
    

    free(local_A);

    // Gather C from all processors to compute the product
    MPI_Gather(local_C, num_elements, MPI_FLOAT, C, num_elements, MPI_FLOAT, root, MPI_COMM_WORLD);

    free(local_C);
}

// Print matrix in output
void printMatrix(float* M, int n) {

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			printf("%f  ", M[i*n+j]);
		}
		printf("\n");
	}

	printf("\n");
}

// Multiply two matrices using Strassen algorithm
float* strassenMatrix(float* A, float* B, int n) {

	// Base case
	if(n <= 256) {
        float* C = (float*) calloc(n * n, sizeof(float));
    	float a = 0;

        for(int kk=0; kk<n; kk+=CHUNK) {
            for(int i=0; i<n; ++i) {
                for(int k=kk; k<kk+CHUNK; ++k) {
                    a = A[i*n+k];
                    for(int j=0; j<n; j+=8) {
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

        return C;
	}

	// Initialize matrix C to return in output (C = A*B)
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
	float* P1 = strassenMatrix(A11, M1, k);
	float* P2 = strassenMatrix(M2, B22, k);
	float* P3 = strassenMatrix(M3, B11, k);
	float* P4 = strassenMatrix(A22, M4, k);
	float* P5 = strassenMatrix(M5, M6, k);
	float* P6 = strassenMatrix(M7, M8, k);
	float* P7 = strassenMatrix(M9, M10, k);

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

    const int n = 4096;
    const int k = n/2;
    const int num_elements = k*k;
    const int repetition = 5;

    int rank, size;
    double start, end;
    double times[repetition];

    float *A, *B;

    float *M1, *M2, *M3, *M4, *M5, *M6, *M7, *M8, *M9, *M10, *M11, *M12, *M13, *M14;

    float *P1, *P2, *P3, *P4, *P5, *P6, *P7;

    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    if(rank == 0) {

        // Generate matrices A and B
        A = (float*) malloc(n * n * sizeof(float));
        B = (float*) malloc(n * n * sizeof(float));

        srand(time(NULL));

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                A[i*n+j] = rand()%10+1;
                B[i*n+j] = rand()%10+1;
            }
        }
    }

    //***** SEQUENTIAL PHASE *****//

    if(rank == 0) {

        for(int i=0; i<repetition; i++) {

            // Calculate time to run Strassen sequential algorithm
            start = MPI_Wtime();
            float *C = strassenMatrix(A, B, n);
            end = MPI_Wtime();

            times[i] = (end-start)*1000;

            free(C);
        }

        printf("Average time took sequential Strassen: %f ms\n", average(times, repetition));

    }

    //***** PARALLEL PHASE *****//

    for(int i=0; i<repetition; i++) {

        start = MPI_Wtime();

        M2 = (float*) malloc(num_elements * sizeof(float));
        M4 = (float*) malloc(num_elements * sizeof(float));
        M6 = (float*) malloc(num_elements * sizeof(float));
        M8 = (float*) malloc(num_elements * sizeof(float));
        M10 = (float*) malloc(num_elements * sizeof(float));
        M12 = (float*) malloc(num_elements * sizeof(float));
        M14 = (float*) malloc(num_elements * sizeof(float));

        if(rank == 0) {

            // Decompose A and B into 8 submatrices
            float *A11,*A12,*A21,*A22,*B11,*B12,*B21,*B22;
            A11 = (float*) malloc(num_elements * sizeof(float));
            A12 = (float*) malloc(num_elements * sizeof(float));
            A21 = (float*) malloc(num_elements * sizeof(float));
            A22 = (float*) malloc(num_elements * sizeof(float));
            B11 = (float*) malloc(num_elements * sizeof(float));
            B12 = (float*) malloc(num_elements * sizeof(float));
            B21 = (float*) malloc(num_elements * sizeof(float));
            B22 = (float*) malloc(num_elements * sizeof(float));

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

            free(M2);
            free(M4);
            free(M6);
            free(M8);
            free(M10);
            free(M12);
            free(M14);

            M1 = addMatrix(A11, A22, k);
            M2 = addMatrix(B11, B22, k);
            M3 = addMatrix(A21, A22, k);
            M4 = B11;
            M5 = A11;
            M6 = subtractMatrix(B12, B22, k);
            M7 = A22;
            M8 = subtractMatrix(B21, B11, k);
            M9 = addMatrix(A11, A12, k);
            M10 = B22;
            M11 = subtractMatrix(A21, A11, k);
            M12 = addMatrix(B11, B12, k);
            M13 = subtractMatrix(A12, A22, k);
            M14 = addMatrix(B21, B22, k);

            free(A12);
            free(A21);
            free(B12);
            free(B21);

            // Allocate memory for the Strassen products
            P1 = (float*) malloc(num_elements * sizeof(float));
            P2 = (float*) malloc(num_elements * sizeof(float));
            P3 = (float*) malloc(num_elements * sizeof(float));
            P4 = (float*) malloc(num_elements * sizeof(float));
            P5 = (float*) malloc(num_elements * sizeof(float));
            P6 = (float*) malloc(num_elements * sizeof(float));
            P7 = (float*) malloc(num_elements * sizeof(float));
        }

        // Multiply matrices in parallel
        multiplyMatrixParallel(M1, M2, P1, 0, k, size);
        multiplyMatrixParallel(M3, M4, P2, 0, k, size);
        multiplyMatrixParallel(M5, M6, P3, 0, k, size);
        multiplyMatrixParallel(M7, M8, P4, 0, k, size);
        multiplyMatrixParallel(M9, M10, P5, 0, k, size);
        multiplyMatrixParallel(M11, M12, P6, 0, k, size);
        multiplyMatrixParallel(M13, M14, P7, 0, k, size);

        free(M2);
        free(M4);
        free(M6);
        free(M8);
        free(M10);
        free(M12);
        free(M14);

        if(rank == 0) {

            free(M1);
            free(M3);
            free(M5);
            free(M7);
            free(M9);
            free(M11);
            free(M13);

            float *C11, *C12, *C21, *C22, *C;
            C = (float*) malloc(n * n * sizeof(float));

            // Calculate matrices to compute matrix C
            float *T1, *T2, *T3, *T4;
            T1 = addMatrix(P1, P4, k);
            T2 = subtractMatrix(T1, P5, k);
            C11 = addMatrix(T2, P7, k);

            C12 = addMatrix(P3, P5, k);
            C21 = addMatrix(P2, P4, k);

            T3 = subtractMatrix(P1, P2, k);
            T4 = addMatrix(T3, P3, k);
            C22 = addMatrix(T4, P6, k);

            for(int i=0; i<k; i++) {
                for(int j=0; j<k; j++) {
                    int index = i*k+j;
                    C[i*n+j] = C11[index];
                    C[i*n+j+k] = C12[index];
                    C[(k+i)*n+j] = C21[index];
                    C[(k+i)*n+k+j] = C22[index];
                }
            }

            free(T1);
            free(T2);
            free(T3);
            free(T4);

            free(C11);
            free(C12);
            free(C21);
            free(C22);

            free(P1);
            free(P2);
            free(P3);
            free(P4);
            free(P5);
            free(P6);
            free(P7);

            //printMatrix(C, n, n);

            free(C);
    
        }

        end = MPI_Wtime();

        times[i] = (end-start)*1000;
    }

    MPI_Finalize();

    if(rank == 0) {
        free(A);
        free(B);
        printf("Average time took parallel Strassen: %f ms\n", average(times, repetition));
    }
    
    return 0;
}
