#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

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
    for(int i=0; i<n/size; i++) {
        for(int k=0; k<n; k++) {
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
	if(n <= 64) {
    	return multiplyMatrixSequential(A, B, n);
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

    int rank, size;
    MPI_Request req[8];
    MPI_Status status;
    double start, end;

	float *A, *B;

	float *P1 = NULL;
    float *P2 = NULL;
    float *P3 = NULL;
    float *P4 = NULL;
    float *P5 = NULL;
    float *P6 = NULL;
    float *P7 = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	//***** SEQUENTIAL PHASE *****//

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

        // Calculate time to run Strassen sequential algorithm
        start = MPI_Wtime();
        float *C = strassenMatrix(A, B, n);
        end = MPI_Wtime();

        free(C);

        printf("Time took sequential Strassen: %f ms\n", (end-start)*1000);

	}

	start = MPI_Wtime();

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

        // Send matrices to other processors
        MPI_Isend(A11, num_elements, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(A12, num_elements, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(B12, num_elements, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(B22, num_elements, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(A21, num_elements, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(B21, num_elements, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(A22, num_elements, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(B11, num_elements, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(A22, num_elements, MPI_FLOAT, 3, 3, MPI_COMM_WORLD, &req[3]);
        MPI_Isend(B22, num_elements, MPI_FLOAT, 3, 3, MPI_COMM_WORLD, &req[3]);
        MPI_Isend(A12, num_elements, MPI_FLOAT, 3, 3, MPI_COMM_WORLD, &req[3]);
        MPI_Isend(B21, num_elements, MPI_FLOAT, 3, 3, MPI_COMM_WORLD, &req[3]);

        // #0 compute P1 and P6
        float *T1, *T2, *T3, *T4;
        T1 = addMatrix(A11, A22, k);
        T2 = addMatrix(B11, B22, k);
        T3 = subtractMatrix(A21, A11, k);
        T4 = addMatrix(B11, B12, k);

        free(A);
        free(B);
        free(A11);
        free(A12);
        free(A21);
        free(A22);
        free(B11);
        free(B12);
        free(B21);
        free(B22);

		P1 = strassenMatrix(T1, T2, k);
		P6 = strassenMatrix(T3, T4, k);

		free(T1);
		free(T2);
		free(T3);
		free(T4);
    }

	if(rank == 1) {
        float *A12, *B12, *A11, *B22;
        A12 = (float*) malloc(num_elements * sizeof(float));
        B12 = (float*) malloc(num_elements * sizeof(float));
        A11 = (float*) malloc(num_elements * sizeof(float));
        B22 = (float*) malloc(num_elements * sizeof(float));
        MPI_Irecv(A11, num_elements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(A12, num_elements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(B12, num_elements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(B22, num_elements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Wait(&req[1], &status);

        // #1 compute P3 and P5

        float *T1, *T2;

        T1 = subtractMatrix(B12, B22, k);
        T2 = addMatrix(A11, A12, k);
        
		P3 = strassenMatrix(A11, T1, k);
		P5 = strassenMatrix(T2, B22, k);

		MPI_Isend(P3, num_elements, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(P5, num_elements, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &req[4]);

		free(A12);
        free(B12);
		free(A11);
        free(B22);
		free(T1);
		free(T2);
    }

    if(rank == 2) {
        float *A21, *B21, *A22, *B11;
        A21 = (float*) malloc(num_elements * sizeof(float));
        B21 = (float*) malloc(num_elements * sizeof(float));
        A22 = (float*) malloc(num_elements * sizeof(float));
        B11 = (float*) malloc(num_elements * sizeof(float));
        MPI_Irecv(A21, num_elements, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(B21, num_elements, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(A22, num_elements, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(B11, num_elements, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Wait(&req[2], &status);

        // #2 compute P2 and P4
        float *T1, *T2;

        T1 = addMatrix(A21, A22, k);
        T2 = subtractMatrix(B21, B11, k);  
        
        P2 = strassenMatrix(T1, B11, k);
		P4 = strassenMatrix(A22, T2, k);

		MPI_Isend(P2, num_elements, MPI_FLOAT, 0, 5, MPI_COMM_WORLD, &req[5]);
        MPI_Isend(P4, num_elements, MPI_FLOAT, 0, 5, MPI_COMM_WORLD, &req[5]);

		free(A21);
        free(B21);
		free(A22);
        free(B11);
		free(T1);
		free(T2);
    }

    if(rank == 3) {
        float *A22, *B22, *A12, *B21;
        A22 = (float*) malloc(num_elements * sizeof(float));
        B22 = (float*) malloc(num_elements * sizeof(float));
        A12 = (float*) malloc(num_elements * sizeof(float));
        B21 = (float*) malloc(num_elements * sizeof(float));
        MPI_Irecv(A22, num_elements, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(B22, num_elements, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(A12, num_elements, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(B21, num_elements, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &req[3]);
        MPI_Wait(&req[3], &status);

        // #3 compute P7

        float *T1, *T2;

        T1 = subtractMatrix(A12, A22, k);
        T2 = addMatrix(B21, B22, k);
        
        P7 = strassenMatrix(T1, T2, k);

		MPI_Isend(P7, num_elements, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, &req[6]);

		free(A22);
        free(B22);
        free(A12);
        free(B21);
		free(T1);
        free(T2);
    }


	if(rank == 0) {
        float *C11, *C12, *C21, *C22, *C;
        C = (float*) malloc(n * n * sizeof(float));

		P2 = (float*) malloc(num_elements * sizeof(float));
		P3 = (float*) malloc(num_elements * sizeof(float));
		P4 = (float*) malloc(num_elements * sizeof(float));
		P5 = (float*) malloc(num_elements * sizeof(float));
		P7 = (float*) malloc(num_elements * sizeof(float));
        
        MPI_Irecv(P3, num_elements, MPI_FLOAT, 1, 4, MPI_COMM_WORLD, &req[4]);
        MPI_Irecv(P5, num_elements, MPI_FLOAT, 1, 4, MPI_COMM_WORLD, &req[4]);
        MPI_Irecv(P2, num_elements, MPI_FLOAT, 2, 5, MPI_COMM_WORLD, &req[5]);
        MPI_Irecv(P4, num_elements, MPI_FLOAT, 2, 5, MPI_COMM_WORLD, &req[5]);
        MPI_Irecv(P7, num_elements, MPI_FLOAT, 3, 6, MPI_COMM_WORLD, &req[6]);

        MPI_Wait(&req[4], &status);
        MPI_Wait(&req[5], &status);
        MPI_Wait(&req[6], &status);

        //Calculate matrices to compute matrix C
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

        //printMatrix(C, n);

        free(C);

        end = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);

	if(P1 != NULL)
		free(P1);
	if(P2 != NULL)
		free(P2);
	if(P3 != NULL)
		free(P3);
	if(P4 != NULL)
		free(P4);
	if(P5 != NULL)
		free(P5);
	if(P6 != NULL)
		free(P6);
	if(P7 != NULL)
		free(P7);

    MPI_Finalize();

	if(rank == 0) {
        printf("Time took parallel Strassen: %f ms\n", (end-start)*1000);
    }

    return 0;

}