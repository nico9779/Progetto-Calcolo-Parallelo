#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Add two square matrices
int* addMatrix(int* M1, int* M2, int n) {

	int* temp = (int*) malloc(n * n * sizeof(int));

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			int index = i*n+j;
			temp[index] = M1[index] + M2[index];
		}
	}

    return temp;
}

// Subtract two square matrices
int* subtractMatrix(int* M1, int* M2, int n) {

	int* temp = (int*) malloc(n * n * sizeof(int));

    for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			int index = i*n+j;
			temp[index] = M1[index] - M2[index];
		}  
	}
        
    return temp;
}

// Multiply two matrices sequentially using standard definition
int* multiplyMatrixSequential(int* A, int* B, int n) {

	int* C = (int*) calloc(n * n, sizeof(int));

	for(int i = 0; i < n; i++) {
		for(int k = 0; k < n; k++) {
			for(int j = 0; j < n; j++) {
				C[i*n+j] += A[i*n+k] * B[k*n+j];
			}
		}
	}

	return C;
}

// Multiply two matrices in parallel
void multiplyMatrixParallel(int* A, int* B, int* C, int root, int n, int size) {

    int num_elements = n*n/size;

    int* local_A = (int*) malloc(num_elements * sizeof(int));
    int* local_C = (int*) calloc(num_elements, sizeof(int));

    // Scatter matrix A between all processors
    MPI_Scatter(A, num_elements, MPI_INT, local_A, num_elements, MPI_INT, root, MPI_COMM_WORLD);

    // Broadcast matrix B between all processors
    MPI_Bcast(B, n*n, MPI_INT, root, MPI_COMM_WORLD);
    
    // Multiply the two matrices to obtain a piece of matrix C
    for(int i=0; i<n/size; i++) {
        for(int k=0; k<n; k++) {
            for(int j=0; j<n; j++) {
                local_C[i*n+j] += local_A[i*n+k] * B[k*n+j];
            }
        }
    }

    free(local_A);

    // Gather C from all processors to compute the product
    MPI_Gather(local_C, num_elements, MPI_INT, C, num_elements, MPI_INT, root, MPI_COMM_WORLD);

    free(local_C);
}

// Print matrix in output
void printMatrix(int* M, int rows, int cols) {

	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			printf("%d ", M[i*cols+j]);
		}
		printf("\n");
	}

	printf("\n");
}

// Multiply two matrices using Strassen algorithm
int* strassenMatrix(int* A, int* B, int n) {

	// Base case
	if(n <= 64) {
    	return multiplyMatrixSequential(A, B, n);
	}

	// Initialize matrix C to return in output (C = A*B)
	int* C = (int*) malloc(n * n * sizeof(int));

	// Dimension of the matrices is the half of the input size
	int k = n/2;

	// Decompose A and B into 8 submatrices
	int* A11 = (int*) malloc(k * k * sizeof(int));
	int* A12 = (int*) malloc(k * k * sizeof(int));
	int* A21 = (int*) malloc(k * k * sizeof(int));
	int* A22 = (int*) malloc(k * k * sizeof(int));
	int* B11 = (int*) malloc(k * k * sizeof(int));
	int* B12 = (int*) malloc(k * k * sizeof(int));
	int* B21 = (int*) malloc(k * k * sizeof(int));
	int* B22 = (int*) malloc(k * k * sizeof(int));

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
	int* M1 = subtractMatrix(B12, B22, k);
	int* M2 = addMatrix(A11, A12, k);
	int* M3 = addMatrix(A21, A22, k);
	int* M4 = subtractMatrix(B21, B11, k);
	int* M5 = addMatrix(A11, A22, k);
	int* M6 = addMatrix(B11, B22, k);
	int* M7 = subtractMatrix(A12, A22, k);
	int* M8 = addMatrix(B21, B22, k);
	int* M9 = subtractMatrix(A11, A21, k);
	int* M10 = addMatrix(B11, B12, k);

	// Create the Strassen matrices
	int* P1 = strassenMatrix(A11, M1, k);
	int* P2 = strassenMatrix(M2, B22, k);
	int* P3 = strassenMatrix(M3, B11, k);
	int* P4 = strassenMatrix(A22, M4, k);
	int* P5 = strassenMatrix(M5, M6, k);
	int* P6 = strassenMatrix(M7, M8, k);
	int* P7 = strassenMatrix(M9, M10, k);

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

	int* M11 = addMatrix(P5, P4, k);
	int* M12 = addMatrix(M11, P6, k);
	int* M13 = addMatrix(P5, P1, k);
	int* M14 = subtractMatrix(M13, P3, k);

	// Compose matrix C from the submatrices
	int* C11 = subtractMatrix(M12, P2, k);
	int* C12 = addMatrix(P1, P2, k);
	int* C21 = addMatrix(P3, P4, k);
	int* C22 = subtractMatrix(M14, P7, k);

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
    double start, end;

    // Allocate memory for the matrices to multiply in Strassen algorithm
    int *M1, *M2, *M3, *M4, *M5, *M6, *M7, *M8, *M9, *M10, *M11, *M12, *M13, *M14;
    M1 = (int*) malloc(num_elements * sizeof(int));
    M2 = (int*) malloc(num_elements * sizeof(int));
    M3 = (int*) malloc(num_elements * sizeof(int));
    M4 = (int*) malloc(num_elements * sizeof(int));
    M5 = (int*) malloc(num_elements * sizeof(int));
    M6 = (int*) malloc(num_elements * sizeof(int));
    M7 = (int*) malloc(num_elements * sizeof(int));
    M8 = (int*) malloc(num_elements * sizeof(int));
    M9 = (int*) malloc(num_elements * sizeof(int));
    M10 = (int*) malloc(num_elements * sizeof(int));
    M11 = (int*) malloc(num_elements * sizeof(int));
    M12 = (int*) malloc(num_elements * sizeof(int));
    M13 = (int*) malloc(num_elements * sizeof(int));
    M14 = (int*) malloc(num_elements * sizeof(int));

    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {
        int *A, *B;

        // Generate matrices A and B
        A = (int*) malloc(n * n * sizeof(int));
        B = (int*) malloc(n * n * sizeof(int));

        srand(time(NULL));

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                A[i*n+j] = rand()%10+1;
                B[i*n+j] = rand()%10+1;
            }
        }

        // Calculate time to run Strassen sequential algorithm
        start = MPI_Wtime();
        int *C = strassenMatrix(A, B, n);
        end = MPI_Wtime();

        free(C);

        printf("Time took sequential Strassen: %f ms\n", (end-start)*1000);

        start = MPI_Wtime();

        // Decompose A and B into 8 submatrices
        int *A11,*A12,*A21,*A22,*B11,*B12,*B21,*B22;
        A11 = (int*) malloc(num_elements * sizeof(int));
        A12 = (int*) malloc(num_elements * sizeof(int));
        A21 = (int*) malloc(num_elements * sizeof(int));
        A22 = (int*) malloc(num_elements * sizeof(int));
        B11 = (int*) malloc(num_elements * sizeof(int));
        B12 = (int*) malloc(num_elements * sizeof(int));
        B21 = (int*) malloc(num_elements * sizeof(int));
        B22 = (int*) malloc(num_elements * sizeof(int));

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
        free(M11);
        free(M12);
        free(M13);
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

        free(A);
        free(B);
        free(A12);
        free(A21);
        free(B12);
        free(B21);
    }

    int *P1, *P2, *P3, *P4, *P5, *P6, *P7;

    // Allocate memory for the Strassen products
    P1 = (int*) malloc(num_elements * sizeof(int));
    P2 = (int*) malloc(num_elements * sizeof(int));
    P3 = (int*) malloc(num_elements * sizeof(int));
    P4 = (int*) malloc(num_elements * sizeof(int));
    P5 = (int*) malloc(num_elements * sizeof(int));
    P6 = (int*) malloc(num_elements * sizeof(int));
    P7 = (int*) malloc(num_elements * sizeof(int));

    // Multiply matrices in parallel
    multiplyMatrixParallel(M1, M2, P1, 0, k, size);
    multiplyMatrixParallel(M3, M4, P2, 0, k, size);
    multiplyMatrixParallel(M5, M6, P3, 0, k, size);
    multiplyMatrixParallel(M7, M8, P4, 0, k, size);
    multiplyMatrixParallel(M9, M10, P5, 0, k, size);
    multiplyMatrixParallel(M11, M12, P6, 0, k, size);
    multiplyMatrixParallel(M13, M14, P7, 0, k, size);

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
    free(M11);
    free(M12);
    free(M13);
    free(M14);

    if(rank == 0) {
        int *C11, *C12, *C21, *C22, *C;
        C = (int*) malloc(n * n * sizeof(int));

        // Calculate matrices to compute matrix C
        int *T1, *T2, *T3, *T4;
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

        //printMatrix(C, n, n);

        free(C);

        end = MPI_Wtime();
    }

    free(P1);
    free(P2);
    free(P3);
    free(P4);
    free(P5);
    free(P6);
    free(P7);

    MPI_Finalize();

    if(rank == 0) {
        printf("Time took parallel Strassen: %f ms\n", (end-start)*1000);
    }
    
    return 0;
}
