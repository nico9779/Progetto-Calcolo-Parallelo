#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

// Allocate memory to store square matrix
int* allocateSquareMatrixMemory(int n) {

    int* temp = (int*) malloc(n * n * sizeof(int));

    return temp;
}

// Allocate memory to store matrix
int* allocateMatrixMemory(int rows, int cols) {

    int* temp = (int*) malloc(rows * cols * sizeof(int));

    return temp;
}

// Add two square matrices
int* addMatrix(int* M1, int* M2, int n) {

	int* temp = allocateSquareMatrixMemory(n);

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

	int* temp = allocateSquareMatrixMemory(n);

    for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			int index = i*n+j;
			temp[index] = M1[index] - M2[index];
		}  
	}
        
    return temp;
}

// Multiply two matrices using standard definition
int* multiplyMatrix(int* A, int* B, int rA, int cA, int cB) {

	int* C = allocateMatrixMemory(rA, cB);

	for(int i = 0; i < rA; i++) {
		for(int j = 0; j < cB; j++) {
			int index = i*cB+j;
            C[index] = A[i*cA] * B[j];
			for(int k = 1; k < cA; k++) {
				C[index] += A[i*cA+k] * B[k*cB+j];
			}
		}
	}

	return C;
}

// Multiply two matrices using Strassen algorithm
int* strassenMatrix(int* A, int* B, int n) {

	// Base case
	if(n <= 64) {
    	return multiplyMatrix(A, B, n, n, n);
	}

	// Initialize matrix C to return in output (C = A*B)
	int* C = allocateSquareMatrixMemory(n);

	// Dimension of the matrices is the half of the input size
	int k = n/2;

	// Decompose A and B into 8 submatrices
	int* A11 = allocateSquareMatrixMemory(k);
	int* A12 = allocateSquareMatrixMemory(k);
	int* A21 = allocateSquareMatrixMemory(k);
	int* A22 = allocateSquareMatrixMemory(k);
	int* B11 = allocateSquareMatrixMemory(k);
	int* B12 = allocateSquareMatrixMemory(k);
	int* B21 = allocateSquareMatrixMemory(k);
	int* B22 = allocateSquareMatrixMemory(k);

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

    const int n = 1024;

    int rank, size;
    double start, end;

    int *A, *B, *C;
    A = (int*) malloc(n * n * sizeof(int));
    B = (int*) malloc(n * n * sizeof(int));
    C = (int*) malloc(n * n * sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                A[i*n+j] = i+j;
                B[i*n+j] = i-j;
            }
        }

        start = MPI_Wtime();
        strassenMatrix(A, B, n);
        end = MPI_Wtime();

        printf("Time took strassen: %f ms\n", (end-start)*1000);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Scatter(A, n*n/size, MPI_INT, A, n*n/size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(B, n*n, MPI_INT, 0, MPI_COMM_WORLD);
    
    for(int i=0; i<n/size; i++) {
        for(int j=0; j<n; j++) {
            int index = i*n+j;
            C[index] = A[i*n] * B[j];
            for(int k=1; k<n; k++) {
                C[index] += A[i*n+k] * B[k*n+j];
            }
        }
    }

    MPI_Gather(C, n*n/size, MPI_INT, C, n*n/size, MPI_INT, 0, MPI_COMM_WORLD);

    free(A);
    free(B);
    free(C);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    MPI_Finalize();

    if(rank == 0) {
        printf("Time took parallel MM: %f ms\n", (end-start)*1000);
    }

    return 0;
}