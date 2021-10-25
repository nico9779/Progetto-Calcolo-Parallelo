#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Find maximum between four values
int findMax(int values[]) {

    int max = 0;

    for(int i=0; i<4; i++) {
		int value = values[i];
		if(value > max) 
			max = value;
	}

    return max;
}

// Allocate memory to store square matrix
int** allocateSquareMatrixMemory(int n) {

    int** temp = (int**) malloc(n * sizeof(int*));
    for(int i=0; i<n; i++) {
        temp[i] = (int*) malloc(n * sizeof(int));
    }

    return temp;
}

// Allocate memory to store matrix
int** allocateMatrixMemory(int row, int col) {

    int** temp = (int**) malloc(row * sizeof(int*));
    for(int i=0; i<row; i++) {
        temp[i] = (int*) malloc(col * sizeof(int));
    }

    return temp;
}

// Deallocate matrix memory
void deallocateMatrixMemory(int** M, int n) {

	for(int i=0; i<n; i++) {
		free(M[i]);
	}

	free(M);
}

// Add two square matrices
int** addMatrix(int** M1, int** M2, int n) {

	int** temp = allocateSquareMatrixMemory(n);

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			temp[i][j] = M1[i][j] + M2[i][j];
		}
	}

    return temp;
}

// Subtract two square matrices
int** subtractMatrix(int** M1, int** M2, int n) {

	int** temp = allocateSquareMatrixMemory(n);

    for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			temp[i][j] = M1[i][j] - M2[i][j];
		}  
	}
        
    return temp;
}

// Multiply two matrices using standard definition
int** multiplyMatrix(int** A, int** B, int rA, int cA, int cB) {

	int** C = allocateMatrixMemory(rA, cB);

	for(int i = 0; i < rA; i++) {
		for(int j = 0; j < cB; j++) {
            C[i][j] = A[i][0] * B[0][j];
			for(int k = 1; k < cA; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	return C;
}

// Print matrix in output
void showMatrix(int** M, int row, int col) {

	for(int i=0; i<row; i++) {
		for(int j=0; j<col; j++) {
			printf("%d ", M[i][j]);
		}
		printf("\n");
	}

	printf("\n");
}

// Multiply two matrices using Strassen algorithm
int** strassenMatrix(int** A, int** B, int n) {

	// Base case
	if(n <= 64) {
    	return multiplyMatrix(A, B, n, n, n);
	}

	// Initialize matrix C to return in output (C = A*B)
	int** C = allocateSquareMatrixMemory(n);

	// Dimension of the matrices is the half of the input size
	int k = n/2;

	// Decompose A and B into 8 submatrices
	int** A11 = allocateSquareMatrixMemory(k);
	int** A12 = allocateSquareMatrixMemory(k);
	int** A21 = allocateSquareMatrixMemory(k);
	int** A22 = allocateSquareMatrixMemory(k);
	int** B11 = allocateSquareMatrixMemory(k);
	int** B12 = allocateSquareMatrixMemory(k);
	int** B21 = allocateSquareMatrixMemory(k);
	int** B22 = allocateSquareMatrixMemory(k);

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			A11[i][j] = A[i][j];
			A12[i][j] = A[i][k+j];
			A21[i][j] = A[k+i][j];
			A22[i][j] = A[k+i][k+j];
			B11[i][j] = B[i][j];
			B12[i][j] = B[i][k+j];
			B21[i][j] = B[k+i][j];
			B22[i][j] = B[k+i][k+j];
    	}
	}

	// Create support matrices in order to calculate Strassen matrices
	int** M1 = subtractMatrix(B12, B22, k);
	int** M2 = addMatrix(A11, A12, k);
	int** M3 = addMatrix(A21, A22, k);
	int** M4 = subtractMatrix(B21, B11, k);
	int** M5 = addMatrix(A11, A22, k);
	int** M6 = addMatrix(B11, B22, k);
	int** M7 = subtractMatrix(A12, A22, k);
	int** M8 = addMatrix(B21, B22, k);
	int** M9 = subtractMatrix(A11, A21, k);
	int** M10 = addMatrix(B11, B12, k);

	// Create the Strassen matrices
	int** P1 = strassenMatrix(A11, M1, k);
	int** P2 = strassenMatrix(M2, B22, k);
	int** P3 = strassenMatrix(M3, B11, k);
	int** P4 = strassenMatrix(A22, M4, k);
	int** P5 = strassenMatrix(M5, M6, k);
	int** P6 = strassenMatrix(M7, M8, k);
	int** P7 = strassenMatrix(M9, M10, k);

    deallocateMatrixMemory(A11, k);
    deallocateMatrixMemory(A12, k);
    deallocateMatrixMemory(A21, k);
    deallocateMatrixMemory(A22, k);
    deallocateMatrixMemory(B11, k);
    deallocateMatrixMemory(B12, k);
    deallocateMatrixMemory(B21, k);
    deallocateMatrixMemory(B22, k);

	deallocateMatrixMemory(M1, k);
	deallocateMatrixMemory(M2, k);
	deallocateMatrixMemory(M3, k);
	deallocateMatrixMemory(M4, k);
	deallocateMatrixMemory(M5, k);
	deallocateMatrixMemory(M6, k);
	deallocateMatrixMemory(M7, k);
	deallocateMatrixMemory(M8, k);
	deallocateMatrixMemory(M9, k);
	deallocateMatrixMemory(M10, k);

	int** M11 = addMatrix(P5, P4, k);
	int** M12 = addMatrix(M11, P6, k);
	int** M13 = addMatrix(P5, P1, k);
	int** M14 = subtractMatrix(M13, P3, k);

	// Compose matrix C from the submatrices
	int** C11 = subtractMatrix(M12, P2, k);
	int** C12 = addMatrix(P1, P2, k);
	int** C21 = addMatrix(P3, P4, k);
	int** C22 = subtractMatrix(M14, P7, k);

    deallocateMatrixMemory(P1, k);
    deallocateMatrixMemory(P2, k);
    deallocateMatrixMemory(P3, k);
    deallocateMatrixMemory(P4, k);
    deallocateMatrixMemory(P5, k);
    deallocateMatrixMemory(P6, k);
    deallocateMatrixMemory(P7, k);

	deallocateMatrixMemory(M11, k);
	deallocateMatrixMemory(M12, k);
	deallocateMatrixMemory(M13, k);
	deallocateMatrixMemory(M14, k);

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			C[i][j] = C11[i][j];
			C[i][j+k] = C12[i][j];
			C[k+i][j] = C21[i][j];
			C[k+i][k+j] = C22[i][j];
    	}
	}

    deallocateMatrixMemory(C11, k);
    deallocateMatrixMemory(C12, k);
    deallocateMatrixMemory(C21, k);
    deallocateMatrixMemory(C22, k);
    
	return C;
}

int main() {

	int rA, cA, rB, cB;
	int** A;
    int** B;
    int** C;
    clock_t start, end;
    double duration;

	printf("\n Enter number of rows for matrix A: ");
    scanf("%d", &rA);
   	printf("\n Enter number of columns for matrix A: ");
    scanf("%d", &cA);

   	printf("\n Enter number of rows for matrix B: ");
    scanf("%d", &rB);
   	printf("\n Enter number of columns for matrix B: ");
    scanf("%d", &cB);

	// If number of column of A are not equal to number of rows of B the matrices are incompatible
	if(cA != rB) {
		printf("Error: incompatible matrices");
		return -1;
	}

    // If the dimension of matrices is not a power of 2
	// get the smallest dimension to multiply matrices using Strassen
	// and fill matrices with zeros if necessary
	int values[] = {rA, cA, rB, cB};
	int n = pow(2, ceil(log2(findMax(values))));

    A = allocateSquareMatrixMemory(n);
    B = allocateSquareMatrixMemory(n);

	int x;
	printf("\n Enter 0 to insert elements of matrices or enter 1 to generate matrices randomly \n");
	scanf("%d", &x);

	if(x == 0) {
		printf("\n Enter elements of matrix A \n");

		for(int i=0; i<rA; i++){
			for(int j=0; j<cA; j++){
				scanf("%d", &A[i][j]);
			}
		}

		printf("\n Enter elements of matrix B \n");

		for(int i=0; i<rB; i++){
			for(int j=0; j<cB; j++){
				scanf("%d", &B[i][j]);
			}
		}
	}
	else if(x == 1) {

		srand(time(NULL));

		for(int i=0; i<rA; i++){
			for(int j=0; j<cA; j++) {
				A[i][j] = rand()%101;
			}
		}

		for(int i=0; i<rB; i++){
			for(int j=0; j<cB; j++) {
				B[i][j] =  rand()%101;
			}
		}
	}

	for(int i=0; i<n; i++) {

		if(i<rA) {

			for(int j=cA; j<n; j++) {
				A[i][j] = 0;
			}

		} else {
			for(int k=0; k<n; k++) {
				A[i][k] = 0;
			}
		}
	}

	for(int i=0; i<n; i++) {

		if(i<rB) {

			for(int j=cB; j<n; j++) {
				B[i][j] = 0;
			}

		} else {

			for(int k=0; k<n; k++) {
				B[i][k] = 0;
			}
		}
	}

	// Multiply matrices using standard definition
	//start = clock();

	//C = multiplyMatrix(A, B, rA, cA, cB);

	//end = clock();

	//duration = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	//showMatrix(C, rA, cB);
	//printf("Time for multiplying matrices using definition: %f ms\n", duration);

	// Best value for the base case is 64
	// for(int i=2; i<11; i++) {
	// 	int base = pow(2, i);
	// 	printf("base: %d\n", base);

	// 	start = clock();
	// 	C = strassenMatrix(A, B, n, base);
	// 	end = clock();
	// 	duration = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	// 	printf("Time for multiplying matrices using Strassen: %f ms\n", duration);
	// }

	// Multiply matrices using Strassen and meausure time
	start = clock();

	C = strassenMatrix(A, B, n);

	end = clock();

	duration = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	//showMatrix(C, rA, cB);
	printf("Time for multiplying matrices using Strassen: %f ms\n", duration);

	deallocateMatrixMemory(A, n);
	deallocateMatrixMemory(B, n);
	deallocateMatrixMemory(C, n);
	
	return 0;
}