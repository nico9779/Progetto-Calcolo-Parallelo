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
int* allocateSquareMatrixMemory(int n) {

    int* temp = (int*) calloc(n * n, sizeof(int));

    return temp;
}

// Allocate memory to store matrix
int* allocateMatrixMemory(int rows, int cols) {

    int* temp = (int*) calloc(rows * cols, sizeof(int));

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
int* multiplyMatrix1(int* A, int* B, int rA, int cA, int cB) {

	int* C = allocateMatrixMemory(rA, cB);
	int* temp = (int*) malloc(rA * cA * sizeof(int));
	int local_sum, t1, t2, t3, t4;

	for(int i = 0; i < rA; i++){
		for(int j = 0; j < cA; j++){
			temp[i*rA+j] = B[j*rA+i];
		}
	}

	for(int i = 0; i < rA; i++) {
		int* p1 = &A[i*cA];
		for(int j = 0; j < cB; j++) {
			int* p2 = &temp[j*cB];
			local_sum = 0;
			for(int k = 0; k < cA; k+=4) {
				t1 = *(p1+k) * *(p2+k);
				t2 = *(p1+k+1) * *(p2+k+1);
				t3 = *(p1+k+2) * *(p2+k+2);
				t4 = *(p1+k+3) * *(p2+k+3);
				local_sum += (t1+t2)+(t3+t4);
			}
			C[i*cB+j] = local_sum;
		}
	}

	free(temp);

	return C;
}

int* multiplyMatrix(int* A, int* B, int rA, int cA, int cB) {

	int* C = allocateMatrixMemory(rA, cB);
	int a = 0;

	for(int i = 0; i < rA; i++) {
		for(int k = 0; k < cA; k++) {
			a = A[i*cA+k];
			for(int j = 0; j < cB; j+=8) {
				C[i*cB+j] += a * B[k*cB+j];
				C[i*cB+j+1] += a * B[k*cB+j+1];
				C[i*cB+j+2] += a * B[k*cB+j+2];
				C[i*cB+j+3] += a * B[k*cB+j+3];
				C[i*cB+j+4] += a * B[k*cB+j+4];
				C[i*cB+j+5] += a * B[k*cB+j+5];
				C[i*cB+j+6] += a * B[k*cB+j+6];
				C[i*cB+j+7] += a * B[k*cB+j+7];
			}
		}
	}

	return C;
}

// Print matrix in output
void showMatrix(int* M, int rows, int cols) {

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

int main() {

	int rA, cA, rB, cB;
	int *A, *B, *C;
    clock_t start, end;
    double duration;

	printf("\nEnter number of rows for matrix A: ");
    scanf("%d", &rA);
   	printf("\nEnter number of columns for matrix A: ");
    scanf("%d", &cA);

   	printf("\nEnter number of rows for matrix B: ");
    scanf("%d", &rB);
   	printf("\nEnter number of columns for matrix B: ");
    scanf("%d", &cB);

	// If number of column of A are not equal to number of rows of B the matrices are incompatible
	if(cA != rB) {
		printf("Error: incompatible matrices\n");
		return -1;
	}

    // If the dimension of matrices is not a power of 2
	// get the smallest dimension to multiply matrices using Strassen
	// and fill matrices with zeros if necessary
	int values[] = {rA, cA, rB, cB};
	int n = pow(2, ceil(log2(findMax(values))));

	// Allocate memory for matrices A and B
    A = allocateSquareMatrixMemory(n);
    B = allocateSquareMatrixMemory(n);

	int x;
	printf("\nEnter 0 to insert elements of matrices or enter 1 to generate matrices randomly\n");
	scanf("%d", &x);

	if(x == 0) {
		printf("\n Enter elements of matrix A \n");

		for(int i=0; i<rA; i++){
			for(int j=0; j<cA; j++){
				scanf("%d", &A[i*cA+j]);
			}
		}

		printf("\n Enter elements of matrix B \n");

		for(int i=0; i<rB; i++){
			for(int j=0; j<cB; j++){
				scanf("%d", &B[i*cB+j]);
			}
		}
	}
	else if(x == 1) {

		srand(time(NULL));

		for(int i=0; i<rA; i++){
			for(int j=0; j<cA; j++) {
				A[i*cA+j] = rand()%10+1;
			}
		}

		for(int i=0; i<rB; i++){
			for(int j=0; j<cB; j++) {
				B[i*cB+j] =  rand()%10+1;
			}
		}
	}

	// Fill matrices with zeros to make n*n matrices if necessary
	for(int i=0; i<n; i++) {

		if(i<rA) {

			for(int j=cA; j<n; j++) {
				A[i*cA+j] = 0;
			}

		} else {
			for(int k=0; k<n; k++) {
				A[i*cA+k] = 0;
			}
		}
	}

	for(int i=0; i<n; i++) {

		if(i<rB) {

			for(int j=cB; j<n; j++) {
				B[i*cB+j] = 0;
			}

		} else {

			for(int k=0; k<n; k++) {
				B[i*cB+k] = 0;
			}
		}
	}

	// Multiply matrices using standard definition
	start = clock();

	C = multiplyMatrix(A, B, rA, cA, cB);

	end = clock();

	duration = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	//showMatrix(C, rA, cB);
	printf("Time for multiplying matrices using definition: %f ms\n", duration);

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

	// Multiply matrices using Strassen and measure time
	start = clock();

	C = strassenMatrix(A, B, n);

	end = clock();

	duration = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	//showMatrix(C, rA, cB);
	printf("Time for multiplying matrices using Strassen: %f ms\n", duration);

	free(A);
	free(B);
	free(C);
	
	return 0;
}