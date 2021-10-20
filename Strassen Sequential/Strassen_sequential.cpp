#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;
typedef vector<vector<int>> matrix;

// Initialize square matrix filling it with zeros
matrix initializeMatrix(int n) {

	matrix temp;

	for(int i=0; i<n; i++){

		vector<int> elements;

		for(int j=0; j<n; j++){
			elements.push_back(0);
		}

		temp.push_back(elements);	
	}

	return temp;
}

// Initialize generic matrix filling it with zeros
matrix initializeMatrix(int row, int col) {

	matrix temp;

	for(int i=0; i<row; i++){

		vector<int> elements;

		for(int j=0; j<col; j++){
			elements.push_back(0);
		}

		temp.push_back(elements);	
	}

	return temp;
}

// Add two square matrices
matrix addMatrix(matrix M1, matrix M2, int n) {

	matrix temp = initializeMatrix(n);

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			temp[i][j] = M1[i][j] + M2[i][j];
		}
	}

    return temp;
}

// Subtract two square matrices
matrix subtractMatrix(matrix M1, matrix M2, int n) {

	matrix temp = initializeMatrix(n);

    for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			temp[i][j] = M1[i][j] - M2[i][j];
		}  
	}
        
    return temp;
}

// Multiply two matrices using standard definition
matrix multiplyMatrix(matrix A, matrix B, int rA, int cA, int cB) {

	matrix C = initializeMatrix(rA, cB);

	for(int i = 0; i < rA; i++) {
		for(int j = 0; j < cB; j++) {
			for(int k = 0; k < cA; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}   
	}

	return C;
}

// Print matrix in output
void showMatrix(matrix M, int row, int col) {

	for(int i=0; i<row; i++) {
		for(int j=0; j<col; j++) {
			cout<<M[i][j]<<" ";
		}
		cout<<"\n";
	}

	cout<<"\n";
}

// Multiply two matrices using Strassen algorithm
matrix strassenMatrix(matrix A, matrix B, int n) {

	// Initialize matrix C to return in output (C = A*B)
	matrix C = initializeMatrix(n);

	// Base case
	if(n == 1) {
		C[0][0] = A[0][0] * B[0][0];
    	return C;
	}

	// Dimension of the matrices is the half of the input size
	int k = n/2;

	// Decompose A and B into 8 submatrices
	matrix A11 = initializeMatrix(k);
	matrix A12 = initializeMatrix(k);
	matrix A21 = initializeMatrix(k);
	matrix A22 = initializeMatrix(k);
	matrix B11 = initializeMatrix(k);
	matrix B12 = initializeMatrix(k);
	matrix B21 = initializeMatrix(k);
	matrix B22 = initializeMatrix(k);

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

	// Create the Strassen matrices
	matrix P1 = strassenMatrix(A11, subtractMatrix(B12, B22, k), k);
	matrix P2 = strassenMatrix(addMatrix(A11, A12, k), B22, k);
	matrix P3 = strassenMatrix(addMatrix(A21, A22, k), B11, k);
	matrix P4 = strassenMatrix(A22, subtractMatrix(B21, B11, k), k);
	matrix P5 = strassenMatrix(addMatrix(A11, A22, k), addMatrix(B11, B22, k), k);
	matrix P6 = strassenMatrix(subtractMatrix(A12, A22, k), addMatrix(B21, B22, k), k);
	matrix P7 = strassenMatrix(subtractMatrix(A11, A21, k), addMatrix(B11, B12, k), k);

	// Compose matrix C from the submatrices
	matrix C11 = subtractMatrix(addMatrix(addMatrix(P5, P4, k), P6, k), P2, k);
	matrix C12 = addMatrix(P1, P2, k);
	matrix C21 = addMatrix(P3, P4, k);
	matrix C22 = subtractMatrix(subtractMatrix(addMatrix(P5, P1, k), P3, k), P7, k);

	for(int i=0; i<k; i++) {
		for(int j=0; j<k; j++) {
			C[i][j] = C11[i][j];
			C[i][j+k] = C12[i][j];
			C[k+i][j] = C21[i][j];
			C[k+i][k+j] = C22[i][j];
    	}
	}
    
	return C;
}

int main() {

	int rA, cA, rB, cB;
	matrix A, B, C;

	cout<<"\n Enter number of rows for matrix A: ";
    cin>>rA;
   	cout<<"\n Enter number of columns for matrix A: ";
    cin>>cA;

   	cout<<"\n Enter number of rows for matrix B: ";
    cin>>rB;
   	cout<<"\n Enter number of columns for matrix B: ";
    cin>>cB;

	// If number of column of A are not equal to number of rows of B the matrices are incompatible
	if(cA != rB) {
		cout<<"Error: incompatible matrices";
		return -1;
	}

	int x;
	cout<<"\n Enter 0 to insert elements of matrices or enter 1 to generate matrices randomly \n";
	cin>>x;

	if(x == 0) {
		cout<<"\n Enter elements of matrix A \n";

		for(int i=0; i<rA; i++){

			vector<int> elements;

			for(int j=0; j<cA; j++){
				int element;
				cin>>element;
				elements.push_back(element);
			}

			A.push_back(elements);
		}

		cout<<"\n Enter elements of matrix B \n";

		for(int i=0; i<rB; i++){

			vector<int> elements;

			for(int j=0; j<cB; j++){
				int element;
				cin>>element;
				elements.push_back(element);
			}

			B.push_back(elements);
		}
	}
	else if(x == 1) {

		srand(time(NULL));

		for(int i=0; i<rA; i++){

			vector<int> elements;

			for(int j=0; j<cA; j++) {
				int element = rand()%101;
				elements.push_back(element);
			}

			A.push_back(elements);
		}

		for(int i=0; i<rB; i++){

			vector<int> elements;

			for(int j=0; j<cB; j++) {
				int element = rand()%101;
				elements.push_back(element);
			}

			B.push_back(elements);
		}
	}

	// If the dimension of matrices is not a power of 2
	// get the smallest dimension to multiply matrices using Strassen
	// and fill matrices with zeros if necessary
	int n = pow(2, ceil(log2(max({rA, cA, rB, cB}))));

	for(int i=0; i<n; i++) {

		if(i<rA) {

			for(int j=cA; j<n; j++) {
				A[i].push_back(0);
			}

		} else {

			vector<int> elements;
			for(int k=0; k<n; k++) {
				elements.push_back(0);
			}

			A.push_back(elements);
		}
		
	}

	for(int i=0; i<n; i++) {

		if(i<rB) {

			for(int j=cB; j<n; j++) {
				B[i].push_back(0);
			}

		} else {

			vector<int> elements;
			for(int k=0; k<n; k++) {
				elements.push_back(0);
			}

			B.push_back(elements);
		}
		
	}

	// Multiply matrices using standard definition
	//auto start = chrono::high_resolution_clock::now();

	//C = multiplyMatrix(A, B, rA, cA, cB);

	//auto stop = chrono::high_resolution_clock::now();

	//auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	//showMatrix(C, rA, cB);
	//cout <<"Time for multiplying matrices using definition: "<<duration.count()<<" ms"<< endl;

	// Multiply matrices using Strassen and meausure time
	auto start = chrono::high_resolution_clock::now();

	C = strassenMatrix(A, B, n);

	auto stop = chrono::high_resolution_clock::now();

	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	showMatrix(C, rA, cB);
	cout <<"Time for multiplying matrices using Strassen: "<<duration.count()<<" ms"<< endl;
	
	return 0;
}