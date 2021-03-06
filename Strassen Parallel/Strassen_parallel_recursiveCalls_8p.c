/*******	PARALLEL STRASSEN MATRIX MULTIPLICATION ALGORITHM USING 8 PROCESSES	******/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Size of the sub-matrices
#define CHUNK 32

// Find the average in a vector of doubles
double average(double vector[], int size) {

    double average = 0;

    for(int i=0; i<size; i++) {
        average += vector[i];
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

// Multiply two matrices sequentially using sub-matrices method and loop unrolling
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

// Multiply two matrices using Strassen algorithm
float* strassenMultiplication(float* A, float* B, int n) {

	// Base case
    // Compute directly the product between matrices A and B
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

	// Dimension of the submatrices is the half of the input size
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

	// Compute support matrices in order to calculate Strassen matrices
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

	// Compute the Strassen matrices calling recursively the Strassen method
	float* P1 = strassenMultiplication(A11, M1, k);
	float* P2 = strassenMultiplication(M2, B22, k);
	float* P3 = strassenMultiplication(M3, B11, k);
	float* P4 = strassenMultiplication(A22, M4, k);
	float* P5 = strassenMultiplication(M5, M6, k);
	float* P6 = strassenMultiplication(M7, M8, k);
	float* P7 = strassenMultiplication(M9, M10, k);

    // Clear memory to avoid memory leak
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

    const int n = 4096;                 // n is the size of the matrix
    const int k = n/2;                  // k is the size of the submatrices
    const int num_elements = k*k;       // number of elements of the submatrices to exchange between processes
    const int repetition = 5;           // number of time to repeat the multiplication and obtain an average execution time

    int rank, size;                     // rank identify the process and size is the number of processes available
    MPI_Request req[8];
    MPI_Status status;
    double start, end, comm_time, start_comm, end_comm;     // variables used to evaluate execution and communication time
    double execution[repetition], communication[repetition];    // array containing execution and communication times of the algorithm for differents execution

	float *A, *B;

	float *P1 = NULL;
    float *P2 = NULL;
    float *P3 = NULL;
    float *P4 = NULL;
    float *P5 = NULL;
    float *P6 = NULL;
    float *P7 = NULL;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // The master process initialize the matrices A and B
    if(rank == 0) {

        // Generate matrices A and B
        A = (float*) malloc(n * n * sizeof(float));
        B = (float*) malloc(n * n * sizeof(float));

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                A[i*n+j] = i+j;
                B[i*n+j] = i-j;
            }
        }
    }

    // Compare the sequential and parallel version of the algorithm

    //***** SEQUENTIAL PHASE *****//

    if(rank == 0) {  

        for(int i=0; i<repetition; i++) {

            // Calculate time to execute Strassen sequential algorithm
            start = MPI_Wtime();
            float *C = strassenMultiplication(A, B, n);
            end = MPI_Wtime();

            execution[i] = (end-start)*1000;

            free(C);
        }

        printf("Average time took sequential Strassen: %f ms\n", average(execution, repetition));
    }

    //***** PARALLEL PHASE *****//

    for(int i=0; i<repetition; i++) {
        
        // Reset to 0 the communication time
	    comm_time = 0;

        // Before starting measuring the execution time I call MPI_Barrier to make sure that all processes are ready to proceed
	    MPI_Barrier(MPI_COMM_WORLD);
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

            // #0 compute P1 and P6
            float *T1, *T2, *T3, *T4, *T5, *T6, *T7, *T8, *T9, *T10, *T11, *T12, *T13, *T14;

            T1 = addMatrix(A11, A22, k);
            T2 = addMatrix(B11, B22, k);
            T3 = addMatrix(A21, A22, k);
            T4 = B11;
            T5 = A11;
            T6 = subtractMatrix(B12, B22, k);
            T7 = A22;
            T8 = subtractMatrix(B21, B11, k);
            T9 = addMatrix(A11, A12, k);
            T10 = B22;
            T11 = subtractMatrix(A21, A11, k);
            T12 = addMatrix(B11, B12, k);
            T13 = subtractMatrix(A12, A22, k);
            T14 = addMatrix(B21, B22, k);

            // Start evaluation communication time
	        start_comm = MPI_Wtime();

            // Send matrices to other processors
            MPI_Isend(T1, num_elements, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Isend(T2, num_elements, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Isend(T3, num_elements, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &req[2]);
            MPI_Isend(T4, num_elements, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &req[2]);
            MPI_Isend(T5, num_elements, MPI_FLOAT, 3, 3, MPI_COMM_WORLD, &req[3]);
            MPI_Isend(T6, num_elements, MPI_FLOAT, 3, 3, MPI_COMM_WORLD, &req[3]);
            MPI_Isend(T7, num_elements, MPI_FLOAT, 4, 4, MPI_COMM_WORLD, &req[4]);
            MPI_Isend(T8, num_elements, MPI_FLOAT, 4, 4, MPI_COMM_WORLD, &req[4]);
            MPI_Isend(T9, num_elements, MPI_FLOAT, 5, 5, MPI_COMM_WORLD, &req[5]);
            MPI_Isend(T10, num_elements, MPI_FLOAT, 5, 5, MPI_COMM_WORLD, &req[5]);
            MPI_Isend(T11, num_elements, MPI_FLOAT, 6, 6, MPI_COMM_WORLD, &req[6]);
            MPI_Isend(T12, num_elements, MPI_FLOAT, 6, 6, MPI_COMM_WORLD, &req[6]);
            MPI_Isend(T13, num_elements, MPI_FLOAT, 7, 7, MPI_COMM_WORLD, &req[7]);
            MPI_Isend(T14, num_elements, MPI_FLOAT, 7, 7, MPI_COMM_WORLD, &req[7]);
 			
            // Stop evaluation communication time
 	        end_comm = MPI_Wtime();

            // Compute communication time in ms
	        comm_time += (end_comm-start_comm)*1000;

            free(A12);
            free(A21);
            free(B12);
            free(B21);

            free(T1);
            free(T2);
            free(T3);
            free(T4);
            free(T5);
            free(T6);
            free(T7);
            free(T8);
            free(T9);
            free(T10);
            free(T11);
            free(T12);
            free(T13);
            free(T14);
        }

        // Each process compute one product 

        if(rank == 1) {
            float *T1, *T2;
            T1 = (float*) malloc(num_elements * sizeof(float));
            T2 = (float*) malloc(num_elements * sizeof(float));

            // Receive matrices from #0
            MPI_Irecv(T1, num_elements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Irecv(T2, num_elements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Wait(&req[1], &status);

            P1 = strassenMultiplication(T1, T2, k);

            // Send results to #0
            MPI_Isend(P1, num_elements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req[1]);

            free(T1);
            free(T2);
        }

        if(rank == 2) {
            float *T3, *T4;
            T3 = (float*) malloc(num_elements * sizeof(float));
            T4 = (float*) malloc(num_elements * sizeof(float));

            // Receive matrices from #0
            MPI_Irecv(T3, num_elements, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &req[2]);
            MPI_Irecv(T4, num_elements, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &req[2]);
            MPI_Wait(&req[2], &status);
            
            P2 = strassenMultiplication(T3, T4, k);

            // Send results to #0
            MPI_Isend(P2, num_elements, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &req[2]);

            free(T3);
            free(T4);
        }

        if(rank == 3) {
            float *T5, *T6;
            T5 = (float*) malloc(num_elements * sizeof(float));
            T6 = (float*) malloc(num_elements * sizeof(float));

            // Receive matrices from #0
            MPI_Irecv(T5, num_elements, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &req[3]);
            MPI_Irecv(T6, num_elements, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &req[3]);
            MPI_Wait(&req[3], &status);

            P3 = strassenMultiplication(T5, T6, k);

            // Send results to #0
            MPI_Isend(P3, num_elements, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &req[3]);

            free(T5);
            free(T6);
        }

        if(rank == 4) {
            float *T7, *T8;
            T7 = (float*) malloc(num_elements * sizeof(float));
            T8 = (float*) malloc(num_elements * sizeof(float));

            // Receive matrices from #0
            MPI_Irecv(T7, num_elements, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &req[4]);
            MPI_Irecv(T8, num_elements, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &req[4]);
            MPI_Wait(&req[4], &status);
            
            P4 = strassenMultiplication(T7, T8, k);

            // Send results to #0
            MPI_Isend(P4, num_elements, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &req[4]);

            free(T7);
            free(T8);
        }

        if(rank == 5) {
            float *T9, *T10;
            T9 = (float*) malloc(num_elements * sizeof(float));
            T10 = (float*) malloc(num_elements * sizeof(float));

            // Receive matrices from #0
            MPI_Irecv(T9, num_elements, MPI_FLOAT, 0, 5, MPI_COMM_WORLD, &req[5]);
            MPI_Irecv(T10, num_elements, MPI_FLOAT, 0, 5, MPI_COMM_WORLD, &req[5]);
            MPI_Wait(&req[5], &status);
            
            P5 = strassenMultiplication(T9, T10, k);

            // Send results to #0
            MPI_Isend(P5, num_elements, MPI_FLOAT, 0, 5, MPI_COMM_WORLD, &req[5]);

            free(T9);
            free(T10);
        }

        if(rank == 6) {
            float *T11, *T12;
            T11 = (float*) malloc(num_elements * sizeof(float));
            T12 = (float*) malloc(num_elements * sizeof(float));

            // Receive matrices from #0
            MPI_Irecv(T11, num_elements, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, &req[6]);
            MPI_Irecv(T12, num_elements, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, &req[6]);
            MPI_Wait(&req[6], &status);
            
            P6 = strassenMultiplication(T11, T12, k);

            // Send results to #0
            MPI_Isend(P6, num_elements, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, &req[6]);

            free(T11);
            free(T12);
        }

        if(rank == 7) {
            float *T13, *T14;
            T13 = (float*) malloc(num_elements * sizeof(float));
            T14 = (float*) malloc(num_elements * sizeof(float));

            // Receive matrices from #0
            MPI_Irecv(T13, num_elements, MPI_FLOAT, 0, 7, MPI_COMM_WORLD, &req[7]);
            MPI_Irecv(T14, num_elements, MPI_FLOAT, 0, 7, MPI_COMM_WORLD, &req[7]);
            MPI_Wait(&req[7], &status);
            
            P7 = strassenMultiplication(T13, T14, k);

            // Send results to #0
            MPI_Isend(P7, num_elements, MPI_FLOAT, 0, 7, MPI_COMM_WORLD, &req[7]);

            free(T13);
            free(T14);
        }


        if(rank == 0) {
            float *C11, *C12, *C21, *C22, *C;
            C = (float*) malloc(n * n * sizeof(float));

            P1 = (float*) malloc(num_elements * sizeof(float));
            P2 = (float*) malloc(num_elements * sizeof(float));
            P3 = (float*) malloc(num_elements * sizeof(float));
            P4 = (float*) malloc(num_elements * sizeof(float));
            P5 = (float*) malloc(num_elements * sizeof(float));
            P6 = (float*) malloc(num_elements * sizeof(float));
            P7 = (float*) malloc(num_elements * sizeof(float));
            
            // Start evaluating communication time
	        start_comm = MPI_Wtime();

            // Receive matrices from other processes
            MPI_Irecv(P1, num_elements, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Irecv(P2, num_elements, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &req[2]);
            MPI_Irecv(P3, num_elements, MPI_FLOAT, 3, 3, MPI_COMM_WORLD, &req[3]);
            MPI_Irecv(P4, num_elements, MPI_FLOAT, 4, 4, MPI_COMM_WORLD, &req[4]);
            MPI_Irecv(P5, num_elements, MPI_FLOAT, 5, 5, MPI_COMM_WORLD, &req[5]);
            MPI_Irecv(P6, num_elements, MPI_FLOAT, 6, 6, MPI_COMM_WORLD, &req[6]);
            MPI_Irecv(P7, num_elements, MPI_FLOAT, 7, 7, MPI_COMM_WORLD, &req[7]);

            MPI_Wait(&req[1], &status);
            MPI_Wait(&req[2], &status);
            MPI_Wait(&req[3], &status);
            MPI_Wait(&req[4], &status);
            MPI_Wait(&req[5], &status);
            MPI_Wait(&req[6], &status);
            MPI_Wait(&req[7], &status);

            // Stop evaluating communication time
            end_comm = MPI_Wtime();

            // Compute communication time in ms
            comm_time += (end_comm-start_comm)*1000;	

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

            //printMatrix(C, n);

            free(C);
        }

        // Stop measuring execution time and call MPI_Barrier to make sure all nodes are done
        MPI_Barrier(MPI_COMM_WORLD);
	    end = MPI_Wtime();

        // Calculate execution and communication time in ms and push it in the array
        execution[i] = (end-start)*1000;
	    communication[i] = comm_time;

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
    }

    MPI_Finalize();

    // Print results
	if(rank == 0) {
        free(A);
        free(B);
        printf("Average time took parallel Strassen: %f ms\n", average(execution, repetition));
        printf("Average time took for communication: %f ms\n", average(communication, repetition));

    }

    return 0;
}
