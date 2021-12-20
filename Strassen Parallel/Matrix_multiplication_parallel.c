/*******	PARALLEL STANDARD MATRIX MULTIPLICATION ALGORITHM	******/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Size of the sub-matrices
#define CHUNK 32

// Find the average in a vector of doubles
double average(double times[], int size) {

    double average = 0;

    for(int i=0; i<size; i++) {
        average += times[i];
    }

    average = average/size;

    return average;
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

int main(int argc, char **argv) {

    const int n = 4096;				// n is the size of the matrix
	const int repetition = 5;		// number of time to repeat the multiplication and obtain an average execution time

    int rank, size;					// rank identify the process and size is the number of processes available
    double start, end;				// start and end are used to evaluate the execution time
	double times[repetition];		// array containing execution times of the algorithm for differents execution

    float *A, *B, *C;
	B = (float*) malloc(n * n * sizeof(float));

	// Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	const int num_elements = n*n/size;		// number of elements scattered between all processes

	// The master process initialize the matrices A and B
    if(rank == 0) {

		A = (float*) malloc(n * n * sizeof(float));

        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                A[i*n+j] = i+j;
                B[i*n+j] = i-j;
            }
        }
	}

	// Repeat the algorithm a fixed number of times
	for(int i=0; i<repetition; i++) {

		// The final matrix C will only be available on the master process
		if(rank == 0) {
			C = (float*) malloc(n * n * sizeof(float));
		}

		// Before starting measuring the execution time I call MPI_Barrier to make sure that all processes are ready to proceed
		MPI_Barrier(MPI_COMM_WORLD);
		start = MPI_Wtime();

		float* local_A = (float*) malloc(num_elements * sizeof(float));
    	float* local_C = (float*) calloc(num_elements, sizeof(float));

		// Scatter matrix A between all processes
		MPI_Scatter(A, num_elements, MPI_FLOAT, local_A, num_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// Broadcast matrix B between all processes
		MPI_Bcast(B, n*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		float a = 0;

		// Compute the product locally
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

		// Gather matrix C in master process
		MPI_Gather(local_C, num_elements, MPI_FLOAT, C, num_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

		free(local_C);
		
		// Stop measuring execution time and call MPI_Barrier to make sure all nodes are done
		MPI_Barrier(MPI_COMM_WORLD);
		end = MPI_Wtime();

		if(rank == 0) {
			//printMatrix(C, n);
			free(C);
		}
		
		// Calculate execution time in ms and push it in the array
		times[i] = (end-start)*1000;
	}
    
    MPI_Finalize();

	// Print average time to execute the algorithm
    if(rank == 0) {
		free(A);
        printf("Average time took parallel matrix multiplication: %f ms\n", average(times, repetition));
    }

	free(B);

    return 0;
}