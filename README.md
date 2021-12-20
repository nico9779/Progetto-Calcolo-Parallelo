# Progetto-Calcolo-Parallelo

The repository is structured in this way:

- The file `Report Parallel Computing.docx` is the report for the project
- The file `Execution times on CAPRI.xlxs` contains tables with the execution times of the sequential 
   and parallel algorithms and the charts on the metrics evaluated

- The directory `Strassen Parallel` contains all the source code for the parallel algorithms:
	- `Matrix_multiplication_parallel.c` is the parallel standard multiplication algorithm
	- `Strassen_seq_par.c` is the version that compare the sequential and parallel Strassen algorithm
	- `Strassen_parallel.c` is the parallel version of the Strassen algorithm
	- `Strassen_parallel_recursiveCalls_4p.c` and `Strassen_parallel_recursiveCalls_8p.c` are the alternative
		version of the Strassen parallel algorithm respectively implemented on 4 and 8 processes
	- `Strassen_parallel_slurm.slurm` is the slurm file to run the algorithm on CAPRI

- The directory `Strassen Sequential` contains all the source code for the sequential algorithms:
	- `MM_sequential_capri.c` contains different implementation for the standard sequential matrix multiplication algorithm
	- `Strassen_sequential.c` is the sequential version of the Strassen algorithm
	- `Strassen_sequential_pointers_C.c` implement a less constrained version of the sequential Strassen algorithm letting the user
	input the matrices of whatever size
	
To compile and execute the code use the following commands:	
```
$ spack load intel-parallel-studio@professional.2019.4
$ mpicc Strassen_parallel.c -o Strassen_parallel.o -O3
$ sbatch Strassen_parallel_slurm.slurm
```
