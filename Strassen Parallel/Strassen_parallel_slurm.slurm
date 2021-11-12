#!/bin/bash

#SBATCH --job-name strassen_parallel
#SBATCH --error strassen_parallel_err_%j.txt
#SBATCH --output strassen_parallel_out_%j.txt
#SBATCH --partition allgroups
#SBATCH --ntasks 4
#SBATCH -c 1
#SBATCH --mem 4G
#SBATCH --time 10:00

spack load intel-parallel-studio@professional.2019.4

mpirun -np 4 ./Strassen_parallel.o


