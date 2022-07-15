#!/bin/bash
#SBATCH  -J name
#SBATCH  -o a.out
#SBATCH  -e a.err
#SBATCH  -N 1
#SBATCH  -n 16

#module load intel/17.0.4.196
#module load mvapich2/2.2
#module load ohpc-intel-mvapich2

cd path

#prun /home/alfredo/Software/NAMD_Git-2021-03-23_Linux-x86_64-multicore/namd2 +auto-provision namd > 1.log 
/home/alfredo/Software/NAMD_Git-2021-03-23_Linux-x86_64-multicore/namd2 +auto-provision +isomalloc_sync namd > 1.log
