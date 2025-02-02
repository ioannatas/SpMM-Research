#!/bin/bash
#SBATCH --account=ehpc-dev-2024d09-056
#SBATCH --partition=pm6-isw2,pm9-isw0,pm11-isw2
#SBATCH --time 1:00:00                 # format: HH:MM:SS
#SBATCH --nodes 1                            # 1 node
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=256
#SBATCH --mem=200000                    # memory per node out of 246000MB
#SBATCH --job-name=job
#SBATCH --output=job_2.out
#SBATCH --error=job_2.err


ulimit -u
# cd SpMM-Research/benchmark_code/CPU/AMD
> job_2.out
> job_2.err

module load gcc/latest 2>&1
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load mkl/2024.1

# module load intel
# module load tbb/latest
# module load compiler-rt/latest
# module load oclfpga/latest
# module load compiler-intel-llvm/latest
# module load gcc/11.2.0 2>&1
lscpu
# cd pipeline_code_bench
# make clean; make -j
# cd ../

# cd spmv_code_bench
# make clean; make -j
# cd ../
cd sddmm_code_bench
make clean; make -j
cd ../
# cd spmv_code_bench
# make clean; make -j
# cd ../

./run_1.sh
# ./proc_run.sh

# machine_info


