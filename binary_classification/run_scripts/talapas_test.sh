#!/bin/bash

#SBATCH --job-name=SL_test
#SBATCH --output=%x-%A-%a.out
#SBATCH --error=%x-%A-%a.err

#SBATCH --partition=gpu           
#SBATCH --time=1-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks=1             ### Number of tasks per array job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --gpus=1              ### General Reservation of gpu:number of gpus
#SBATCH --constraint="gpu-40gb|gpu-80gb|2xgpu-80gb|3xgpu-80gb"

#SBATCH --account=tau  ### Account used for job submission

#SBATCH --mail-type=all
#SBATCH --mail-user=mtrappet@uoregon.edu

#SBATCH --array=0-19           ### Array index

echo "=========================================================="
echo "Job Info: Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================================="