#!/bin/bash

#SBATCH --job-name=SL_test
#SBATCH --output=./logs/%x-%A-%a.out
#SBATCH --error=./logs/%x-%A-%a.err

#SBATCH --partition=gpu           
#SBATCH --time=00:05:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks=1             ### Number of tasks per array job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --gpus=1              ### General Reservation of gpu:number of gpus

#SBATCH --account=tau  ### Account used for job submission
#SBATCH --array=0-4           ### Array index

module load cuda/13.0

conda activate talapas_mandi_env

echo "=========================================================="
echo "Job Info: Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES}"
python -c "import jax; print(jax.devices())"
echo "=========================================================="