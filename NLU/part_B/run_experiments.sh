#!/bin/bash
#SBATCH -p edu-long
#SBATCH -t 23:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH -o /home/alessandro.devidi/nlu_project/sbatch_out/bert_experiments.out
#SBATCH -e /home/alessandro.devidi/nlu_project/sbatch_out/bert_experiments_err.out

export PATH="/home/alessandro.devidi/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

# call your program here
module load cuda

conda activate nlu25

#cd scripts

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python main.py


wait