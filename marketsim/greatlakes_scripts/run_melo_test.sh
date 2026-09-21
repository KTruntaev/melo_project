#!/bin/bash
#SBATCH --job-name=egta
#SBATCH --partition=standard
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5g
#SBATCH --account=wellman98

echo "Job Id listed below:"
echo $SLURM_JOB_ID
module load cuda cudnn
module load clang
module load gcc

source venv/bin/activate
python3 -u marketsim/tests/melo_tests/melo_test_1.py