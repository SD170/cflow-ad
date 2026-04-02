#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 10        # number of cores 
#SBATCH --mem=200G   # memory in GB
#SBATCH -G 1   # number of GPUs
#SBATCH -t 1-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -A class_cse579spring2026 # class account
#SBATCH -q class      # QOS
#SBATCH -o logs/cflow-btad-01-%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="sdutta60@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment



echo "Starting the script..."
cd $SCRATCH/cflow-ad

echo "Loading required envs..."
module load mamba/latest
echo "Activating the env..."
source activate pytorch-gpu-2.3.1-cuda-12.1

echo "Installing dependencies..."
echo "Installing pip..."
python -m pip install -U pip
echo "Installing requirements..."
python -m pip install -r requirements.txt
echo "Installing FrEIA..."
python -m pip install "git+https://github.com/vislearn/FrEIA.git@cc5cf5ebee08f9bb762bab5a6535c11d19ccb026"

echo "Training the model..."
python main.py --gpu 0 --pro -inp 512 --dataset btad --class-name 01


# with AUPRO
# python main.py --gpu 0 --pro -inp 512 --dataset btad --class-name 01
# python main.py --gpu 0 --pro -inp 512 --dataset btad --class-name 02
# python main.py --gpu 0 --pro -inp 512 --dataset btad --class-name 03

# wihtout AUPRO
# python main.py --gpu 0 -inp 512 --dataset btad --class-name 01
# python main.py --gpu 0 -inp 512 --dataset btad --class-name 02
# python main.py --gpu 0 -inp 512 --dataset btad --class-name 03