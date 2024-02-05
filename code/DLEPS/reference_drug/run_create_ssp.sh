#!/bin/bash
#SBATCH --job-name=dleps
#SBATCH --output output_ssp.log
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH --time=12:00:00    # Set the maximum time your job will run (hh:mm:ss)

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda/2020.11  # Change this to the appropriate Anaconda version

# Activate your Python environment
# source activate your_conda_environment  # Change this to the name of your Conda environment

# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
# python create_data.py
python create_ssp.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"