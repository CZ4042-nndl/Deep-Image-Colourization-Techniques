#!/bin/bash
#SBATCH -p SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --job-name=job2
#SBATCH --output=job_output_2.log
#SBATCH --error=error_2.err
#SBATCH --mem=64G

module load anaconda
source activate NNDL_env
cd /home/FYP/harshrao001/CZ4042/
python Colorization_with_only_encoder_decoder.py
