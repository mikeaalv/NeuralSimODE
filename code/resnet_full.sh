#!/bin/bash
#SBATCH -J testmodel                               # Job name
#SBATCH -p GPU                                      # Queue (partition) name
#SBATCH -N 1                                       # Total # of nodes
#SBATCH -t 10:00:00                                # Run time (hh:mm:ss)
#SBATCH --gres=gpu:p100:2
#SBATCH --output=testmodel.%j.out   # Name of stdout output file
#SBATCH --error=testmodel.%j.err    # Name of stderr error file
#SBATCH --mail-user=Yue.Wu@uga.edu
#SBATCH --mail-type=ALL                            # Send email at begin and end of job

#echo commands to stdout
set -x
pwd
date

#interactive
# interact -p GPU --gres=gpu:p100:2 -N 1 -t 1:00:00
#load conda environment
. /opt/packages/anaconda/anaconda3-5.2.0/etc/profile.d/conda.sh
#load pytorch
conda activate /home/mikeaalv/method/pytorch

time python3 train_mlp_full.py --batch-size 50000 --test-batch-size 50000 --epochs 100 --learning-rate 0.01 --seed 1 --net-struct 'resnet18_mlp'
