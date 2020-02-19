#PBS -S /bin/bash
#PBS -q gpu_q
#PBS -N resnet_model
#PBS -l nodes=1:ppn=3:gpus=2:default
#PBS -l walltime=200:00:00
#PBS -l mem=60gb
#PBS -M yuewu_mike@163.com
#PBS -m abe
cd $PBS_O_WORKDIR
echo $PBS_GPUFILE


module load PyTorch/1.2.0_conda

# source activate ${PYTORCHROOT}
source activate /home/yw44924/methods/pytorch_env

time python train_mlp_full.py --batch-size 50000 --test-batch-size 50000 --epochs 100 --learning-rate 0.01 --seed 1 --net-struct 'resnet18_mlp' 1>> ./testmodel.${PBS_JOBID}.out 2>> ./testmodel.${PBS_JOBID}.err

source deactivate
