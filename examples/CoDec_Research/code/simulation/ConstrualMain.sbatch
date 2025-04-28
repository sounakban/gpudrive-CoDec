#!/bin/bash
#SBATCH --job-name=ConstrualSimulation
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=ConstrualMain20G30M.out


cd /scratch/sb10438/gpudrive-CoDec/examples/CoDec_Research/code/simulation/
singularity exec --nv --overlay /scratch/sb10438/images/gpudrive/overlay-50G-10M.ext3:rw \
        /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
        /bin/bash -c "source /ext3/env.sh; python ./ConstrualMain.py"


## Exit the Singularity session
exit
