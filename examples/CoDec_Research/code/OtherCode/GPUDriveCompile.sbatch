#!/bin/bash
#SBATCH --job-name=GPUDriveCompilation
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=GPUDriveCompile.out


cd /scratch/sb10438/gpudrive-CoDec/
mkdir build
cd build
singularity exec --nv --overlay /scratch/sb10438/images/gpudrive/overlay-50G-10M.ext3:rw \
        /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
        /bin/bash -c "source /ext3/env.sh; cmake .. -DCMAKE_BUILD_TYPE=Release; make; cd ..; pip install -e .; exit"


