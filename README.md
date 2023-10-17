# capstone

## repo for dsga capstone

## steps to reproduce

1. Log in: Greene’s login node.
	- ssh greene
2. Log in to Burst node
	- ssh burst
3. Request a job / computational resource and wait until Slurm grants it.
    - srun --account="" --parition=interactive --gres=gpu --time=2:00:00 --pty /bin/bash
4. Singularity container (needs to be setup)
    - singularity exec --bind /scratch --nv --overlay /scratch/cg3278/overlay-25GB-500K.ext3:rw /scratch/cg3278/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash
5. Activate conda environment with your own deep learning libraries.
    - conda activate ''
6. Run your code, make changes/debugging.
    - sh test.sh
     >> bash scripts/ours_v2_bs4.sh

## test.sh
Contains the script to run ours_vs_bs4.sh (inside of gradient-inversion-geenrative-image-prior)

