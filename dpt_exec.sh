#!/bin/bash
#SBATCH --job-name DPT_experiment   ## name that will show up in the queue
#SBATCH --exclusive
#SBATCH --output /fsx/awesome/dpt_log-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --partition=g40423
#SBATCH -t 5-24:00:00  ## time limit: (D-HH:MM)
#SBATCH --account laion
#SBATCH --ntasks-per-node=1
#SBATCH --requeue ## requeue the job if preempted

# Training on a slurm cluster
# Personal WANDB API key
export WANDB_API_KEY=618e11c734b0f6069af4735cde3d3d515930d678
# Stability.ai API key
export WANDB_API_KEY=local-6cd1ebf260e154dcd6af9d7ccac6230f4f52e9e6

export PYTHONBREAKPOINT="web_pdb.set_trace"

cd /fsx/awesome
echo "Starting training" 
singularity exec --nv -B /fsx/awesome:/fsx/awesome singularity_base_container.sif ipython /fsx/awesome/DPT/DeepThinking.ipynb
echo "All Singularity Commands executed"
# cp /fsx/awesome/dpt_log-${SLURM_JOB_ID}.out /fsx/awesome/DPT_MAIN-${SLURM_JOB_ID}.out
# Explicitly exit the script
exit 0