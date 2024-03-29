#!/bin/bash
#SBATCH --job-name DPT_experiment   ## name that will show up in the queue
#SBATCH --exclusive
#SBATCH --output /fsx/awesome/dpt_log-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --partition=g40
#SBATCH -t 5-24:00:00  ## time limit: (D-HH:MM)
#SBATCH --account laion
#SBATCH --ntasks-per-node=1
#SBATCH --requeue ## requeue the job if preempted
#SBATCH --priority=normal ## set priority to TOP
#------------------------------------------------------
#SBATCH --container-image=neel04/react_image:latest 
#SBATCH --container-mounts=/fsx/awesome:/fsx/awesome
#SBATCH --container-workdir=/fsx/awesome/DPT
#SBATCH --container-writable
#SBATCH --container-name=react_image_v1

# Training on a slurm cluster
# Personal WANDB API key
export MASTER_PORT=29450
# obtain the IP address of the master node
export MASTER_ADDR=$(hostname -i)

export PYTHONBREAKPOINT="web_pdb.set_trace"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=2

cd /fsx/awesome
echo "Starting training" 
ipython /fsx/awesome/DPT/DeepThinking.ipynb
echo "All Commands executed"
exit 0