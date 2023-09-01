# ReAct

A (summer) project to embed adaptive-computation mechanisms in attention based models as an inductive bias to help towards OOD extrapolation.

Writeup/Website: [Notion Website](https://dripfeedofideas.notion.site/dripfeedofideas/ReAct-bef052956a0d45f29fb5a5383e7d737d)

WandB link: [here](https://wandb.ai/neel/ReAct/groups/bADD_32/workspace?workspace=user-neel) 

Code: You're here 😉

## Vast.ai

To reproduce the training runs: Find the precise commit id you want to replicate from the `WandB` or simply use the defaults.

1. Navigate to the docker "temporary slot" and enter the `Docker` image creds:

- DockerHub location: `neel04/react_image:latest`
- version tag: `latest`

2. Paste this as the `onstart.sh` script. Make sure to fill in the appropriate VAST and Wandb.ai key:

```shell
export VAST_API_KEY = ...
export WANDB_API_KEY = ...
export TASK = ... # refers to the branches of repo. "main" is `bAdd`, then `reverse_string` and `prefix_sum`

cd /workspace/
rm -rf /fsx/

#VAST config to shutdown instance after the job is done
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
echo $VAST_API_KEY > ~/.vast_api_key

# Clone ReAct repository and move it to /fsx/awesome/DPT/
git clone -b $TASK https://neel04/ReAct.git /fsx/awesome/DPT/
# Create a directory for the outputs
mkdir -p /fsx/awesome/DPT/outputs

# Change directory to /fsx/awesome/DPT/
cd /fsx/awesome/DPT

# Set CUDA memory allocation configuration to max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run React training
sh ./dpt_exec.sh
# Stop the instance
cd /workspace; ./vast stop instance ${VAST_CONTAINERLABEL:2}
```

3. Rent a 3090 instance (24Gb+ VRAM) and relax as it caches the docker file and runs the training.

4. Enjoy


# Credits

Credits go to Avi Schwarzchild's and Arpit Bansal's (et al.) repository on which this code is built on. Check out their amazing work [here!](https://github.com/aks2203/deep-thinking)

Huge thanks to [Algovera.ai](https://app.algovera.ai/) for sponsoring this project 🚀!

## Important commands
Docker container command:

```docker
docker run -it --rm -v /workspaces/ReAct:/fsx/awesome/DPT -w /fsx/awesome/DPT neel04/react_image:latest
# ... git clone and stuff
cd /fsx/awesome/DPT; sh ./dpt_exec.sh
```

Runs the training script by executing `DeepThinking.ipynb`, which in turn modifies some files and configs, finally executing the main trigger program.

(This rigmarole was due to this codebase originally working with SLURM on an HPC and then I never cleaned it up... But I suppose that's a story for another time.)