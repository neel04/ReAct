# ReAct

A (summer) project to embed adaptive-computation mechanisms in attention based models as an inductive bias to help towards OOD extrapolation.

Writeup/Website: [Notion Website](https://dripfeedofideas.notion.site/dripfeedofideas/ReAct-bef052956a0d45f29fb5a5383e7d737d)

WandB link: [here](https://wandb.ai/neel/ReAct/groups/bADD_32/workspace?workspace=user-neel) 

Code: You're here 😉

# Vast.ai

To reproduce the training runs: Find the precise commit id you want to replicate from the `WandB` or simply use the defaults.

1. Navigate to the docker "temporary slot" and enter the `Docker` image creds:

- DockerHub location: `neel04/react_image:latest`
- version tag: `latest`

2. Paste this as the `onstart.sh` script. Make sure to fill in the appropriate VAST and Wandb.ai key.

For commit, you can use the `default` or use the commit id from the WandB runs. Here are some I'd recommend:

- `bAdd`: `ac3b5a4bf328e01ea1f37ccdbe0cd1053f0abe53`
- `reverse_string` : `07af7653514a976d3e0355d544f32c1693a563c6`
- `prefix_sum` : `e006c2e859bfab3106c110f76a065fbe3a89fb45`

```shell
export VAST_API_KEY=...
export WANDB_API_KEY=...

export TASK="main"  # refers to the branches of repo. "main" is `bAdd`, then `reverse_string` and `prefix_sum`
export COMMIT_ID="default"  # "default" will not checkout any commit

cd /workspace/
rm -rf /fsx/

# VAST config to shutdown instance after the job is done
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
echo $VAST_API_KEY > ~/.vast_api_key

# Clone ReAct repository and move it to /fsx/awesome/DPT/
git clone -b $TASK https://github.com/neel04/ReAct.git /fsx/awesome/DPT/
cd /fsx/awesome/DPT

# If commit id is not default, checkout that commit
if [ "$COMMIT_ID" != "default" ]; then
  git checkout $COMMIT_ID
fi

# Create a directory for the outputs
mkdir -p /fsx/awesome/DPT/outputs

# Change directory to /fsx/awesome/DPT/
cd /fsx/awesome/DPT

# Set CUDA memory allocation configuration to max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run React training
chmod +x ./dpt_exec.sh
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

## Made with 🩸, 💧 & 😢

Email: [neelgupta04@outlook.com](mailto:neelgupta04@outlook.com)

Github: [neel04](https://github.com/neel04/ReAct) (links to the code for this project)

Discord: `awesome_ruler_007` - or you can usually find me on [Yannic's](https://ykilcher.com/discord) server or "[Learn AI Together](https://discord.gg/ARjZvPnt)”