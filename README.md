# ReAct


## Important commands
Docker container:

```docker
docker run -it --rm -v /workspaces/ReAct:/fsx/awesome/DPT -w /fsx/awesome/DPT neel04/react_image:latest
cd /fsx/awesome/DPT; sh ./dpt_exec.sh
```

Runs the training script by executing `DeepThinking.ipynb`, which in turn modifies some files and configs, finally executing the main trigger program