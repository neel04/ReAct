FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN apt-get -y update
RUN apt-get -y install nano vim libfreetype6-dev git gcc build-essential
RUN pip3 uninstall pil
RUN pip3 install s3fs opencv-python-headless scipy tensorflow_datasets Ipython matplotlib wandb
RUN pip3 install -U torchmetrics Pillow Image web-pdb accelerate rotary_embedding_torch
RUN pip3 install einops hydra-core omegaconf tensorboard tensorboard-plugin-wit
RUN pip3 install torchvision tqdm comet-ml nbformat easy-to-hard-data jupyterlab
RUN pip3 install icecream matplotlib numpy pandas Pillow scipy seaborn svglib