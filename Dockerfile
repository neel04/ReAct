FROM nvcr.io/nvidia/pytorch:23.05-py3

# Create a directory in the image
RUN mkdir -p /fsx/awesome/DPT
# Set the working directory
WORKDIR /fsx/awesome/DPT
# Copy the entire ReAct directory into the image
COPY . .

RUN apt-get -y update
RUN apt-get -y install nano vim libfreetype6-dev git gcc build-essential
RUN pip3 uninstall pil
RUN pip3 install opencv-python-headless Ipython matplotlib wandb
RUN pip3 install -U torchmetrics Pillow web-pdb accelerate rotary_embedding_torch
RUN pip3 install einops hydra-core omegaconf
RUN pip3 install torchvision tqdm nbformat easy-to-hard-data jupyterlab
RUN pip3 install icecream matplotlib numpy pandas scipy

# create jupyter_notebook_config.py
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.contents_manager_class = 'notebook.services.contents.largefilemanager.LargeFileManager'" > /root/.jupyter/jupyter_notebook_config.py