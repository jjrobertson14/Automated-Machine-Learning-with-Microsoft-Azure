# TODO update to involve install dependencies / set up python environment,
# TODO access and run Notebook in VS Code
# Build with this to use the host network for downloads docker build --network=host .
# TODO make sure this works to Bind Mount the data files dir
# Run with Volume with `docker run -v $(pwd)/data:/opt/workdir/app/Automated-Machine-Learning-with-Microsoft-Azure/ bash:latest /`

FROM ubuntu:20.04
USER root

# Set Environment Variables
ENV TZ=America/New_York

# Set workdir
ENV WORKDIR=/opt/workdir
RUN mkdir -p $WORKDIR
RUN mkdir -p $WORKDIR/app
WORKDIR $WORKDIR


# Update and Install dependencies with apt-get
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y &&\
    apt-get install -y git \
                       pip \
                       libssl-dev \
                       zlib1g-dev \
                       libbz2-dev \
                       libreadline-dev \
                       libsqlite3-dev \
                       llvm \
                       libncurses5-dev \
                       libncursesw5-dev \
                       xz-utils \
                       tk-dev \
                       libgdbm-dev \
                       lzma \
                       lzma-dev \
                       tcl-dev \
                       libxml2-dev \
                       libxmlsec1-dev \
                       libffi-dev \
                       liblzma-dev \
                       wget \
                       curl \
                       make \
                       build-essential \
                       python-openssl \
                       cmake \
                       libblas-dev \
                       liblapack-dev \
                       libatlas-base-dev \
                       gfortran

# (NOTE: Use guide here to install pyenv https://github.com/pyenv/pyenv#installation)
# (Install python, python 3.7 is good for ML 2022/05/03)
# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv &&\
    # Edit .bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc &&\
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc &&\
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc &&\
    # Edit .bash_profile
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile &&\
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile &&\
    echo 'eval "$(pyenv init -)"' >> ~/.bash_profile &&\
    # Source bash_profile
    . ~/.bash_profile &&\
    pyenv install 3.7.13 &&\
    pyenv global 3.7.13

# Install dependencies with pip
RUN python3 -m pip install --upgrade pip &&\
    pip install setuptools  &&\
    pip install --upgrade setuptools &&\
    pip install ruamel.yaml \
                wheel \
                pybind11 \
                Cython \
                scipy \
                numpy \
                mlflow

# AzureML packages
RUN pip install azureml.core \
            azureml.automl.core \
            azureml.train.automl \
            azureml.train.widgets \
            azureml-mlflow \
            azureml-interpret

# SciKit Learn packages
RUN pip install sklearn.datasets 

CMD pip list

# TODO Diff pip list at this point with the Docker container's to see what is missing and add the rest of the installations right above here


# TODO set environment variable to create app directory (at top) (remember to replace all instances of Automated-Machine-Learning-with-Microsoft-Azure with it)
# Add project files
# git clone https://github.com/jjrobertson14/Automated-Machine-Learning-with-Microsoft-Azure.git
# COPY PyCaretBenchmarkNotebook.ipynb /opt/workdir/app/
# TODO cd to directory, and run Notebook with name
# ???? cd Automated-Machine-Learning-with-Microsoft-Azure ????

# Run chmod in case it is needed
# RUN chmod u+x /opt/workdir/app/*

# TODO Start python kernel with CMD command?
# CMD ["python"]
# CMD [ "npm", "run", "dev" ]
# TODO Start notebook running with CMD command?
# CMD [ "npm", "run", "dev" ]

# TODO If have to install more stuff, update and Install any remaining dependencies with apt-get, remember ~/.bash_history
# At this point should be able to access notebook from VS Code...




# MORE TODOS
# TODO Move installations into separate Dockerfiles that inherit from each other?
# TODO create tutorial (and/or start script) in the git repo to install docker and nvidia stuff to run...
# WARNING: UNTESTED
# Run in WSL2
#
# Links for help: 
# - Cuda processing on WSL2 and example containers to benchmark GPU usage: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
# - PyCaret: https://pycaret.gitbook.io/docs/get-started/tutorials
# - Nvidia RAPIDS with PyCaret: https://developer.nvidia.com/blog/streamline-your-model-builds-with-pycaret-rapids-on-nvidia-gpus/
#
# ### 11.1.4. Known Limitations (of NVIDIA Container Toolkit on WSL2 requiring use of Docker-CE for Linux inside WSL 2)
# The following features are not supported in this release:
# Note that NVIDIA Container Toolkit has not yet been validated with Docker Desktop WSL 2 backend. Use Docker-CE for Linux instead inside your WSL 2 Linux distribution.
# CUDA debugging or profiling tools are not supported in WSL 2. This capability will be added in a future release.

# (Steps from https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
# Step 1: Install NVIDIA Driver for GPU Support
# Download and install NVIDIA GeForce Game Ready or NVIDIA RTX Quadro Windows 11 display driver on your system with a compatible GeForce or NVIDIA RTX/Quadro card from
# https://developer.nvidia.com/cuda/wsl
# 
# Note: This is the only driver you need to install. Do not install any Linux display driver in WSL.
# 
#
# Step 2. Install WSL 2
# 1. Launch your preferred Windows Terminal / Command Prompt / Powershell and install WSL:
# wsl.exe --install
# 2. Ensure you have the latest WSL kernel:
# wsl.exe --update
# 3. Run WSL 2
# wsl.exe
#
# Now set up Docker and install nvidia-docker and cuda / wsl-ubuntu-11-4 packages 
# From home dir...
# 
# sudo apt-get update && sudo apt-get upgrade
# 
# curl https://get.docker.com | sh
# sudo apt-get update && apt-get install -y nvidia-docker2
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# source .bash_profile
# 
# wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
# sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
# sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
#
# sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
# sudo apt-get update && apt-get -y install cuda
# 
# Now Test the installed packages by running these docker containers from Main Docker Image Repository
# sudo docker run --cpu all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
# docker run -it --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3-jupyter