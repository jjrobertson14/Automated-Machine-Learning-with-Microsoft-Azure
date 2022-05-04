# TODO update to involve install dependencies / set up python environment,
# TODO access and run Notebook in VS Code
# Build with this to use the host network for downloads docker build --network=host .
# TODO make sure this works to Bind Mount the data files dir
# Run with Volume with `docker run -v $(pwd)/data:/opt/workdir bash:latest /`

FROM ubuntu:20.04
USER root

# Set Environment Variables
ENV TZ=America/New_York

# Set workdir
RUN mkdir -p /opt/workdir
RUN mkdir -p /opt/workdir/app
WORKDIR /opt/workdir


# Update and Install dependencies with apt-get
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y &&\
    apt-get install libssl-dev \
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

# TODO: Use guide here to install pyenv https://github.com/pyenv/pyenv#installation
# Install python, python 3.7 is good for ML 2022/05/03
RUN sudo pyenv install 3.7.13
# git clone https://github.com/pyenv/pyenv.git ~/.pyenv
# pyenv install 3.7.13
# pyenv global 3.7.13
# ???? pyenv  version-origin ????
# TODO edit .bashrc
# TODO edit .bash_profile
# source .bash_profile

# Install dependencies with pip
# python -m pip install --upgrade pip
# pip install setuptools \
# pip install --upgrade setuptools

# pip install ruamel.yaml \
# azureml.train.automl \
# wheel \
# pybind11 \
# Cython \
# scipy \
# numpy \
# mlflow

# pip install azureml.core
# pip install azureml.automl.core
# pip install azureml.train.automl
# pip install azureml.train.widgets
# pip install azureml-mlflow
# pip install azureml-interpret

# pip install sklearn.datasets

# pip install raiwidgets
# TODO Update and Install any remaining dependencies with apt-get
# TODO Finish above TODOs and diff pip list at this point with the Docker container's to see what is missing and add the rest of the installations right above here




# TODO set environment variable to create app directory (at top)
# TODO cd to directory, and run Notebook with name
# Add project files
# git clone https://github.com/jjrobertson14/Automated-Machine-Learning-with-Microsoft-Azure.git
COPY PyCaretBenchmarkNotebook.ipynb /opt/workdir/app/
# ???? cd Automated-Machine-Learning-with-Microsoft-Azure ????

# Run chmod in case it is needed
RUN chmod u+x /opt/workdir/app/*

# TODO Start python kernel with CMD command?
CMD ["python"]
# CMD [ "npm", "run", "dev" ]

# TODO Start notebook running with CMD command?
# CMD [ "npm", "run", "dev" ]

# At this point should be able to access notebook from VS Code...




# MORE TODOS
# TODO create start script in the git repo to install docker and nvidia stuff to run...
# WARNING: UNTESTED
# Run in WSL2
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
# TEST IT
# sudo docker run --cpu all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
# docker run -it --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3-jupyter

