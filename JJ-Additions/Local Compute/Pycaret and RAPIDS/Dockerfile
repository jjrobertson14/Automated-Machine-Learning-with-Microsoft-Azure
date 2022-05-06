# Build with this to use the host network for downloads docker build --network=host --tag=pycaret-rapids:latest .
# TODO make sure this works with the Bind Mount of the data files dir
# Run with Volume with `docker run --rm --gpus all --network=host --mount type=bind,src="$(pwd)/data",target=/opt/workdir/data pycaret-rapids:latest`

FROM ubuntu:20.04
USER root

# On ARG and ENV, in case you want to add:
#   You can use ENV instructions in a Dockerfile to define variable values. These values persist in the built image. 
#   However, often persistence is not what you want. Users want to specify variables differently depending on which host they build an image on.
#   The ARG instruction lets Dockerfile authors define values that users can set at build-time using the --build-arg flag:
#       docker build --build-arg HTTP_PROXY=http://10.20.30.2:1234 --build-arg FTP_PROXY=http://40.50.60.5:4567 .
#   This flag allows you to pass the build-time variables that are accessed like regular environment variables in the RUN instruction of the Dockerfile. Also, these values don’t persist in the intermediate or final images like ENV values do. You must add --build-arg for each build argument.

# Set Environment Variables
ENV TZ=America/New_York

# Set workdir
ENV BASEDIR=/opt/workdir
RUN mkdir -p $BASEDIR
WORKDIR $BASEDIR


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
                       gfortran \
                       jupyter

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

# Upgrade pip and install packages with it
RUN cat ~/.bash_profile && . ~/.bash_profile && pyenv versions && which python && which pip &&\
    python -m pip install --upgrade pip &&\
    python -m pip install wheel \
                # Install SciKit Learn and SciPy packages with pip
                scikit-learn \
                sklearn-pandas \
                scikit-plot \
                scipy \
                # Install PyCaret with pip
                pycaret \
                # Install remaining dependencies with pip
                setuptools \
                ruamel.yaml \
                pybind11 \
                Cython \
                scipy \
                numpy \
                mlflow \
                ez_setup &&\
                # Upgrade dependencies
                # pip install --upgrade scikit-learn setuptools &&\
    python -m pip freeze > requirements.txt

# NOTE: I have previously seen pip install errors:
    # Go here for all the help you should need for handling Python dependency errors, it's short.
    #   (https://pip.pypa.io/en/stable/topics/dependency-resolution/#dealing-with-dependency-conflicts)
    # There are no catch-all solutions except using Conda or Python Poetry.
    # 
    # (like "packageX version has requirement packageY>=version, but you'll have packageY version which is incompatible.)
    # 
    # (RUN 1:)
    # yellowbrick 1.4 has requirement scikit-learn>=1.0.0, but you'll have scikit-learn 0.23.2 which is incompatible.
    # numba 0.54.1 has requirement numpy<1.21,>=1.17, but you'll have numpy 1.22.3 which is incompatible.
    #
    # (RUN X:)
    # Fixed by moving package installations around and removing unneccessary installs (unused AzureML ones mainly). For the fussiest big packages, put install into one RUN command, and install the packages it whines aboute before the RUN command

# Add project files
COPY PyCaretBenchmarkNotebook.ipynb .

# Start notebook webserver at localhost with CMD command
# CMD [ "jupyter", "notebook", "PyCaretBenchmarkNotebook.ipynb" ]
ENTRYPOINT [ "jupyter", "notebook", "--ip='*'", "--port=8888", "--allow-root", "PyCaretBenchmarkNotebook.ipynb" ]

# TODO If have to install more stuff, update and Install any remaining dependencies with apt-get, remember ~/.bash_history
# At this point should be able to access notebook from VS Code...


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