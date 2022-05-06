# Run in WSL2
##### Links for help: 
- Cuda processing on WSL2 and example containers to benchmark GPU usage: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- PyCaret: https://pycaret.gitbook.io/docs/get-started/tutorials
- Nvidia RAPIDS with PyCaret: https://developer.nvidia.com/blog/streamline-your-model-builds-with-pycaret-rapids-on-nvidia-gpus/

### 11.1.4. Known Limitations (of NVIDIA Container Toolkit on WSL2 requiring use of Docker-CE for Linux inside WSL 2)
The following features are not supported in this release:
Note that NVIDIA Container Toolkit has not yet been validated with Docker Desktop WSL 2 backend. Use Docker-CE for Linux instead inside your WSL 2 Linux distribution.
CUDA debugging or profiling tools are not supported in WSL 2. This capability will be added in a future release.

## Step 1: Install NVIDIA Driver for GPU Support
(Steps from https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
Download and install NVIDIA GeForce Game Ready or NVIDIA RTX Quadro Windows 11 display driver on your system with a compatible GeForce or NVIDIA RTX/Quadro card from
https://developer.nvidia.com/cuda/wsl

Note: This is the only driver you need to install. Do not install any Linux display driver in WSL.

## Step 2. Install WSL 2
1. Launch your preferred Windows Terminal / Command Prompt / Powershell and install WSL:
wsl.exe --install
2. Ensure you have the latest WSL kernel:
wsl.exe --update
3. Run WSL 2
wsl.exe

## Now set up Docker and install nvidia-docker and cuda / wsl-ubuntu-11-4 packages 
From home dir...

sudo apt-get update && sudo apt-get upgrade

curl https://get.docker.com | sh
sudo apt-get update && apt-get install -y nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
source .bash_profile

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb

sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
sudo apt-get update && apt-get -y install cuda

Now Test the installed packages by running these docker containers from Main Docker Image Repository
sudo docker run --cpu all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
docker run -it --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3-jupyter