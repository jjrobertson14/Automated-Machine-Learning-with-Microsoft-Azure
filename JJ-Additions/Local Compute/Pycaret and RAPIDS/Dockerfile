# Build with this to use the host network for downloads docker build --network=host .

# TODO make sure this works to Bind Mount the data files dir
# Run with Volume with `docker run -v $(pwd)/data:/opt/workdir bash:latest /`


# TODO update to involve install dependencies / set up python environment,
# TODO Access and run Notebook in VS Code
FROM ubuntu:20.04
USER root

# Set timezone
ENV TZ=America/New_York

# Set workdir
RUN mkdir -p /opt/workdir
RUN mkdir -p /opt/workdir/app
WORKDIR /opt/workdir

# TODO determine and lookup in previous notes dependencies necessary for the notebook to run in WSL2
# TODO Update and Install dependencies with apt-get
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y &&\
    apt-get install pyenv \
    pip

# TODO: Use guide here to install pyenv https://github.com/pyenv/pyenv#installation
# Install python, python 3.7 is good for ML 2022/05/03
RUN sudo pyenv install 3.7.13

# TODO check what is installed in pip
# TODO Install python dependencies with pip
# RUN [script name - initialize-python-environment for notebook to run in].sh

# Add project files
COPY PyCaretBenchmarkNotebook.ipynb /opt/workdir/app/

# Run chmod in case it is needed
RUN chmod u+x /opt/workdir/app/*

CMD ["python"]

# TODO Start python kernel with CMD command?
# CMD [ "npm", "run", "dev" ]

# TODO Start notebook running with CMD command?
# CMD [ "npm", "run", "dev" ]

# At this point should be able to access notebook from VS Code...