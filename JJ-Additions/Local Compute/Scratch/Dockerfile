FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20220314.v1

ARG DEBIAN_FRONTEND=noninteractive
ARG ACCEPT_EULA=Y
RUN ls -altr
RUN apt-get update -y && apt-get upgrade -y &&\
    apt-get install -y build-essential \
                       cmake \
                       curl \
                       gfortran \
                       git \
                       jupyter \
                       libatlas-base-dev \
                       libblas-dev \
                       libbz2-dev \
                       libffi-dev \
                       libgdbm-dev \
                       liblapack-dev \
                       liblzma-dev \
                       libncurses5-dev \
                       libncursesw5-dev \
                       libreadline-dev \
                       libsqlite3-dev \
                       libssl-dev \
                       libxml2-dev \
                       libxmlsec1-dev \
                       llvm \
                       lzma \
                       lzma-dev \
                       make \
                       tcl-dev \
                       tk-dev \
                       wget \
                       xz-utils \
                       zlib1g-dev
RUN conda -V
# RUN pip install azureml-train-automl 
RUN pip install azureml.interpret azureml-dataset-runtime azureml.train azureml-train-automl jinja2 MarkupSafe raiwidgets python-dotenv pybridge 
COPY './conda_dependencies.scratch.yml' .
# RUN conda env create -f './conda_dependencies.yml'
RUN conda update -n base -c defaults conda && export PIP_LOG="/tmp/pip_log.txt" && touch ${PIP_LOG} && tail -f ${PIP_LOG} & conda env create -f "./conda_dependencies.scratch.yml" && killall tail && rm ${PIP_LOG}

RUN echo "Hello from custom container!" > ~/hello.txt

# d TODO try installing this stuff too, I really need to see an error message...
# d TODO running the conda env create (RUN conda env create -f './conda_dependencies.yml')
# d TODO try installing ONLY 
# can try removing entries from pip_packages in the Notebook
#       TODO! pin the installed package versions once this works

# TODO if can't get anything to work, drop Docker, try to use the conda environment local run instead of Docker.

# conda_packages = ['pip',
#                   'pyspark',
#                   'scikit-learn'
#                  ]

# pip_packages =   ['azureml.interpret',
#                   'azureml-dataset-runtime',
#                   'azureml.train',
#                   'azureml-train-automl',
#                   'jinja2',
#                   'MarkupSafe',
#                   'raiwidgets',
#                   'python-dotenv',
#                   'pybridge'
#                  ]

# d TODO try to get more logs from the pip or conda processes? (also I wonder why logs are not appearing in Notebook)
# RUN export PIP_LOG="/tmp/pip_log.txt" && touch ${PIP_LOG} && tail -f ${PIP_LOG} & conda env create --prefix ${CONDA_ENV_PATH} -f "${TMP_ENV_DIR}/train/conda_dependencies.yml" && killall tail && rm ${PIP_LOG}
#
# that is one liner of \/
#
# export PIP_LOG="/tmp/pip_log.txt"
# touch ${PIP_LOG}
# tail -f ${PIP_LOG} & 
# conda env create --prefix ${CONDA_ENV_PATH} -f "${TMP_ENV_DIR}/train/conda_dependencies.yml"
# killall tail
# rm ${PIP_LOG}
