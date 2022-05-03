# TODO Check out Conda usage
# TODO update to involve install dependencies / set up python environment,
# TODO Access and run Notebook in VS Code
FROM ubuntu:20.04

# TODO set workdir
RUN mkdir -p /home/node/app
WORKDIR /home/node/app

# Init
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y

# Set timezone
ENV TZ=America/New_York
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

# TODO install any other dependencies through apt-get
RUN apt-get update -y &&

# TODO install python dependencies
# RUN [script name - initialize-python-environment for notebook to run in].sh

# TODO add project files
COPY . .

# TODO Start python with CMD command
CMD [ "npm", "run", "dev" ]

# At this point should be able to access notebook from VS Code...

# TODO use the USER command?