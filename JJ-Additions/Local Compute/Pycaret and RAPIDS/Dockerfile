# TODO make sure this works to Bind Mount the data files dir
# Run with Volume with `docker run -v $(pwd)/data:/opt/workdir bash:latest /`


# TODO update to involve install dependencies / set up python environment,
# TODO Access and run Notebook in VS Code
FROM ubuntu:20.04
USER root

# Set workdir
RUN mkdir -p /opt/workdir
RUN mkdir -p /opt/workdir/app
WORKDIR /opt/workdir

# Init
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y

# Set timezone
ENV TZ=America/New_York
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

# TODO Install any other dependencies through apt-get
# RUN apt-get update -y &&

# TODO Install python dependencies
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