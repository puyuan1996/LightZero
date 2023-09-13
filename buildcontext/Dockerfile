# Start from Ubuntu 20.04
FROM ubuntu:20.04

# Set the working directory
WORKDIR /app

# Install Python 3.8 and other dependencies
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip gcc g++ swig git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a symlink for Python
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# Update pip and setuptools
RUN python -m pip install --upgrade pip setuptools

# Clone and install the package in editable mode
RUN git clone https://github.com/opendilab/LightZero.git
RUN pip install -e ./LightZero

# The command to run when this image starts up
CMD ["python", "./LightZero/zoo/classic_control/cartpole/config/cartpole_muzero_config.py"]

# docker build cmd:
# docker build -t ubuntu-py38-lz .
