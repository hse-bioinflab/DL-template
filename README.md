# Introduction

This repository serves as a comprehensive template for conducting deep learning (DL) studies at the HSE bioinformatics lab. It includes an all-in-one notebook to train, validate, and test a DL model, and then use it to generate whole-genome predictions.

With this framework, we aim to simplify the process of DL-based research and provide a standardized approach for conducting experiments in our lab.

# Setup

1. Establish an SSH connection with port forwarding using the following command:

```bash
# Forwarding local port 8888 to remote port 8888
ssh -L localhost:8888:localhost:8888 -p <server-port> <username>@<server-address>
```

Replace <server-address> with the address of the server you want to connect to, and <server-port> with the port number.

3. Clone the repository to the server:

```bash
git clone git@github.com:hse-bioinflab/DL-template.git
```

3. Build the Docker container with all the necessary dependencies:

```bash
cd DL-templates/docker
docker build -t dl-template:1.0 .
cd ..
```

4. Launch the JupyterLab frontend:

```bash
docker run -it --rm --runtime=nvidia --gpus all \
    --shm-size=4GB -p 8888:8888 \
    -v $(pwd):/workspace/ \
    dl-template:1.0
```

Check your console for a URL starting with http://127.0.0.1:8888/lab?token=. Copy the URL and open it in your browser.

Remember to keep the SSH session running while you work.

# Notes

* To use GPUs inside the Docker container, one must install the [NVIDIA Docker runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
* Make sure that your server user has rights to use Docker.
