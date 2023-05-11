FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install git, wget, and gcc
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git wget && \
    apt-get clean && rm -rf /var/cache/apt/archives /var/lib/apt/lists

# Older versions of the bedtools contain bug in the bedtools merge command
RUN wget "https://github.com/arq5x/bedtools2/releases/download/v2.30.0/bedtools.static.binary" && \
    chmod a+x bedtools.static.binary && \
    mv bedtools.static.binary /usr/bin/bedtools

RUN pip install \
    jupyterlab==3.6.3 \
    tensorboard==2.12.2 \
    pytorch-ignite==0.4.12

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888
