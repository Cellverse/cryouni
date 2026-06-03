FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /

# Install basic tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    apt-utils \
    git \
    rpm \
    rsync \
    tmux \
    vim \
    wget \
    libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

# NCCL RDMA Plugin For Distributed Training
RUN wget https://taco-1251783334.cos.ap-shanghai.myqcloud.com/nccl/nccl-rdma-sharp-plugins-1.3-1.x86_64.rpm && \
    rpm -ivh --nodeps --force nccl-rdma-sharp-plugins-1.3-1.x86_64.rpm && \
    rm -f nccl-rdma-sharp-plugins-1.3-1.x86_64.rpm

SHELL ["/bin/bash", "-c"]

# RAPIDS For GPU-accelerated analysis
RUN pip install --no-cache-dir \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.10.*" "dask-cudf-cu12==25.10.*" "cuml-cu12==25.10.*" \
    "cugraph-cu12==25.10.*" "nx-cugraph-cu12==25.10.*" "cuxfilter-cu12==25.10.*" \
    "cucim-cu12==25.10.*" "pylibraft-cu12==25.10.*" "raft-dask-cu12==25.10.*" \
    "cuvs-cu12==25.10.*"
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    deepspeed \
    fvcore \
    h5py \
    ipykernel \
    jupyterlab \
    lightning \
    matplotlib \
    mrcfile \
    "numpy<2.3.0" \
    omegaconf \
    pandas \
    pre-commit \
    rich \
    scikit-fmm \
    scikit-learn \
    scipy \
    seaborn \
    starfile \
    tensorboard \
    timm \
    typer \
    umap-learn

ENTRYPOINT ["/usr/bin/bash"]

##### Build from dockerfile #####
# docker build -f dockerfile -t <DOCKER_NAME> .

##### Build from existing image #####
# docker load -i <DOCKER.tar>
# docker run -v <LOCAL_PATH>:<DOCKER_PATH> --gpus all --shm-size="1g" -it <DOCKER_NAME> /bin/bash
# docker commit <CONTAINER_ID> <DOCKER_NAME>
# docker save <DOCKER_NAME> -o <DOCKER.tar>
