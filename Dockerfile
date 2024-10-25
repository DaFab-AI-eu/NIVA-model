# https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-03.html#rel-23-03
FROM nvcr.io/nvidia/tensorflow:23.03-tf2-py3

# Set environment variables for easier reference
ENV NIVA_PATH=/home/niva \
    SOFTWARE_PATH=$NIVA_PATH/software \
    LIBGEOS_PATH=$SOFTWARE_PATH/libgeos 

# Create necessary directories
RUN mkdir -p $NIVA_PATH $NIVA_PATH/ai4boundaries_data

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    zlib1g-dev \
    libbz2-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    curl \
    libncursesw5-dev \
    libgdbm-dev \
    liblzma-dev \
    tk-dev \
    xz-utils \
    libgeos++-dev libgeos-dev libgeos-doc \
    && rm -rf /var/lib/apt/lists/*

# LibGeos installation
# apt-get install libgeos-dev

# RUN git clone https://github.com/libgeos/geos.git $LIBGEOS_PATH && \
#     rm -rf $LIBGEOS_PATH/build $LIBGEOS_PATH/install && \
#     mkdir -p $LIBGEOS_PATH/build $LIBGEOS_PATH/install
# RUN cd $LIBGEOS_PATH/build && cmake -DCMAKE_INSTALL_PREFIX=$LIBGEOS_PATH/install .. && \
#     make -j && \
#     make install -j

# ENV PATH=$LIBGEOS_PATH/install/bin:$PATH \
#     LD_LIBRARY_PATH=$LIBGEOS_PATH/install/lib:$LD_LIBRARY_PATH

# Clone Niva_Project's docker_env_creation branch
RUN cd $NIVA_PATH && \
    git clone -b docker_env_creation https://github.com/DaFab-AI-eu/NIVA-model.git niva_repo

# Install env from requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r $NIVA_PATH/niva_repo/requirements_training.txt

# Entry point
WORKDIR $NIVA_PATH/niva_repo
CMD ["/bin/bash"]
