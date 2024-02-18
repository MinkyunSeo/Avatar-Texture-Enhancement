ARG CUDA_VERSION=11.6.2
ARG OS_VERSION=20.04
ARG USER_ID=1007
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION
ARG USER_ID

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: A specific GPU architectures (Compute Compatibility == 8.6) is only included and supported here.
## Specify the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus
## (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=86
ARG TORCH_CUDA_ARCH_LIST="Ampere"

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Asia/Seoul
## Set langauge
ENV LC_ALL=C.UTF-8
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.8-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget \
    unzip \
    htop && \
    rm -rf /var/lib/apt/lists/*

# Create non root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u ${USER_ID} user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new uer and workdir.
USER ${USER_ID}
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install packages.
RUN python3.8 -m pip install --upgrade pip setuptools pathtools promise pybind11
# Install pytorch and submodules
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python -m pip install \
    torch==1.12.1+cu${CUDA_VER} \
    torchvision==0.13.1+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

# Install pytorch3d
RUN pip install fvcore iopath
RUN FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install required python packages
RUN pip install cycler==0.10.0
RUN pip install decorator==4.4.1
RUN pip install imageio==2.8.0
RUN pip install kiwisolver==1.1.0
RUN pip install matplotlib==3.1.3
RUN pip install networkx==2.4
RUN pip install numpy==1.21.1
RUN pip install opencv-python==4.2.0.32
RUN pip install pathlib==1.0.1
RUN pip install Pillow==9.0.0
RUN pip install PyOpenGL==3.1.5
RUN pip install pyparsing==2.4.6
RUN pip install python-dateutil==2.8.1
RUN pip install PyWavelets==1.1.1
RUN pip install scikit-image==0.16.2
RUN pip install scipy==1.4.1
RUN pip install Shapely==1.7.0
RUN pip install six==1.14.0
RUN pip install tqdm==4.43.0
RUN pip install trimesh==3.5.23
RUN pip install xxhash==1.4.3
RUN pip install tyro
RUN pip install pyciede2000

# Change working directory
WORKDIR /home/user

# Set dev env
## Install zsh, tmux, and libfuse2
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
      zsh \
      tmux \
      libfuse2

# Install oh-my-zsh, theme, and plugins
WORKDIR /home/user
ENV ZSH_CUSTOM=/home/user/.oh-my-zsh/custom
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN git clone https://github.com/spaceship-prompt/spaceship-prompt.git "$ZSH_CUSTOM/themes/spaceship-prompt" --depth=1 && \
    ln -s "$ZSH_CUSTOM/themes/spaceship-prompt/spaceship.zsh-theme" "$ZSH_CUSTOM/themes/spaceship.zsh-theme"
RUN git clone https://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting

# zsh as default entrypoint.
WORKDIR /home/user
CMD /bin/zsh -l