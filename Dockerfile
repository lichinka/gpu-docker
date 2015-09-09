FROM ubuntu:14.04

MAINTAINER Lucas Benedicic <benedicic@cscs.ch>

# environment variables
ENV DEBIAN_FRONTEND noninteractive

# install software to enable CUDA inside the container
RUN apt-get update                              && \
    apt-get install -y make         \
                       gcc-4.8      \
                       g++-4.8      \
                       gfortran-4.8 \
                       wget         \
                    --no-install-recommends     && \
    rm -rf /var/lib/apt/lists

# user and locale configuration
RUN useradd dev                                                     && \
    echo "ALL            ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    cp /usr/share/zoneinfo/Europe/Zurich /etc/localtime             && \
    dpkg-reconfigure locales                                        && \
    locale-gen en_US.UTF-8                                          && \
    /usr/sbin/update-locale LANG=en_US.UTF-8

WORKDIR /home/dev
ENV     HOME /home/dev
ENV     LC_ALL en_US.UTF-8

## install NVIDIA driver 352.30
ENV CUDATOOLKIT_HOME /usr/local/cuda-7.0
RUN ln -s /usr/bin/gcc-4.8 /usr/bin/gcc   && \
    ln -s /usr/bin/g++-4.8 /usr/bin/g++
RUN wget --no-verbose -O NVIDIA-Linux.run http://us.download.nvidia.com/XFree86/Linux-x86_64/352.30/NVIDIA-Linux-x86_64-352.30.run && \
    chmod +x NVIDIA-Linux.run                           && \
    ./NVIDIA-Linux.run --accept-license           \
                       --no-kernel-module         \
                       --no-kernel-module-source  \
                       --no-install-vdpau-wrapper \
                       --no-unified-memory        \
                       --silent                         && \
    rm NVIDIA-Linux.run

# install CUDA 7
RUN wget --no-verbose -O cuda_7.0.run http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run && \
    chmod +x cuda_7.0.run                                       && \
    ./cuda_7.0.run --samples           \
                   --samplespath=$HOME \
                   --silent            \
                   --toolkit           \
                   --toolkitpath=$CUDATOOLKIT_HOME   && \
    rm cuda_7.0.run /tmp/cuda?install*

# change ownership to non-root user
RUN chown --recursive dev:dev $HOME
USER dev

CMD ["bash"]
