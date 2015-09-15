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

### setup CUDA 7 environment
ENV CUDATOOLKIT_HOME /opt/cuda
RUN ln -s /usr/bin/gcc-4.8 /usr/bin/gcc   && \
    ln -s /usr/bin/g++-4.8 /usr/bin/g++
RUN wget --no-verbose -O cuda_7.0.run http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run && \
    chmod +x cuda_7.0.run                               && \
    ./cuda_7.0.run --samples           \
                   --samplespath=$HOME \
                   --silent            \
                   --toolkit           \
                   --toolkitpath=$CUDATOOLKIT_HOME      && \
    rm cuda_7.0.run /tmp/cuda?install*

# copy .bashrc
ADD bashrc .bashrc

# change ownership to non-root user
RUN chown --recursive dev:dev $HOME
USER dev

# compile an example
RUN cd ${HOME}/NVIDIA_CUDA-7.0_Samples                  && \
    cd 1_Utilities/bandwidthTest                        && \
    make clean all

CMD ["bash"]
