FROM cuda:latest

MAINTAINER Lucas Benedicic <benedicic@cscs.ch>

ENV DEBIAN_FRONTEND noninteractive

# install MPI library
RUN apt-get install -y libmpich-dev   \
		       mpich          \
                       openssh-server \
                    --no-install-recommends	&& \
    apt-get clean

# include the sample source code and build it
ADD src /usr/local/cuda/samples/g2g
WORKDIR /usr/local/cuda/samples/g2g
RUN ./build.sh

# SSH configuration
ADD ssh/id_rsa.pub /root/.ssh/authorized_keys
RUN chmod 700  $HOME/.ssh                       && \
    chmod 600  $HOME/.ssh/authorized_keys       && \
    mkdir      /var/run/sshd                    && \
    chmod 0755 /var/run/sshd

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
