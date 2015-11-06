FROM cuda:latest

MAINTAINER Lucas Benedicic <benedicic@cscs.ch>

ENV DEBIAN_FRONTEND noninteractive

# include the sample source code and build it
ADD src /usr/local/cuda/samples/cudalucas
WORKDIR /usr/local/cuda/samples/cudalucas
RUN make

CMD ./run.sh
