#!/bin/bash

docker run --rm                                     \
           -it                                      \
           -P                                       \
           --privileged                             \
           cudagpu:latest
