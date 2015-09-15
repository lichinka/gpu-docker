#
# ~/.bashrc
#

alias ll='ls -lah --color=auto'
PS1='[\u@\h \W]\$ '

export PATH=${PATH}:${CUDATOOLKIT_HOME}/bin
export LD_LIBRARY_PATH=${CUDATOOLKIT_HOME}/lib64
