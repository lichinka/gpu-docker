## On GPU virtualization using Hypervisors and Linux containers

###### NOTE: the reader might refer to [this white paper](http://sp.parallels.com/fileadmin/media/hcap/pcs/documents/ParCloudStorage_Mini_WP_EN_042014.pdf) for a short introduction/comparison about hypervisors and linux containers.

### Introduction

Infrastructure-as-a-Service (IaaS) clouds have the potential to enable access to
the latest and most powerful computational accelerators. Yet today’s clouds are 
typically homogeneous without access to even the most commonly used accelerators.
Given the diversity surrounding the choice of GPUs, host systems, and hypervisors,
it is perhaps no surprise that Amazon is the only major cloud provider offering 
customers access to GPU-enabled instances [1].
Recently, however, this is starting to change as open source and other freely 
available hypervisors now provide sufficiently robust PCI passthrough functionality
that enable GPU and other accelerator direct access to the hardware resources.
Moreover, as major vendors include improved hardware support for virtualization, e.g., 
Intel's VT-x and AMD's AMD-V, PCI passthrough is available with a minimal overhead [1]
compared to previous microarchitecture generations [2].

Today it is possible to access GPUs at high performance within all of the major 
hypervisors, merging many of the advantages of cloud computing (e.g. custom 
images, software defined networking, etc.) with the accessibility of on-demand
accelerator hardware. 

### Performance

In [1], a characterization of the performance of both NVIDIA Fermi and Kepler 
GPUs operating in PCI passthrough mode using different hypervisors and Linux
containers is presented. This work confirms an overhead of around 1% when
accessing a Kepler GPU from a Sandy Bridge socket running a KVM hypervisor
or LXC container.
These results contrast with a previous work testing, among others, Linpack 
performance under KVM from a Nehalem socket with almost 30% overhead [2].

Some of Intel's current research efforts are also oriented to improve and
ease the performance and accessibility of GPU virtualization using the *i915*
kernel module (see [Intel Graphics Virtualization Technology](https://01.org/igvt-g)).
In this context, work presented in [3] shows an overhead of around 5% using 
a "mediated pass-through" technology known as *gVirt*.

Nvidia also provides dedicated GPU access for virtual desktops (see [Virtual
GPUs](http://www.nvidia.com/object/virtual-gpus.html)). However, this 
technology is currently available for just Citrix Xen and VmWare Horizon 
hypervisors, which are commercial solutions. Tests presented in [1] show that
both Xen and VMWare can achieve 96-99% of the base systems performance, respectively.


### Hands-on testing: single node

### Building a Docker container for accessing a single GPU

The host system setup included a Linux x86_64 with kernel 4.1.6, Nvidia driver 352.30 and the development toolkit for CUDA 7.0 on the host. All included CUDA examples were compiled and correctly executed on the host side.

The container setup featured an Ubuntu 14.04 base image. After launching the container, we made sure the device characters `/dev/nvidia0`, `/dev/nvidiactl` and `/dev/nvidia-uvm` were accessible from inside the container by providing the `--privileged` flag to Docker 1.8.1.

The following sections contain the steps taken to compile and execute the `bandwidthTest` bundled sample from inside the container. The GCC compiler family (version 4.8.4) was used.

#### 1. test: driver 352.30 on the host, nothing on the container
```
1_Utilities/bandwidthTest$ ../../bin/x86_64/linux/release/bandwidthTest 
[CUDA Bandwidth Test] - Starting...
Running on...

cudaGetDeviceProperties returned 35
-> CUDA driver version is insufficient for CUDA runtime version
CUDA error at bandwidthTest.cu:255 code=35(cudaErrorInsufficientDriver) "cudaSetDevice(currentDevice)" 
```

#### 2. test: nothing on the host, driver 352.30 on the container
```
dev@b76def5159bf:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ make
"/usr/local/cuda-7.0"/bin/nvcc -ccbin g++ -I../../common/inc  -m64    -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bandwidthTest.o -c bandwidthTest.cu
"/usr/local/cuda-7.0"/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bandwidthTest bandwidthTest.o 
mkdir -p ../../bin/x86_64/linux/release
cp bandwidthTest ../../bin/x86_64/linux/release
dev@b76def5159bf:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ ../../bin/x86_64/linux/release/bandwidthTest 
[CUDA Bandwidth Test] - Starting...
Running on...

modprobe: ERROR: ../libkmod/libkmod.c:556 kmod_search_moddep() could not open moddep file '/lib/modules/4.1.6-1-ARCH/modules.dep.bin'
cudaGetDeviceProperties returned 38
-> no CUDA-capable device is detected
CUDA error at bandwidthTest.cu:255 code=38(cudaErrorNoDevice) "cudaSetDevice(currentDevice)" 
dev@b76def5159bf:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ 
```

#### 3. test: driver 352.30 on the host, driver 352.30 on the container
```
dev@831eed66a674:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ make
"/usr/local/cuda-7.0"/bin/nvcc -ccbin g++ -I../../common/inc  -m64    -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bandwidthTest.o -c bandwidthTest.cu
"/usr/local/cuda-7.0"/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bandwidthTest bandwidthTest.o 
mkdir -p ../../bin/x86_64/linux/release
cp bandwidthTest ../../bin/x86_64/linux/release
dev@831eed66a674:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ ../../bin/x86_64/linux/release/bandwidthTest 
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Quadro K1100M
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)    Bandwidth(MB/s)
   33554432         9680.4

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)    Bandwidth(MB/s)
   33554432         9615.7

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)    Bandwidth(MB/s)
   33554432         31407.7

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
dev@831eed66a674:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$
```

#### 4. test: driver 352.30 on the host, driver 352.21 on the container
```
dev@30f6f0663b4e:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ make
"/usr/local/cuda-7.0"/bin/nvcc -ccbin g++ -I../../common/inc  -m64    -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bandwidthTest.o -c bandwidthTest.cu
"/usr/local/cuda-7.0"/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bandwidthTest bandwidthTest.o 
mkdir -p ../../bin/x86_64/linux/release
cp bandwidthTest ../../bin/x86_64/linux/release
dev@30f6f0663b4e:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ ../../bin/x86_64/linux/release/bandwidthTest 
[CUDA Bandwidth Test] - Starting...
Running on...

cudaGetDeviceProperties returned 38
-> no CUDA-capable device is detected
CUDA error at bandwidthTest.cu:255 code=38(cudaErrorNoDevice) "cudaSetDevice(currentDevice)" 
dev@30f6f0663b4e:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ 
```

#### 5. test: driver 352.30 on the host, driver 352.41 on the container
```
dev@b76def5159bf:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ make
"/usr/local/cuda-7.0"/bin/nvcc -ccbin g++ -I../../common/inc  -m64    -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bandwidthTest.o -c bandwidthTest.cu
"/usr/local/cuda-7.0"/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bandwidthTest bandwidthTest.o 
mkdir -p ../../bin/x86_64/linux/release
cp bandwidthTest ../../bin/x86_64/linux/release
dev@b76def5159bf:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ ../../bin/x86_64/linux/release/bandwidthTest 
[CUDA Bandwidth Test] - Starting...
Running on...

cudaGetDeviceProperties returned 38
-> no CUDA-capable device is detected
CUDA error at bandwidthTest.cu:255 code=38(cudaErrorNoDevice) "cudaSetDevice(currentDevice)" 
dev@b76def5159bf:~/NVIDIA_CUDA-7.0_Samples/1_Utilities/bandwidthTest$ 
```

#### Discussion
As it follows from the tests in this section, the same version of the Nvidia driver should be installed on the host and on the container in order for a CUDA program to be able to access the GPU device. Moreover, the kernel module should only be installed on the host for this to work.

### Building a Docker container for accessing multiple Nvidia GPUs
The results after a first round of tests show some issues with this approach. First, all existing GPUs must be exposed to the container for the Nvidia driver inside the container to initialize correctly.

Second, inconsistencies appear if trying to concurrently access different GPUs from the host and the container. This suggests that the host driver and the container driver are not aware of each other. For this reason, it is not possible to share the devices.

### Hands-on testing: multi-node
The next round of tests should involve moving from single node to show that an accellerator cluster can be efficiently used with minimal overhead. Research in this direction has already began [5,6], including the analysis of latencies overhead when using SR-IOV-enabled ethernet and Infiniband.

Nevertheless, open questions remain, such as the impact multi-node virtualization might have on a technology like GPU-Direct, which enables DMA from the GPU to the NIC.

### Building a KVM image with GPU Passtrough technology
(...)

### Conclusion
Based on the results presented above, changes in the proprietary Nvidia driver are needed in order for the CUDA platform to be consistently accessible from either KVM or LXC (see also [1]). In this sense, an architecure like Intel's KvmGT [4]
seems as a potential solution for the presented analysis (see  [KvmGT](https://01.org/igvt-g/documentation/kvmgt-full-gpu-virtualization-solution)).

### References
* [1] Walters, John Paul, et al. "GPU Passthrough Performance: A Comparison of KVM, Xen, VMWare ESXi, and LXC for CUDA and OpenCL Applications." *Cloud Computing (CLOUD), 2014 IEEE 7th International Conference on.* IEEE, 2014.
* [2] Younge, Andrew J., et al. "Analysis of virtualization technologies for high performance computing environments." *Cloud Computing (CLOUD), 2011 IEEE International Conference on.* IEEE, 2011.
* [3] Tian, Kun, Yaozu Dong, and David Cowperthwaite. "A full GPU virtualization solution with mediated pass-through." *Proc. USENIX ATC.* 2014.
* [4] Song, Jike. "KVMGT: a Full GPU Virtualization Solution." *KVM Forum*. The Linux Foundation. 2014.
* [5] Dong, Yaozu, et al. "High performance network virtualization with SR-IOV." *Journal of Parallel and Distributed Computing* 72.11 (2012): 1471-1480.
* [6] Jose, Jithin, et al. "SR-IOV support for virtualization on infiniband clusters: Early experience." *Cluster, Cloud and Grid Computing (CCGrid), 2013 13th IEEE/ACM International Symposium on.* IEEE, 2013.
