#define BENCHMARK "OSU MPI Bandwidth Test for OpenCL"
/*
* Copyright (C) 2002-2010 the Network-Based Computing Laboratory
* (NBCL), The Ohio State University.
*
* Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
*/

/*
This program is available under BSD licensing.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

(1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of The Ohio State University nor the names of
their contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* OpenCL modifications by S. Alam and L. Benedicic, CSCS */

#include <CL/cl.h>

/* CLErrorString from SHOC suite http://ft.ornl.gov/doku/shoc/start */

inline const char *CLErrorString(cl_int err)
{
    switch (err)
    {
      case CL_SUCCESS:                         return "CL_SUCCESS";
      case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";
      case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";
      case CL_COMPILER_NOT_AVAILABLE:          return
"CL_COMPILER_NOT_AVAILABLE";
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return
"CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
      case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
      case CL_PROFILING_INFO_NOT_AVAILABLE:    return
"CL_PROFILING_INFO_NOT_AVAILABLE";
      case CL_MEM_COPY_OVERLAP:                return "CL_MEM_COPY_OVERLAP";
      case CL_IMAGE_FORMAT_MISMATCH:           return
"CL_IMAGE_FORMAT_MISMATCH";
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:      return
"CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case CL_BUILD_PROGRAM_FAILURE:           return
"CL_BUILD_PROGRAM_FAILURE";
      case CL_MAP_FAILURE:                     return "CL_MAP_FAILURE";
      case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";
      case CL_INVALID_DEVICE_TYPE:             return "CL_INVALID_DEVICE_TYPE";
      case CL_INVALID_PLATFORM:                return "CL_INVALID_PLATFORM";
      case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";
      case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";
      case CL_INVALID_QUEUE_PROPERTIES:        return
"CL_INVALID_QUEUE_PROPERTIES";
      case CL_INVALID_COMMAND_QUEUE:           return
"CL_INVALID_COMMAND_QUEUE";
      case CL_INVALID_HOST_PTR:                return "CL_INVALID_HOST_PTR";
      case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return
"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case CL_INVALID_IMAGE_SIZE:              return "CL_INVALID_IMAGE_SIZE";
      case CL_INVALID_SAMPLER:                 return "CL_INVALID_SAMPLER";
      case CL_INVALID_BINARY:                  return "CL_INVALID_BINARY";
      case CL_INVALID_BUILD_OPTIONS:           return
"CL_INVALID_BUILD_OPTIONS";
      case CL_INVALID_PROGRAM:                 return "CL_INVALID_PROGRAM";
      case CL_INVALID_PROGRAM_EXECUTABLE:      return
"CL_INVALID_PROGRAM_EXECUTABLE";
      case CL_INVALID_KERNEL_NAME:             return "CL_INVALID_KERNEL_NAME";
      case CL_INVALID_KERNEL_DEFINITION:       return
"CL_INVALID_KERNEL_DEFINITION";
      case CL_INVALID_KERNEL:                  return "CL_INVALID_KERNEL";
      case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";
      case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";
      case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";
      case CL_INVALID_WORK_DIMENSION:          return
"CL_INVALID_WORK_DIMENSION";
      case CL_INVALID_WORK_GROUP_SIZE:         return
"CL_INVALID_WORK_GROUP_SIZE";
      case CL_INVALID_WORK_ITEM_SIZE:          return
"CL_INVALID_WORK_ITEM_SIZE";
      case CL_INVALID_GLOBAL_OFFSET:           return
"CL_INVALID_GLOBAL_OFFSET";
      case CL_INVALID_EVENT_WAIT_LIST:         return
"CL_INVALID_EVENT_WAIT_LIST";
      case CL_INVALID_EVENT:                   return "CL_INVALID_EVENT";
      case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";
      case CL_INVALID_GL_OBJECT:               return "CL_INVALID_GL_OBJECT";
      case CL_INVALID_BUFFER_SIZE:             return "CL_INVALID_BUFFER_SIZE";
      case CL_INVALID_MIP_LEVEL:               return "CL_INVALID_MIP_LEVEL";
      default:                                 return "UNKNOWN";
  }
}


#define err_status(err) \
    {                       \
        if (err != CL_SUCCESS)                  \
           printf ("%s in %s line %d\n",        \
           CLErrorString(err), __FILE__,        \
           __LINE__);                           \
    }


#define MAX_REQ_NUM 1000
#define FIELD_WIDTH 20
#define MAX_ALIGNMENT 65536
#define MAX_MSG_SIZE (1<<22)
#define FLOAT_PRECISION 2
#define MYBUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

//#include <cuda.h>
//#include <cuda_runtime.h>

int loop = 100;
int window_size = 64;
int skip = 10;

int loop_large = 20;
int window_size_large = 64;
int skip_large = 2;
int i;
int large_message_size = 8192;

MPI_Request request[MAX_REQ_NUM];
MPI_Status  reqstat[MAX_REQ_NUM];

int main(int argc, char *argv[])
{
    int myid, numprocs, i, j;
    int size, align_size;

// host buffer
    char *s_buf, *r_buf, *s_buf1, *r_buf1;
    double t_start = 0.0, t_end = 0.0, t = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    align_size = getpagesize();
    assert(align_size <= MAX_ALIGNMENT);

#ifdef PINNED
   // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    err_status(ret);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
            &device_id, &ret_num_devices);
    err_status(ret);

    printf("%d device(s) in %d platform(s)\n",ret_num_devices, ret_num_platforms);
    char cBuffer[1024];
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer,
NULL);
    err_status(ret);
    printf("CL_DEVICE_NAME:       %s\n", cBuffer);

    // Create an OpenCL context
    cl_context context = clCreateContext (NULL,
                                          1, 
                                          &device_id, 
                                          NULL, 
                                          NULL,
                                          &ret);
    err_status(ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue (context, 
                                                           device_id, 
                                                           0,
                                                           &ret);
    err_status(ret);

    // Create memory buffers on the device
    cl_mem s_mem = clCreateBuffer(context, 
                                  CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                  // CL_MEM_COPY_HOST_PTR is only valid with non-NULL pointer
                                  MYBUFSIZE, 
                                  NULL, 
                                  &ret);
    err_status(ret);
    cl_mem r_mem = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 
                                  // CL_MEM_COPY_HOST_PTR is only valid with non-NULL pointer
                                  MYBUFSIZE, 
                                  NULL, 
                                  &ret);
    err_status(ret);

   // pinned memory (blocked call)
   s_buf1 = (char *) clEnqueueMapBuffer(command_queue,
                                        s_mem, 
                                        CL_TRUE,
                                        CL_MAP_WRITE,
                                        0, 
                                        MYBUFSIZE, 
                                        0,
                                        NULL,
                                        NULL,
                                        &ret);
   err_status(ret);
   r_buf1 = (char *) clEnqueueMapBuffer(command_queue,
                                        r_mem, 
                                        CL_TRUE,
                                        CL_MAP_WRITE, 
                                        0, 
                                        MYBUFSIZE, 
                                        0,
                                        NULL, 
                                        NULL, 
                                        &ret);
   err_status(ret);

#else
    if (myid == 0) printf("# Using PAGEABLE host memory!\n");
    s_buf1 = (char*) malloc(MYBUFSIZE);
    r_buf1 = (char*) malloc(MYBUFSIZE);
#endif

    s_buf =
        (char *) (((unsigned long) s_buf1 + (align_size - 1)) /
                  align_size * align_size);
    r_buf =
        (char *) (((unsigned long) r_buf1 + (align_size - 1)) /
                  align_size * align_size);

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_Finalize();

        return EXIT_FAILURE;
    }

    if(myid == 0) {
        fprintf(stdout, "# %s\n", BENCHMARK);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH,
                "Bandwidth (MB/s)");
        fflush(stdout);
    }

    /* Bandwidth test */
    for(size = 1; size <= MAX_MSG_SIZE; size *= 2) {
        /* touch the data */
        for(i = 0; i < size; i++) {
            s_buf[i] = 'a';
            r_buf[i] = 'b';
        }
        //   puts("2");
        if(size > large_message_size) {
            loop = loop_large;
            skip = skip_large;
            window_size = window_size_large;
        }

        if(myid == 0) {
            for(i = 0; i < loop + skip; i++) {
                if(i == skip) {
                    t_start = MPI_Wtime();
                }

                for(j = 0; j < window_size; j++) {
                    MPI_Isend(s_buf, size, MPI_CHAR, 1, 100, MPI_COMM_WORLD,
                            request + j);
                }

                MPI_Waitall(window_size, request, reqstat);
                MPI_Recv(r_buf, 4, MPI_CHAR, 1, 101, MPI_COMM_WORLD,
                        &reqstat[0]);
            }

            t_end = MPI_Wtime();
            // printf("%d %d\n",myid,size);
            t = t_end - t_start;
        }

        else if(myid == 1) {
            for(i = 0; i < loop + skip; i++) {
                for(j = 0; j < window_size; j++) {
                    MPI_Irecv(r_buf, size, MPI_CHAR, 0, 100, MPI_COMM_WORLD,
                            request + j);
                }

                MPI_Waitall(window_size, request, reqstat);
                MPI_Send(s_buf, 4, MPI_CHAR, 0, 101, MPI_COMM_WORLD);
            }
            // printf("%d %d\n",myid,size);
        }

        if(myid == 0) {
            double tmp = size / 1e6 * loop * window_size;

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, tmp / t);
            fflush(stdout);
        }
    }

#ifdef PINNED
//    cudaFree(s_buf1);
//    cudaFree(r_buf1);
//   clReleaseMemObject(s_mem);
//   clReleaseMemObject(r_mem);

#else
    free(s_buf1);
    free(r_buf1);
#endif

    MPI_Finalize();

    return EXIT_SUCCESS;
}

