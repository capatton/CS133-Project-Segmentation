 // This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#include <CL/cl.h>

// OpenCL kernel to perform an element-wise addition 
const char* programSource =
"#define SQR(x) (x)*(x)\n"
"#define CUB(x) (x)*(x)*(x)\n"
"#define phi(i,j) phi[(i)*m+(j)]\n"
"#define u(i,j) u[(i)*m+(j)]\n"
"#define curv(i,j) curv[(i)*m+(j)]\n"
"#define epsilon 5e-5f\n"
"\n"
"__kernel\n"
"void segmentation(__global float* contour, __global const float* u, __global int* dataDimensions,\n"
"__global float* curv, __global float* phi) {\n"
"float c1,c2;\n"
"int m = dataDimensions[0];\n"
"int n = dataDimensions[1];\n"
"int MaxIter = dataDimensions[2];\n"
"\n"
"int iter;\n"
"int i, j, k;\n"
"\n"
"float mu = 0.18*255*255;\n"
"float dt = 0.225/mu;\n"
"\n"
"float xcent = (m-1) / 2.0;\n"
"float ycent = (n-1) / 2.0;\n"
"float r;\n"
"if (m < n)\n"
"{\n"
"r = m / 2.0;\n"
"}\n"
"else\n"
"{\n"
"r = n / 2.0;\n"
"}\n"
"\n"
"for(i=0; i < m; i++) {\n"
"for(j=0; j < n; j++) {\n"
"\n"
"float xx = i;\n"
"float yy = j;\n"
"phi(i, j) = sqrt(SQR(xx-xcent) + SQR(yy-ycent)) - r;\n"
"curv(i, j) = 0;\n"
"}\n"
"}\n"
"for(iter=0; iter<MaxIter; iter++) {\n"
"\n"
"float num1 = 0;\n"
"float num2 = 0;\n"
"int den1 = 0;\n"
"int den2 = 0;\n"
"\n"
"for(i=0; i<m; i++) {\n"
"for(j=0; j < n; j++) {\n"
"if(phi(i,j) < 0) {\n"
"num1 += 256*u(i,j);\n"
"den1 += 1;\n"
"}\n"
"else if(phi(i,j) > 0) {\n"
"num2 += 256*u(i,j);\n"
"den2 += 1;\n"
"}\n"
"}\n"
"}\n"
"\n"
"c1 = num1/den1;\n"
"c2 = num2/den2;\n"
"\n"
"for(i=1;i<m-1;i++) {\n"
"for(j=1; j < n-1; j++) {\n"
"float Dx_p = phi(i+1,j) - phi(i,j);\n"
"float Dx_m = phi(i,j) - phi(i-1,j);\n"
"float Dy_p = phi(i,j+1) - phi(i,j);\n"
"float Dy_m = phi(i,j) - phi(i,j-1);\n"
"\n"
"float Dx_0 = (phi(i+1,j) - phi(i-1,j))/2;\n"
"float Dy_0 = (phi(i,j+1) - phi(i,j-1))/2;\n"
"\n"
"float Dxx = Dx_p - Dx_m ;\n"
"float Dyy = Dy_p - Dy_m ;\n"
"\n"
"float Dxy = (phi(i+1,j+1) - phi(i+1,j-1) - phi(i-1,j+1) + phi(i-1,j-1)) / 4;\n"
"\n"
"float Grad = sqrt(Dx_0*Dx_0 + Dy_0*Dy_0);\n"
"float K = (Dx_0*Dx_0*Dyy - 2*Dx_0*Dy_0*Dxy + Dy_0*Dy_0*Dxx) / (CUB(Grad) + epsilon);\n"
"\n"
"curv(i, j) = Grad*(mu*K + SQR(256*u(i,j)-c1) - SQR(256*u(i,j)-c2));\n"
"}\n"
"}\n"
"for(j=0; j < n; j++) {\n"
"curv( 0, j) = curv( 1, j);\n"
"curv(m-1,j) = curv(m-2,j);\n"
"}\n"
"for(i=0; i < m; i++) {\n"
"curv(i, 0 ) = curv(i, 1 );\n"
"curv(i,n-1) = curv(i,n-2);\n"
"}\n"
"for(i=0; i<m; i++) {\n"
"for (j=0; j<n; j++) {\n"
"phi(i, j) += curv(i, j) * dt;\n"
"}\n"
"}\n"
"}\n"
"for(i=1; i<m; i++) {\n"
"for (j=1; j<n; j++) {\n"
"if (phi(i, j)*phi(i-1, j)<0 || phi(i, j)*phi(i, j-1)<0)\n"
"contour[i*n+j] = 0.99;\n"
"else\n"
"contour[i*n+j] = 0;\n"
"}\n"
"}\n"
"\n"
"return;\n"
"}\n"
;


int main(int argc, char **argv) {    
    int MaxIter = 50;

    if(argc<2) 
        return -1;

    char* fname = argv[1];
    char fname_out[50] = "contour.bmp";

    if (argc>2)
        strcpy(fname_out, argv[2]);
    if (argc>3)
        MaxIter = atoi(argv[3]);

    int N1;
    int N2;
    int i, j;

    float *img;
    
    //reads fname, stores the array of floats in img, N1 = width of image, N2 = height of image
    int err = imread(&img, &N1, &N2, fname);
    if (err!=0) return err;



    
    // Elements in each array
    const int elements = N1 * N2;   
    
    // Compute the size of the data 
    size_t datasize = sizeof(float)*elements;

    // Allocate space for input/output data

    //Float *img is u
    float *contour = (float*)calloc(elements, sizeof(float));
    float *curv = (float*)calloc(elements, sizeof(float));
    float *phi = (float*)calloc(elements, sizeof(float));
    int *dataDimensions = (int*)malloc(3*sizeof(int));

    // Init data
    dataDimensions[0] = N1;
    dataDimensions[1] = N2;
    dataDimensions[2] = MaxIter;


    // -------------------------DONT MODIFY SECTION BELOW-------------------------------
    // Use this to check the output of each API call
    cl_int status;  
     
    // Retrieve the number of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
 
    // Allocate enough space for each platform
    cl_platform_id *platforms = NULL;
    platforms = (cl_platform_id*)malloc(
        numPlatforms*sizeof(cl_platform_id));
 
    // Fill in the platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    // Retrieve the number of devices
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, 
        NULL, &numDevices);

    // Allocate enough space for each device
    cl_device_id *devices;
    devices = (cl_device_id*)malloc(
        numDevices*sizeof(cl_device_id));

    // Fill in the devices 
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,        
        numDevices, devices, NULL);

    // Create a context and associate it with the devices
    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL, 
        NULL, &status);

    // Create a command queue and associate it with the device 
    cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(context, devices[0], 0, 
        &status);

    // -----------------------------DONT EDIT SECTION ABOVE THIS-------------------------------


    cl_mem contourBuf;
    contourBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize,                       
       NULL, &status);

    cl_mem imgBuf;
    imgBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize,                        
        NULL, &status);

    cl_mem dataDimensionsBuf;
    dataDimensionsBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * sizeof(int),
        NULL, &status); 

    cl_mem curvBuf;
    curvBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize,
        NULL, &status); 

    cl_mem phiBuf;
    phiBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize,
        NULL, &status); 

    
    status = clEnqueueWriteBuffer(cmdQueue, imgBuf, CL_FALSE, 
        0, datasize, img, 0, NULL, NULL);
    
    status = clEnqueueWriteBuffer(cmdQueue, dataDimensionsBuf, CL_FALSE, 
        0, 3 * sizeof(int), dataDimensions, 0, NULL, NULL);

    status = clEnqueueWriteBuffer(cmdQueue, curvBuf, CL_FALSE, 
        0, datasize, curv, 0, NULL, NULL);

    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(cmdQueue, phiBuf, CL_FALSE, 
        0, datasize, phi, 0, NULL, NULL);

    status = clEnqueueWriteBuffer(cmdQueue, contourBuf, CL_FALSE, 
        0, datasize, contour, 0, NULL, NULL);



    // Create a program with source code
    cl_program program = clCreateProgramWithSource(context, 1, 
        (const char**)&programSource, NULL, &status);

    // Build (compile) the program for the device
    status = clBuildProgram(program, numDevices, devices, 
        NULL, NULL, NULL);

    // Create the vector addition kernel
    cl_kernel kernel;
    kernel = clCreateKernel(program, "segmentation", &status);

    // Associate the input and output buffers with the kernel 
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &contourBuf);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &imgBuf);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dataDimensionsBuf);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &curvBuf);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &phiBuf);

    // Define an index space (global work size) of work 
    // items for execution. A workgroup size (local work size) 
    // is not required, but can be used.
    size_t globalWorkSize[1];   
 
    // There are 'elements' work-items 
    globalWorkSize[0] = 1;

    // Execute the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, 
        globalWorkSize, NULL, 0, NULL, NULL);


    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, contourBuf, CL_TRUE, 0, 
        datasize, contour, 0, NULL, NULL);

    imwrite(contour, N1, N2, fname_out);

    // Free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(contourBuf);
    clReleaseMemObject(imgBuf);
    clReleaseMemObject(dataDimensionsBuf);
    clReleaseMemObject(curvBuf);
    clReleaseMemObject(phiBuf);

    clReleaseContext(context);

    // Free host resources
    free(phi);
    free(curv);
    free(img);
    free(contour);
    free(dataDimensions);
   
    free(platforms);
    free(devices);

    return 0;
}
