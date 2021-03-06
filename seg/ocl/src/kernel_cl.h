const char* kernel_cl = 
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
"phi(i, j) += curv(i, j) codeProfileOpenCL.txt codeProfileOpenMPI.txt codeProfileParallel.txt codeProfile.txt denoise.c denoise.h denoise_main.c denoise_main.h kernel.cl kernel_cl.h kernel.h Makefile mri.c mri.h mri_main.c mri_main.h seg_main.c seg_main.h util.c util.h dt;\n"
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
