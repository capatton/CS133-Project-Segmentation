#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <sys/time.h>

#include "util.h"
#include <mpi.h>

#define SQR(x) (x)*(x)
#define CUB(x) (x)*(x)*(x)

#define epsilon 5e-5f
const int MASTER = 0;

int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);

	int pNum, pRank;
	MPI_Comm_size(MPI_COMM_WORLD, &pNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &pRank);

	int MaxIter = 50;

	if(argc<2) 
		return -1;

	char* fname = argv[1];
	char fname_out[50] = "contour.bmp";

	if (argc>2)
		strcpy(fname_out, argv[2]);
	if (argc>3)
		MaxIter = atoi(argv[3]);

	int width;
	int	height;
	int i, j;

	float *img;
	float *contour;
	
	int err;
	//reads fname, stores the array of floats in img, N1 = width of image, N2 = height of image
	if (pRank == MASTER)
	{
		err = imread(&img, &width, &height, fname); // err ignored for now
	}
	//give all other processes the width and height
	MPI_Bcast(&width, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&height, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	
	const int IMG_AMT_PER_PROCESSOR = width * height / pNum + 1;

	// scatter img from master to all other processes
	float *img_local = (float*)calloc(IMG_AMT_PER_PROCESSOR, sizeof(float));
	MPI_Scatter(img, IMG_AMT_PER_PROCESSOR, MPI_FLOAT, img_local, IMG_AMT_PER_PROCESSOR, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

	contour = (float*)calloc(IMG_AMT_PER_PROCESSOR, sizeof(float));

	// ---------------------------- START OF SEGMENTATION FUNCTION ------------------------------------
	// curv and phi each need the amount of space needed for the image, plus a little buffer on the left and the right
	// to grab the column from the neighboring process
	const float BUFFER_SPACE = 2 * height;
	float* curv	= (float*)calloc(IMG_AMT_PER_PROCESSOR + BUFFER_SPACE,sizeof(float));
	float* phi	= (float*)calloc(IMG_AMT_PER_PROCESSOR + BUFFER_SPACE,sizeof(float));

	float c1,c2;

	int iter;
	int k;	

	float mu = 0.18*255*255;
	float dt = 0.225/mu;

	float xcent = (width-1) / 2.0;
	float ycent = (height-1) / 2.0;
	float r = fmin(width,height) / 2.0;

	const int SECTION_WIDTH = width / pNum;

	//You wrote the notes for this.  I think this is right.
	for(i=0; i < SECTION_WIDTH; i++) {
		for(j=0; j < height; j++) {

			float xx =  i+(pRank*width)/pNum;
			float yy =  j;
			phi[i*width + j]	= sqrtf(SQR(xx-xcent) + SQR(yy-ycent)) - r;
			curv[i*width + j]	= 0;
		}
	}

	for(iter=0; iter<MaxIter; iter++) {

		float num1 = 0;
		float num2 = 0;
		int   den1 = 0;
		int   den2 = 0;

		// Each process calculates their own num1/den1/.... then reduces to one value on rank 0, then rank 0 calculates c1/c2, 
		// then broadcasts that
		for(i=0; i<SECTION_WIDTH; i++) {
			for(j=0; j < height; j++) {
				if(phi[i*width + j] < 0) {
					num1 += 256*img[i*width + j];
					den1 +=  1;
				}
				else if(phi[i*width + j]  > 0) {
					num2  += 256*img[i*width + j];
					den2  += 1;
				}
			}
		}

		float num1_out;
		float num2_out;
		int den1_out;
		int den2_out;
		MPI_Reduce((void*)&num1, (void*)&num1_out, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);	
		MPI_Reduce((void*)&num2, (void*)&num2_out, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);	
		MPI_Reduce((void*)&den1, (void*)&den1_out, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);	
		MPI_Reduce((void*)&den2, (void*)&den2_out, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);	


		if (pRank == MASTER){
			c1 = num1_out/den1_out;
			c2 = num2_out/den2_out;
		}

		MPI_Bcast(&c1, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&c2, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

		// I THINK ITS CORRECT UP TO HERE
		
		for(i=1;i<SECTION_WIDTH-1;i++) {
			for(j=1; j < height-1; j++) {
				float Dx_p = phi[(i+1)*width + j] - phi[i*width + j];
				float Dx_m = phi[i*width + j] - phi[(i-1)*width + j];
				float Dy_p = phi[i*width + j + 1] - phi[i*width + j];
				float Dy_m = phi[i*width + j] - phi[i*width + j - 1];

				float Dx_0 = (phi[(i+1)*width + j] - phi[(i-1)*width + j])/2;
				float Dy_0 = (phi[i*width + j + 1] - phi[i*width + j - 1])/2;

				float Dxx = Dx_p - Dx_m ;
				float Dyy = Dy_p - Dy_m ;

				float Dxy = (phi[(i+1)*width + j+1] - phi[(i+1)*width + j-1]- phi[(i-1)*width + j+1] + phi[(i-1)*width + j-1]) / 4;

				float Grad	= sqrtf(Dx_0*Dx_0 + Dy_0*Dy_0);
				float K		= (Dx_0*Dx_0*Dyy - 2*Dx_0*Dy_0*Dxy + Dy_0*Dy_0*Dxx) / (CUB(Grad) + epsilon);

				curv[i*width + j] = Grad*(mu*K + SQR(256*img[i*width + j]-c1) - SQR(256*img[i*width + j]-c2));
			}
		}
		for(j=0; j < height; j++) {
			curv[j] = curv[width + j];
			curv[(width - 1)*width + j] = curv[(width-2)*width + j];
		}
		for(i=0; i < SECTION_WIDTH; i++) {
			curv[i*width] = curv[i*width + 1];
			curv[i*width + (height - 1)] = curv[i*width + (height - 2)];
		}
		for(i=0; i<SECTION_WIDTH; i++) {
			for (j=0; j<height; j++) {
				phi[i*width + j] += curv[i*width + j] * dt;
			}
		}
	}

	for(i=1; i<SECTION_WIDTH; i++) {
		for (j=1; j<height; j++) {
			if (phi[i*width + j]*phi[(i-1)*width + j]<0 || phi[i*width + j]*phi[i*width + j-1]<0) 
				contour[i*height+j] = 0.99;
			else 
				contour[i*height+j] = 0;
		}
	}

	free(phi);
	free(curv);

	// ---------------------------- END OF SEGMENTATION FUNCTION ------------------------------------
	float* contour_out = (float*)calloc(width*height, sizeof(float));

	//gather the contour results
	MPI_Gather(contour, SECTION_WIDTH, MPI_FLOAT, contour_out, SECTION_WIDTH, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

	if (pRank == MASTER){
		imwrite(contour_out, width, height, fname_out);
	}

	MPI_Finalize();
	return 0;
}
