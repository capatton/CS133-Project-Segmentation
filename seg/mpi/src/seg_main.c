#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <sys/time.h>

#include "segmentation.h"
#include "util.h"

int main(int argc, char *argv[]) {

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
	
	//reads fname, stores the array of floats in img, N1 = width of image, N2 = height of image
	int err = imread(&img, &width, &height, fname);
	if (err!=0) return err;

	contour = (float*)calloc(width*height, sizeof(float));

	// ---------------------------- START OF SEGMENTATION FUNCTION ------------------------------------
	float* curv	= (float*)calloc(width*height,sizeof(float));
	float* phi	= (float*)calloc(width*height,sizeof(float));

	float c1,c2;

	int iter;
	int i, j, k;	

	float mu = 0.18*255*255;
	float dt = 0.225/mu;

	float xcent = (width-1) / 2.0;
	float ycent = (height-1) / 2.0;
	float r = fmin(width,height) / 2.0;

	for(i=0; i < width; i++) {
		for(j=0; j < height; j++) {

			float xx = i;
			float yy = j;
			phi(i, j)	= sqrtf(SQR(xx-xcent) + SQR(yy-ycent)) - r;
			curv(i, j)	= 0;
		}
	}
	for(iter=0; iter<MaxIter; iter++) {

		float num1 = 0;
		float num2 = 0;
		int   den1 = 0;
		int   den2 = 0;

		for(i=0; i<width; i++) {
			for(j=0; j < height; j++) {
				if(phi(i,j) < 0) {
					num1 += 256*img(i,j);
					den1 +=  1;
				}
				else if(phi(i,j) > 0) {
					num2  += 256*img(i,j);
					den2  += 1;
				}
			}
		}

		c1 = num1/den1;
		c2 = num2/den2;

		for(i=1;i<width-1;i++) {
			for(j=1; j < n-1; j++) {
				float Dx_p = phi(i+1,j) - phi(i,j);
				float Dx_m = phi(i,j) - phi(i-1,j);
				float Dy_p = phi(i,j+1) - phi(i,j);
				float Dy_m = phi(i,j) - phi(i,j-1);

				float Dx_0 = (phi(i+1,j) - phi(i-1,j))/2;
				float Dy_0 = (phi(i,j+1) - phi(i,j-1))/2;

				float Dxx = Dx_p - Dx_m ;
				float Dyy = Dy_p - Dy_m ;

				float Dxy = (phi(i+1,j+1) - phi(i+1,j-1) - phi(i-1,j+1) + phi(i-1,j-1)) / 4;

				float Grad	= sqrtf(Dx_0*Dx_0 + Dy_0*Dy_0);
				float K		= (Dx_0*Dx_0*Dyy - 2*Dx_0*Dy_0*Dxy + Dy_0*Dy_0*Dxx) / (CUB(Grad) + epsilon);

				curv(i, j) = Grad*(mu*K + SQR(256*img(i,j)-c1) - SQR(256*img(i,j)-c2));
			}
		}
		for(j=0; j < height; j++) {
			curv( 0, j) = curv( 1, j);
			curv(width-1,j) = curv(width-2,j);
		}
		for(i=0; i < width; i++) {
			curv(i, 0 ) = curv(i, 1 );
			curv(i,height-1) = curv(i,height-2);
		}
		for(i=0; i<width; i++) {
			for (j=0; j<height; j++) {
				phi(i, j) += curv(i, j) * dt;
			}
		}
	}
	for(i=1; i<width; i++) {
		for (j=1; j<height; j++) {
			if (phi(i, j)*phi(i-1, j)<0 || phi(i, j)*phi(i, j-1)<0) 
				contour[i*height+j] = 0.99;
			else 
				contour[i*height+j] = 0;
		}
	}

	free(phi);
	free(curv);

	// ---------------------------- END OF SEGMENTATION FUNCTION ------------------------------------

	imwrite(contour, N1, N2, fname_out);

	return 0;
}
