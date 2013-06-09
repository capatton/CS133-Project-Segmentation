#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <sys/time.h>

#include "segmentation.h"
#include "util.h"

int main(int argc, char *argv[]) {

	const char *kernelCode = 
		"#define SQR(x) (x)*(x)\n"
		"define CUB(x) (x)*(x)*(x)\n"
		"#define phi(i,j)	phi[(i)*m+(j)]\n"
		"#define u(i,j)		u[(i)*m+(j)]\n"
		"#define curv(i,j)	curv[(i)*m+(j)]\n"
		"#define epsilon 5e-5f\n";
		
		
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
	int	N2;
	int i, j;

	float *img;
	float *contour;
	
	//reads fname, stores the array of floats in img, N1 = width of image, N2 = height of image
	int err = imread(&img, &N1, &N2, fname);
	if (err!=0) return err;

	contour = (float*)calloc(N1*N2, sizeof(float));

	//contour = output
	segmentation(contour, img, N1, N2, MaxIter);

	imwrite(contour, N1, N2, fname_out);

	return 0;
}
