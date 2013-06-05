#include <math.h>
#include <memory.h>
#include <complex.h>
#include <fftw3.h> 

#include "mri.h"

int mri(
		float* img, 
		fftwf_complex* f, 
		float* mask, 
		float lambda,
		int N1,
		int N2)
{
	int i, j;

	fftwf_complex* f0	= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));
	fftwf_complex* dx	= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));
	fftwf_complex* dy	= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));

	fftwf_complex* dx_new = (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));
	fftwf_complex* dy_new = (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));

	fftwf_complex* dtildex	= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));
	fftwf_complex* dtildey	= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));
	fftwf_complex* u_fft2	= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));
	fftwf_complex* u		= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));

	fftwf_complex* fftmul	= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));
	fftwf_complex* Lap		= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));
	fftwf_complex* diff		= (fftwf_complex*) calloc(N1*N2,sizeof(fftwf_complex));

	float sum = 0;

	for(i=0; i<N1; i++)
		for(j=0; j<N2; j++)
			sum += (SQR(crealf(f(i,j))/N1) + SQR(cimagf(f(i,j))/N1));

	float normFactor = 1.f/sqrtf(sum);
	float scale		 = sqrtf(N1*N2);

	for(i=0; i<N1; i++) {
		for(j=0; j<N2; j++) {
			f(i, j)  = f(i, j)*normFactor;
			f0(i, j) = f(i, j);
		}
	}
	Lap(N1-1, N2-1)	= 0.f;
	Lap(N1-1, 0)	= 1.f; 
	Lap(N1-1, 1)	= 0.f;
	Lap(0, N2-1)	= 1.f;
	Lap(0, 0)		= -4.f; 
	Lap(0, 1)		= 1.f;
	Lap(1, N2-1)	= 0.f;
	Lap(1, 0)		= 1.f; 
	Lap(1, 1)		= 0.f;

	fftwf_plan p0,p1,p3;	

	p0 = fftwf_plan_dft_2d(N1,N2, Lap, Lap, FFTW_FORWARD, FFTW_ESTIMATE);
	p1 = fftwf_plan_dft_2d(N1,N2, diff, diff, FFTW_FORWARD, FFTW_ESTIMATE);
	p3 = fftwf_plan_dft_2d(N1,N2, u_fft2, u, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftwf_execute(p0);
	fftwf_destroy_plan(p0);

	for(i=0;i<N1;i++)
		for(j=0;j<N2;j++)					
			fftmul(i,j) = 1.0/((lambda/Gamma1)*mask(i,j) - Lap(i,j) + Gamma2);

	int OuterIter,iter;
	for(OuterIter= 0; OuterIter<MaxOutIter; OuterIter++) {
		for(iter = 0; iter<MaxIter; iter++) {

			for(i=0;i<N1;i++)	
				for(j=0;j<N2;j++)
					diff(i,j)  = dtildex(i,j)-dtildex(i,(j-1)>=0?(j-1):0) + dtildey(i,j)- dtildey((i-1)>=0?(i-1):0,j) ;

			fftwf_execute(p1);

			for(i=0;i<N1;i++)
				for(j=0;j<N2;j++)
					u_fft2(i,j) = fftmul(i,j)*(f(i,j)*lambda/Gamma1*scale-diff(i,j)+Gamma2*u_fft2(i,j)) ;

			fftwf_execute(p3);
			for(i=0;i<N1;i++)
				for(j=0;j<N2;j++)
					u(i,j)/=N1*N2;			

			for(i=0;i<N1;i++) {
				for(j=0;j<N2;j++) {
					float tmp;
					float Thresh=1.0/Gamma1;

					dx(i,j)     = u(i,j<(N2-1)?(j+1):j)-u(i,j)+dx(i,j)-dtildex(i,j) ;
					dy(i,j)     = u(i<(N1-1)?(i+1):i,j)-u(i,j)+dy(i,j)-dtildey(i,j) ;

					tmp = sqrtf(SQR(crealf(dx(i,j)))+SQR(cimagf(dx(i,j))) + SQR(crealf(dy(i,j)))+SQR(cimagf(dy(i,j))));
					tmp = max(0,tmp-Thresh)/(tmp+(tmp<Thresh));
					dx_new(i,j) =dx(i,j)*tmp;
					dy_new(i,j) =dy(i,j)*tmp;
					dtildex(i,j) = 2*dx_new(i,j) - dx(i,j);
					dtildey(i,j) = 2*dy_new(i,j) - dy(i,j);
					dx(i,j)      = dx_new(i,j);
					dy(i,j)      = dy_new(i,j);
				}
			}
		}
		for(i=0;i<N1;i++) {
			for(j=0;j<N2;j++) {
				f(i,j) += f0(i,j) - mask(i,j)*u_fft2(i,j)/scale;  
			}
		}
	}

	for(i=0; i<N1; i++) {
		for(j=0; j<N2; j++) {
			img(i, j) = sqrt(SQR(crealf(u(i, j))) + SQR(cimagf(u(i, j))));
		}
	}
	fftwf_destroy_plan(p1);
	fftwf_destroy_plan(p3);

	return 0;
}
