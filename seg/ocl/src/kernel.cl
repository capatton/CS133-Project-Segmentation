#define SQR(x) (x)*(x)
#define CUB(x) (x)*(x)*(x)
#define phi(i,j)    phi[(i)*m+(j)]
#define u(i,j)      u[(i)*m+(j)]
#define curv(i,j)   curv[(i)*m+(j)]
#define epsilon 5e-5f

__kernel
void segmentation(__global float* contour, __global const float* u, __global int* dataDimensions, 
__global float* curv, __global float* phi) {
float c1,c2;
int m = dataDimensions[0];
int n = dataDimensions[1];
int MaxIter = dataDimensions[2];

int iter;
int i, j, k;

float mu = 0.18*255*255;
float dt = 0.225/mu;

float xcent = (m-1) / 2.0;
float ycent = (n-1) / 2.0;
float r;
if (m < n)
{
	r = m / 2.0;
}
else
{
	r = n / 2.0;
}

for(i=0; i < m; i++) {
for(j=0; j < n; j++) {

float xx = i;
float yy = j;
phi(i, j) = sqrt(SQR(xx-xcent) + SQR(yy-ycent)) - r;
curv(i, j) = 0;
}
}
for(iter=0; iter<MaxIter; iter++) {

float num1 = 0;
float num2 = 0;
int den1 = 0;
int den2 = 0;

for(i=0; i<m; i++) {
for(j=0; j < n; j++) {
if(phi(i,j) < 0) {
num1 += 256*u(i,j);
den1 += 1;
}
else if(phi(i,j) > 0) {
num2 += 256*u(i,j);
den2 += 1;
}
}
}

c1 = num1/den1;
c2 = num2/den2;

for(i=1;i<m-1;i++) {
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

float Grad = sqrt(Dx_0*Dx_0 + Dy_0*Dy_0);
float K = (Dx_0*Dx_0*Dyy - 2*Dx_0*Dy_0*Dxy + Dy_0*Dy_0*Dxx) / (CUB(Grad) + epsilon);

curv(i, j) = Grad*(mu*K + SQR(256*u(i,j)-c1) - SQR(256*u(i,j)-c2));
}
}
for(j=0; j < n; j++) {
curv( 0, j) = curv( 1, j);
curv(m-1,j) = curv(m-2,j);
}
for(i=0; i < m; i++) {
curv(i, 0 ) = curv(i, 1 );
curv(i,n-1) = curv(i,n-2);
}
for(i=0; i<m; i++) {
for (j=0; j<n; j++) {
phi(i, j) += curv(i, j) * dt;
}
}
}
for(i=1; i<m; i++) {
for (j=1; j<n; j++) {
if (phi(i, j)*phi(i-1, j)<0 || phi(i, j)*phi(i, j-1)<0)
contour[i*n+j] = 0.99;
else
contour[i*n+j] = 0;
}
}

return;
}
;