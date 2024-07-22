# include "../cuda_header.h"
# include <iostream>
# include <random>
# include <ctime>
# include <cassert>
using namespace std;
constexpr unsigned block_size=32;
__global__ void kernel_matmul(const float *A,const float *B,size_t M,size_t N,size_t K,float *output)
{
	size_t i=threadIdx.y,j=threadIdx.x,x,y;
	float sum=0;
	for(size_t y0=0;y0<N;y0+=block_size)
	{
		__shared__ float a[block_size][block_size],b[block_size][block_size];
		x=blockIdx.y*blockDim.y+j;
		y=y0+i;
		a[i][j]=(x<M && y<N?A[x*N+y]:0);
		x=y0+i;
		y=blockIdx.x*blockDim.x+j;
		b[i][j]=(x<N && y<K?B[x*K+y]:0);
		__syncthreads();
		for(size_t k=0;k<block_size;k++) sum+=a[k][i]*b[k][j];
		__syncthreads();
	}
	x=blockIdx.y*blockDim.y+i;
	y=blockIdx.x*blockDim.x+j;
	if(x<M && y<K) output[x*K+y]=sum;
}
void matmul(const float *A,const float *B,size_t m,size_t n,size_t k,float *output)
{
	size_t size_A=m*n*sizeof(float),size_B=n*k*sizeof(float),size_C=m*k*sizeof(float);
	float *d_A,*d_B,*d_C;
	cudaMalloc(&d_A,size_A);
	cudaMalloc(&d_B,size_B);
	cudaMalloc(&d_C,size_C);
	cudaMemcpy(d_A,A,size_A,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,size_B,cudaMemcpyHostToDevice);
	dim3 grid_dim((k-1)/block_size+1,(m-1)/block_size+1),block_dim(block_size,block_size);
	kernel_matmul<<<grid_dim,block_dim>>>(d_A,d_B,m,n,k,d_C);
	cudaMemcpy(output,d_C,size_C,cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
int main()
{
	static constexpr int n=4096;
	static float A[n][n],B[n][n],C[n][n],real_C[n][n];
	mt19937 rnd(0);
	uniform_real_distribution<float> rng(0,1);
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			A[i][j]=rng(rnd),B[i][j]=rng(rnd);
	cout<<"CPU: "<<PERF_CPU([]{
		for(int i=0;i<n;i++)
			for(int j=0;j<n;j++)
				for(int k=0;k<n;k++)
					real_C[i][k]+=A[i][j]*B[j][k];
	})<<" sec\n";
	cout<<"GPU: "<<PERF_GPU([]{
		matmul(&A[0][0],&B[0][0],n,n,n,&C[0][0]);
	})<<" sec\n";
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			assert(fabs(C[i][j]-real_C[i][j])<0.01*fabs(real_C[i][j]));
	return 0;
}