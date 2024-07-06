# include <cuda_runtime.h>
# include <iostream>
# include <random>
# include <ctime>
# include <cassert>
using namespace std;
constexpr int BLOCK_SIZE=16;
__global__ void kernel_matmul(const float *A,const float *B,size_t M,size_t N,size_t K,float *__restrict__ output)
{
	int i=threadIdx.x,j=threadIdx.y,x=blockIdx.x*blockDim.x+i,z=blockIdx.y*blockDim.y+j;
	for(int y=0;y<N;y+=BLOCK_SIZE)
	{
		__shared__ float d_A[BLOCK_SIZE][BLOCK_SIZE],d_B[BLOCK_SIZE][BLOCK_SIZE];
		d_A[i][j]=(x<M && y+j<N?A[x*N+y+j]:0);
		d_B[i][j]=(y+i<N && z<K?B[(y+i)*K+z]:0);
		__syncthreads();
		if(x<M && z<K)
			for(int k=0;k<BLOCK_SIZE;k++)
				output[x*K+z]+=d_A[i][k]*d_B[k][j];
		__syncthreads();
	}
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
	kernel_matmul<<<dim3((m-1)/BLOCK_SIZE+1,(k-1)/BLOCK_SIZE+1),dim3(BLOCK_SIZE,BLOCK_SIZE)>>>(d_A,d_B,m,n,k,d_C);
	cudaMemcpy(output,d_C,size_C,cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
int main()
{
	constexpr int n=1024;
	static float A[n][n],B[n][n],C[n][n],real_C[n][n];
	mt19937 rnd(0);
	uniform_real_distribution<float> rng(0,1);
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			A[i][j]=rng(rnd),B[i][j]=rng(rnd);
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			for(int k=0;k<n;k++)
				real_C[i][k]+=A[i][j]*B[j][k];
	clock_t start=clock();
	matmul(&A[0][0],&B[0][0],n,n,n,&C[0][0]);
	clock_t end=clock();
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			assert(fabs(C[i][j]-real_C[i][j])<1e-4);
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"\n";
	return 0;
}