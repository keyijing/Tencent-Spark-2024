# include <cuda_runtime.h>
# include <cmath>
# include <iostream>
# include <random>
# include <ctime>
# include <cassert>
using namespace std;
constexpr int BLOCK_SIZE=16;
__global__ void kernel_add1(const float *input_vecs,size_t n,size_t dim,float *__restrict__ output_vecs)
{
	int r=threadIdx.x,j=blockIdx.x*blockDim.x+threadIdx.y;
	if(j>=dim) return;
	output_vecs[r*dim+j]=0;
	for(int i=r;i<n;i+=BLOCK_SIZE)
		output_vecs[r*dim+j]+=input_vecs[i*dim+j];
}
__global__ void kernel_add2(const float *output_vecs,size_t dim,float *__restrict__ output_vec)
{
	int j=blockIdx.x*blockDim.x+threadIdx.x;
	if(j>=dim) return;
	output_vec[j]=0;
	for(int i=0;i<BLOCK_SIZE;i++)
		output_vec[j]+=output_vecs[i*dim+j];
}
void reduce(const float *input_vecs,size_t n,size_t dim,float *output_vec)
{
	size_t input_size=n*dim*sizeof(float),output_size=dim*sizeof(float);
	float *d_input_vecs,*d_output_vecs,*d_output_vec;
	cudaMalloc(&d_input_vecs,input_size);
	cudaMalloc(&d_output_vecs,BLOCK_SIZE*output_size);
	cudaMalloc(&d_output_vec,output_size);
	cudaMemcpy(d_input_vecs,input_vecs,input_size,cudaMemcpyHostToDevice);
	kernel_add1<<<(dim+BLOCK_SIZE-1)/BLOCK_SIZE,dim3(BLOCK_SIZE,BLOCK_SIZE)>>>(d_input_vecs,n,dim,d_output_vecs);
	kernel_add2<<<(dim+BLOCK_SIZE*BLOCK_SIZE-1)/(BLOCK_SIZE*BLOCK_SIZE),BLOCK_SIZE*BLOCK_SIZE>>>(d_output_vecs,dim,d_output_vec);
	cudaMemcpy(output_vec,d_output_vec,output_size,cudaMemcpyDeviceToHost);
	cudaFree(d_input_vecs);
	cudaFree(d_output_vecs);
	cudaFree(d_output_vec);
}
int main()
{
	constexpr int n=4096;
	static float input_vecs[n][n],output_vec[n],real_output_vec[n];
	mt19937 rnd(0);
	uniform_real_distribution<float> rng(0,1);
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			input_vecs[i][j]=rng(rnd),real_output_vec[j]+=input_vecs[i][j];
	clock_t start=clock();
	reduce(&input_vecs[0][0],n,n,output_vec);
	// for(int i=0;i<n;i++)
	// 	for(int j=0;j<n;j++)
	// 		output_vec[j]+=input_vecs[i][j];
	clock_t end=clock();
	for(int i=0;i<n;i++) assert(fabs(output_vec[i]-real_output_vec[i])<1);
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<" s\n";
	return 0;
}