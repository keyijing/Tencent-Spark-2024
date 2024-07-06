# include <cuda_runtime.h>
# include <cmath>
# include <iostream>
# include <random>
# include <ctime>
# include <cassert>
using namespace std;
constexpr int BLOCK_SIZE=32;
const dim3 block_dim(BLOCK_SIZE,BLOCK_SIZE);
template<bool fst>
__global__ void kernel_reduce(const float *input_vecs,size_t n,size_t m,float *block_sum)
{
	__shared__ float a[BLOCK_SIZE][BLOCK_SIZE];
	int i=threadIdx.x,j=threadIdx.y,x=blockIdx.x*blockDim.x+i,y=blockIdx.y*blockDim.y+j;
	if(x<n && y<m)
	{
		if constexpr(fst) a[i][j]=input_vecs[y*n+x];
		else a[i][j]=input_vecs[x*m+y];
	}
	else a[i][j]=0;
	__syncthreads();
	if(j%2==0) a[i][j]+=a[i][j+1];
	__syncthreads();
	if(j%4==0) a[i][j]+=a[i][j+2];
	__syncthreads();
	if(j%8==0) a[i][j]+=a[i][j+4];
	__syncthreads();
	if(j%16==0) a[i][j]+=a[i][j+8];
	__syncthreads();
	if(j==0 && x<n) block_sum[x*gridDim.y+blockIdx.y]=a[i][0]+a[i][16];
}
template<bool fst=false>
void reduce_impl(const float *input_vecs,size_t n,size_t m,float *output_vec,float *buf)
{
	if(m<=BLOCK_SIZE)
	{
		kernel_reduce<fst><<<(n-1)/BLOCK_SIZE+1,block_dim>>>(input_vecs,n,m,output_vec);
	}
	else
	{
		dim3 grid_dim((n-1)/BLOCK_SIZE+1,(m-1)/BLOCK_SIZE+1);
		kernel_reduce<fst><<<grid_dim,block_dim>>>(input_vecs,n,m,buf);
		reduce_impl(buf,n,grid_dim.y,output_vec,buf+n*grid_dim.y);
	}
}
void reduce_sum(const float *input_vecs,size_t n,size_t dim,float *output_vec)
{
	size_t input_size=n*dim*sizeof(float),output_size=dim*sizeof(float);
	float *d_input_vecs,*d_output_vec;
	cudaMalloc(&d_input_vecs,input_size);
	cudaMalloc(&d_output_vec,2*((n-1)/BLOCK_SIZE+1)*output_size);
	cudaMemcpy(d_input_vecs,input_vecs,input_size,cudaMemcpyHostToDevice);
	reduce_impl<true>(d_input_vecs,dim,n,d_output_vec,d_output_vec+output_size);
	cudaMemcpy(output_vec,d_output_vec,output_size,cudaMemcpyDeviceToHost);
	cudaFree(d_input_vecs);
	cudaFree(d_output_vec);
}
int main()
{
	constexpr int n=16384,m=16384;
	static float input_vecs[n][m],output_vec[m],real_output_vec[m];
	mt19937 rnd(0);
	uniform_real_distribution<float> rng(0,1);
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			input_vecs[i][j]=rng(rnd),real_output_vec[j]+=input_vecs[i][j];
	clock_t start=clock();
	reduce_sum(&input_vecs[0][0],n,m,output_vec);
	// for(int i=0;i<n;i++)
	// 	for(int j=0;j<m;j++)
	// 		output_vec[j]+=input_vecs[i][j];
	clock_t end=clock();
	for(int i=0;i<m;i++) assert(fabs(output_vec[i]-real_output_vec[i])<1);
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<" s\n";
	return 0;
}