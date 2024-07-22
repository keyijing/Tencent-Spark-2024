# include "../cuda_header.h"
# include <iostream>
# include <cmath>
# include <random>
# include <cassert>
using namespace std;
constexpr unsigned full_mask=(unsigned)-1,warp_size=32,warp_num=32;
template<bool fst>
__global__ void kernel_reduce(const float *input_vecs,size_t n,size_t m,float *block_sum)
{
	size_t i=threadIdx.y,j=threadIdx.x,x,y;
	float v;
	if constexpr(fst)
	{
		__shared__ float a[32][33];
		x=blockIdx.y*blockDim.y+j;
		y=blockIdx.x*blockDim.x+i;
		a[j][i]=(x<n && y<m?input_vecs[y*n+x]:0);
		__syncthreads();
		v=a[i][j];
		x=blockIdx.y*blockDim.y+i;
	}
	else
	{
		x=blockIdx.y*blockDim.y+i;
		y=blockIdx.x*blockDim.x+j;
		v=(x<n && y<m?input_vecs[x*m+y]:0);
	}
	v+=__shfl_down_sync(full_mask,v,16,32);
	v+=__shfl_down_sync(full_mask,v,8,16);
	v+=__shfl_down_sync(full_mask,v,4,8);
	v+=__shfl_down_sync(full_mask,v,2,4);
	v+=__shfl_down_sync(full_mask,v,1,2);
	if(j==0 && x<n) block_sum[x*gridDim.x+blockIdx.x]=v;
}
template<bool fst=false>
void reduce_impl(const float *input_vecs,size_t n,size_t m,float *output_vec,float *buf)
{
	dim3 grid_dim((m-1)/warp_size+1,(n-1)/warp_num+1),block_dim(warp_size,warp_num);
	if(m<=warp_size)
	{
		kernel_reduce<fst><<<grid_dim,block_dim>>>(input_vecs,n,m,output_vec);
	}
	else
	{
		kernel_reduce<fst><<<grid_dim,block_dim>>>(input_vecs,n,m,buf);
		reduce_impl(buf,n,grid_dim.x,output_vec,buf+n*grid_dim.x);
	}
}
void reduce_sum(const float *input_vecs,size_t n,size_t dim,float *output_vec)
{
	size_t input_size=n*dim*sizeof(float),output_size=dim*sizeof(float);
	float *d_input_vecs,*d_output_vec;
	cudaMalloc(&d_input_vecs,input_size);
	cudaMalloc(&d_output_vec,2*((n-1)/warp_size+1)*output_size);
	cudaMemcpy(d_input_vecs,input_vecs,input_size,cudaMemcpyHostToDevice);
	reduce_impl<true>(d_input_vecs,dim,n,d_output_vec,d_output_vec+output_size);
	cudaMemcpy(output_vec,d_output_vec,output_size,cudaMemcpyDeviceToHost);
	cudaFree(d_input_vecs);
	cudaFree(d_output_vec);
}
int main()
{
	static constexpr int n=2e4,m=2e4;
	static float input_vecs[n][m],output_vec[m],real_output_vec[m];
	mt19937 rnd(0);
	uniform_real_distribution<float> rng(0,1);
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			input_vecs[i][j]=rng(rnd);
	cout<<"CPU: "<<PERF_CPU([]{
		for(int i=0;i<n;i++)
		{
			float *p=input_vecs[i],*q=real_output_vec;
			for(int j=0;j<m;j++,p++,q++) *q+=*p;
		}
	})<<" sec\n";
	cout<<"GPU: "<<PERF_GPU([]{
		reduce_sum(&input_vecs[0][0],n,m,output_vec);
	})<<" sec\n";
	for(int i=0;i<m;i++)
		assert(fabs(output_vec[i]-real_output_vec[i])<0.01*real_output_vec[i]);
	return 0;
}