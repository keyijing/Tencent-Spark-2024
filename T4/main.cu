# include "../cuda_header.h"
# include <iostream>
# include <vector>
# include <random>
# include <cassert>
using namespace std;
constexpr unsigned full_mask=(unsigned)-1,warp_size=32,threads_per_block=warp_size*warp_size;
__global__ void kernel_init(const int *a,size_t n,size_t *acc,size_t *block_sum)
{
	__shared__ size_t warp_sum[warp_size];
	size_t i=threadIdx.x,id=blockIdx.x*blockDim.x+i,v,v0,v1;
	unsigned mask=__ballot_sync(full_mask,id<n && a[id]);
	v=__popc(mask<<(31-i%warp_size));
	if((i+1)%warp_size==0) warp_sum[i/warp_size]=v;
	__syncthreads();
	if(i<warp_size)
	{
		v0=warp_sum[i];
		v1=__shfl_sync(full_mask,v0,0,2);
		if(i&1) v0+=v1;
		v1=__shfl_sync(full_mask,v0,1,4);
		if(i&2) v0+=v1;
		v1=__shfl_sync(full_mask,v0,3,8);
		if(i&4) v0+=v1;
		v1=__shfl_sync(full_mask,v0,7,16);
		if(i&8) v0+=v1;
		v1=__shfl_sync(full_mask,v0,15,32);
		if(i&16) v0+=v1;
		warp_sum[i]=v0;
	}
	__syncthreads();
	if(id<n) acc[id]=v+(i<warp_size?0:warp_sum[i/warp_size-1]);
	if(i==warp_size-1) block_sum[blockIdx.x]=v0;
}
__global__ void kernel_acc(const size_t *a,size_t n,size_t *acc,size_t *block_sum)
{
	__shared__ size_t warp_sum[warp_size];
	size_t i=threadIdx.x,id=blockIdx.x*blockDim.x+i,v=(id<n?a[id]:0),v0,v1;
	v1=__shfl_sync(full_mask,v,0,2);
	if(i&1) v+=v1;
	v1=__shfl_sync(full_mask,v,1,4);
	if(i&2) v+=v1;
	v1=__shfl_sync(full_mask,v,3,8);
	if(i&4) v+=v1;
	v1=__shfl_sync(full_mask,v,7,16);
	if(i&8) v+=v1;
	v1=__shfl_sync(full_mask,v,15,32);
	if(i&16) v+=v1;
	if((i+1)%warp_size==0) warp_sum[i/warp_size]=v;
	__syncthreads();
	if(i<warp_size)
	{
		v0=warp_sum[i];
		v1=__shfl_sync(full_mask,v0,0,2);
		if(i&1) v0+=v1;
		v1=__shfl_sync(full_mask,v0,1,4);
		if(i&2) v0+=v1;
		v1=__shfl_sync(full_mask,v0,3,8);
		if(i&4) v0+=v1;
		v1=__shfl_sync(full_mask,v0,7,16);
		if(i&8) v0+=v1;
		v1=__shfl_sync(full_mask,v0,15,32);
		if(i&16) v0+=v1;
		warp_sum[i]=v0;
	}
	__syncthreads();
	if(id<n) acc[id]=v+(i<warp_size?0:warp_sum[i/warp_size-1]);
	if(i==warp_size-1) block_sum[blockIdx.x]=v0;
}
__global__ void kernel_add_block(size_t n,size_t *acc,const size_t *block_acc)
{
	size_t i=(blockIdx.x+1)*blockDim.x+threadIdx.x;
	if(i<n) acc[i]+=block_acc[blockIdx.x];
}
void accmulate(const size_t *a,size_t n,size_t *acc,size_t *buf)
{
	if(n<=threads_per_block)
	{
		kernel_acc<<<1,threads_per_block>>>(a,n,acc,buf);
	}
	else
	{
		size_t grid_dim=(n-1)/threads_per_block+1;
		kernel_acc<<<grid_dim,threads_per_block>>>(a,n,acc,buf);
		accmulate(buf,grid_dim,buf+grid_dim,buf+2*grid_dim);
		kernel_add_block<<<grid_dim-1,threads_per_block>>>(n,acc,buf+grid_dim);
	}
}
__global__ void kernel_replace(const int *a,size_t n,const size_t *acc,int *output)
{
	size_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n && a[i]) output[acc[i]-1]=a[i];
}
size_t debubble(vector<int> &data)
{
	size_t n=data.size(),m;
	int *d_a,*d_output;
	size_t *d_acc;
	unsigned grid_dim=(n-1)/threads_per_block+1;
	cudaMalloc(&d_a,n*sizeof(int));
	cudaMalloc(&d_output,n*sizeof(int));
	cudaMalloc(&d_acc,(n+3*grid_dim)*sizeof(size_t));
	cudaMemcpy(d_a,data.data(),n*sizeof(int),cudaMemcpyHostToDevice);
	size_t *d_buf=d_acc+n;
	kernel_init<<<grid_dim,threads_per_block>>>(d_a,n,d_acc,d_buf);
	if(n>threads_per_block)
	{
		accmulate(d_buf,grid_dim,d_buf+grid_dim,d_buf+2*grid_dim);
		kernel_add_block<<<grid_dim-1,threads_per_block>>>(n,d_acc,d_buf+grid_dim);
	}
	cudaMemcpy(&m,d_acc+(n-1),sizeof(size_t),cudaMemcpyDeviceToHost);
	kernel_replace<<<grid_dim,threads_per_block>>>(d_a,n,d_acc,d_output);
	cudaMemset(d_output+m,0,(n-m)*sizeof(int));
	cudaMemcpy(data.data(),d_output,n*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_output);
	cudaFree(d_acc);
	return m;
}
int main()
{
	static constexpr int n=1<<26;
	static size_t m=0;
	static vector<int> vec(n),ans(n);
	mt19937 rnd;
	bernoulli_distribution r1(0.5);
	for(int &i:vec) i=r1(rnd)*rnd();
	cout<<"CPU: "<<PERF_CPU([]{
		for(int i:vec)
			if(i) ans[m++]=i;
	})<<" sec\n";
	cout<<"GPU: "<<PERF_GPU([]{
		debubble(vec);
	})<<" sec\n";
	assert(vec==ans);
	return 0;
}