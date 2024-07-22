# include "../cuda_header.h"
# include <thrust/device_ptr.h>
# include <thrust/fill.h>
# include <iostream>
# include <vector>
# include <limits>
# include <random>
# include <cassert>
using namespace std;
constexpr int threads_per_block=1024;
__device__ __inline__ void my_swap(int *x,int *y)
{
	int &&tmp=move(*x);
	*x=move(*y);*y=move(tmp);
}
__global__ void kernel_sort(int *a,size_t n,size_t m,size_t k)
{
	size_t i=blockIdx.x*blockDim.x+threadIdx.x,lo=i&(k-1),hi=i^lo;
	i=(hi<<1)^lo^k;
	if(i<n && (a[i^k]>a[i])^((hi&m)>0)) my_swap(a+(i^k),a+i);
}
void sort(vector<int> &nums)
{
	size_t sz=nums.size(),n=1;
	while(n<sz) n<<=1;
	int *d_a;
	cudaMalloc(&d_a,n*sizeof(int));
	cudaMemcpy(d_a,nums.data(),sz*sizeof(int),cudaMemcpyHostToDevice);
	thrust::fill_n(thrust::device_pointer_cast(d_a)+sz,n-sz,numeric_limits<int>::max());
	for(size_t m=1;m<n;m<<=1)
		for(size_t k=m;k>0;k>>=1)
			kernel_sort<<<(n/2-1)/threads_per_block+1,threads_per_block>>>(d_a,n,m,k);
	cudaMemcpy(nums.data(),d_a,sz*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_a);
}
int main()
{
	constexpr int n=4e7;
	static vector<int> vec(n);
	mt19937 rnd;
	for(int &i:vec) i=rnd();
	cout<<"CPU: "<<PERF_CPU([]{
		auto tmp=vec;
		sort(tmp.begin(),tmp.end());
	})<<" sec\n";
	cout<<"GPU: "<<PERF_GPU([]{
		sort(vec);
	})<<" sec\n";
	assert(is_sorted(vec.begin(),vec.end()));
	return 0;
}