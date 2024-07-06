# include <cuda_runtime.h>
# include <iostream>
# include <vector>
# include <limits>
# include <random>
# include <ctime>
# include <cassert>
using namespace std;
constexpr int BLOCK_SIZE=256;
__device__ __inline__ void my_swap(int *x,int *y)
{
	int tmp=*x;
	*x=*y;*y=tmp;
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
	nums.resize(n,numeric_limits<int>::max());
	int *d_a;
	cudaMalloc(&d_a,n*sizeof(int));
	cudaMemcpy(d_a,nums.data(),n*sizeof(int),cudaMemcpyHostToDevice);
	for(size_t m=1;m<n;m<<=1)
		for(size_t k=m;k>0;k>>=1)
			kernel_sort<<<(n/2-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_a,n,m,k);
	cudaMemcpy(nums.data(),d_a,sz*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	nums.resize(sz);
}
int main()
{
	constexpr int n=1<<25;
	mt19937 rnd;
	vector<int> vec(n);
	for(int &i:vec) i=rnd();
	clock_t start=clock();
	sort(vec);
	clock_t end=clock();
	assert(is_sorted(vec.begin(),vec.end()));
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"\n";
	return 0;
}