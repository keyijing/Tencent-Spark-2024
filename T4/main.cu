# include <cuda_runtime.h>
# include <iostream>
# include <vector>
# include <random>
# include <ctime>
# include <cassert>
using namespace std;
constexpr int BLOCK_SIZE=1024;
__global__ void kernel_init(const int *a,size_t n,size_t *c)
{
	size_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n) c[i]=(bool)a[i];
}
__global__ void kernel_acc(const size_t *input,size_t n,size_t *acc,size_t *block_sum)
{
	__shared__ size_t a[BLOCK_SIZE];
	size_t i=threadIdx.x,x=blockIdx.x*blockDim.x+i;
	a[i]=(x<n?input[x]:0);
	__syncthreads();
	i<<=1;
	for(int j=1;j<BLOCK_SIZE;j<<=1)
	{
		if(i<BLOCK_SIZE) a[i|j]+=a[i|(j-1)];
		size_t t=i&(j<<1);
		i^=t;i^=t>>1;
		__syncthreads();
	}
	if(x<n) acc[x]=a[i];
	if(i==blockDim.x-1) block_sum[blockIdx.x]=a[i];
}
__global__ void kernel_add_block(size_t n,size_t *acc,const size_t *block_acc)
{
	size_t i=(blockIdx.x+1)*blockDim.x+threadIdx.x;
	if(i<n) acc[i]+=block_acc[blockIdx.x];
}
void accmulate(const size_t *a,size_t n,size_t *acc,size_t *buf)
{
	if(n<=BLOCK_SIZE)
	{
		kernel_acc<<<1,BLOCK_SIZE>>>(a,n,acc,buf);
	}
	else
	{
		size_t m=(n-1)/BLOCK_SIZE+1;
		kernel_acc<<<m,BLOCK_SIZE>>>(a,n,acc,buf);
		accmulate(buf,m,buf+m,buf+2*m);
		kernel_add_block<<<m-1,BLOCK_SIZE>>>(n,acc,buf+m);
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
	size_t *d_c,*d_acc;
	cudaMalloc(&d_a,n*sizeof(int));
	cudaMalloc(&d_output,n*sizeof(int));
	cudaMalloc(&d_c,n*sizeof(size_t));
	cudaMalloc(&d_acc,(n+3*((n-1)/BLOCK_SIZE+1))*sizeof(size_t));
	cudaMemcpy(d_a,data.data(),n*sizeof(int),cudaMemcpyHostToDevice);
	kernel_init<<<(n-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_a,n,d_c);
	accmulate(d_c,n,d_acc,d_acc+n);
	cudaMemcpy(&m,d_acc+(n-1),sizeof(size_t),cudaMemcpyDeviceToHost);
	kernel_replace<<<(n-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_a,n,d_acc,d_output);
	cudaMemset(d_output+m,0,(n-m)*sizeof(int));
	cudaMemcpy(data.data(),d_output,n*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_output);
	cudaFree(d_c);
	cudaFree(d_acc);
	return m;
}
int main()
{
	constexpr int n=1<<26;
	size_t m=0;
	vector<int> vec(n),ans(n);
	mt19937 rnd;
	bernoulli_distribution r1(0.5);
	for(int &i:vec) i=r1(rnd)*rnd();
	for(int i:vec)
		if(i) ans[m++]=i;
	clock_t start=clock();
	assert(debubble(vec)==m);
	clock_t end=clock();
	assert(vec==ans);
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"\n";
	return 0;
}